#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "mw/infer/runtime/postprocess/softmax.h"
#include "mw/infer/runtime/postprocess/topk.h"

#if defined(MW_INFER_HAS_CUDA_POSTPROCESS)
#include <cuda_runtime_api.h>
#endif

namespace mw::infer {
namespace {

Tensor MakeCpuFloatTensor(std::vector<int64_t> shape,
                          const std::vector<float>& data,
                          std::string name = {}) {
  TensorDesc desc;
  desc.info.name = std::move(name);
  desc.info.data_type = DataType::kFloat32;
  desc.info.shape = std::move(shape);
  desc.device = Device{DeviceType::kCpu, 0};
  Tensor tensor = Tensor::Allocate(std::move(desc));
  std::memcpy(tensor.data(), data.data(), tensor.bytes());
  return tensor;
}

Tensor MakeCpuInt64Tensor(std::vector<int64_t> shape,
                          const std::vector<int64_t>& data,
                          std::string name = {}) {
  TensorDesc desc;
  desc.info.name = std::move(name);
  desc.info.data_type = DataType::kInt64;
  desc.info.shape = std::move(shape);
  desc.device = Device{DeviceType::kCpu, 0};
  Tensor tensor = Tensor::Allocate(std::move(desc));
  std::memcpy(tensor.data(), data.data(), tensor.bytes());
  return tensor;
}

std::vector<float> CopyCpuFloats(const Tensor& tensor) {
  EXPECT_EQ(tensor.device().type, DeviceType::kCpu);
  EXPECT_EQ(tensor.data_type(), DataType::kFloat32);
  const auto* data = static_cast<const float*>(tensor.data());
  return std::vector<float>(data, data + tensor.element_count());
}

std::vector<int64_t> CopyCpuIndices(const Tensor& tensor) {
  EXPECT_EQ(tensor.device().type, DeviceType::kCpu);
  EXPECT_EQ(tensor.data_type(), DataType::kInt64);
  const auto* data = static_cast<const int64_t*>(tensor.data());
  return std::vector<int64_t>(data, data + tensor.element_count());
}

void ExpectRowSumsToOne(const std::vector<float>& values, int64_t rows,
                        int64_t columns) {
  for (int64_t row = 0; row < rows; ++row) {
    float sum = 0.0F;
    for (int64_t column = 0; column < columns; ++column) {
      sum += values[static_cast<std::size_t>(row * columns + column)];
    }
    EXPECT_NEAR(sum, 1.0F, 1.0e-5F);
  }
}

TEST(ClassificationPostprocessTest, SoftmaxNormalizesSingleBatchLogits) {
  Tensor logits = MakeCpuFloatTensor({3}, {1.0F, 2.0F, 3.0F}, "logits");

  Tensor probabilities = Softmax(logits);

  EXPECT_EQ(probabilities.name(), "softmax");
  EXPECT_EQ(probabilities.shape(), std::vector<int64_t>({3}));
  std::vector<float> values = CopyCpuFloats(probabilities);
  ExpectRowSumsToOne(values, 1, 3);
  EXPECT_LT(values[0], values[1]);
  EXPECT_LT(values[1], values[2]);
}

TEST(ClassificationPostprocessTest, SoftmaxNormalizesMultiBatchLogits) {
  Tensor logits = MakeCpuFloatTensor(
      {2, 3}, {1.0F, 2.0F, 3.0F, 3.0F, 2.0F, 1.0F}, "logits");

  Tensor probabilities = Softmax(logits);

  EXPECT_EQ(probabilities.shape(), std::vector<int64_t>({2, 3}));
  std::vector<float> values = CopyCpuFloats(probabilities);
  ExpectRowSumsToOne(values, 2, 3);
  EXPECT_LT(values[0], values[1]);
  EXPECT_LT(values[1], values[2]);
  EXPECT_GT(values[3], values[4]);
  EXPECT_GT(values[4], values[5]);
}

TEST(ClassificationPostprocessTest, TopKSelectsScoresAndIndices) {
  Tensor scores = MakeCpuFloatTensor({4}, {0.1F, 0.9F, 0.4F, 0.9F}, "scores");

  TopKResult result = TopK(scores, 2);

  EXPECT_EQ(result.scores.name(), "topk_scores");
  EXPECT_EQ(result.indices.name(), "topk_indices");
  EXPECT_EQ(result.scores.shape(), std::vector<int64_t>({2}));
  EXPECT_EQ(result.indices.shape(), std::vector<int64_t>({2}));
  EXPECT_EQ(CopyCpuFloats(result.scores), std::vector<float>({0.9F, 0.9F}));
  EXPECT_EQ(CopyCpuIndices(result.indices), std::vector<int64_t>({1, 3}));
}

TEST(ClassificationPostprocessTest, TopKSupportsMultiBatchScores) {
  Tensor scores = MakeCpuFloatTensor(
      {2, 4}, {0.1F, 0.8F, 0.4F, 0.9F, 0.7F, 0.2F, 0.7F, 0.1F}, "scores");

  TopKResult result = TopK(scores, 2);

  EXPECT_EQ(result.scores.shape(), std::vector<int64_t>({2, 2}));
  EXPECT_EQ(result.indices.shape(), std::vector<int64_t>({2, 2}));
  EXPECT_EQ(CopyCpuFloats(result.scores),
            std::vector<float>({0.9F, 0.8F, 0.7F, 0.7F}));
  EXPECT_EQ(CopyCpuIndices(result.indices), std::vector<int64_t>({3, 1, 0, 2}));
}

TEST(ClassificationPostprocessTest, DispatchesHostTensorByInputDevice) {
  Tensor logits =
      MakeCpuFloatTensor({2, 3}, {1.0F, 2.0F, 3.0F, 3.0F, 2.0F, 1.0F});

  Tensor probabilities = Softmax(logits);
  TopKResult result = TopK(probabilities, 1);

  EXPECT_EQ(probabilities.device().type, DeviceType::kCpu);
  EXPECT_EQ(result.indices.device().type, DeviceType::kCpu);
  EXPECT_EQ(CopyCpuIndices(result.indices), std::vector<int64_t>({2, 0}));
}

TEST(ClassificationPostprocessTest, RejectsInvalidInputs) {
  Tensor scores =
      MakeCpuFloatTensor({2, 3}, {1.0F, 2.0F, 3.0F, 3.0F, 2.0F, 1.0F});
  Tensor bad_rank = MakeCpuFloatTensor({1, 1, 3}, {1.0F, 2.0F, 3.0F});
  Tensor bad_type = MakeCpuInt64Tensor({3}, {1, 2, 3});

  EXPECT_THROW(static_cast<void>(Softmax(Tensor{})), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(Softmax(bad_rank)), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(Softmax(bad_type)), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(TopK(scores, 0)), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(TopK(scores, 4)), std::invalid_argument);
}

#if defined(MW_INFER_HAS_CUDA_POSTPROCESS)

bool HasUsableCudaDevice() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

Tensor MakeCudaFloatTensor(std::vector<int64_t> shape,
                           const std::vector<float>& data,
                           std::string name = {}) {
  TensorDesc desc;
  desc.info.name = std::move(name);
  desc.info.data_type = DataType::kFloat32;
  desc.info.shape = std::move(shape);
  desc.device = Device{DeviceType::kCuda, 0};
  Tensor tensor = Tensor::Allocate(std::move(desc));
  EXPECT_EQ(cudaMemcpy(tensor.data(), data.data(), tensor.bytes(),
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  return tensor;
}

std::vector<float> CopyCudaFloats(const Tensor& tensor) {
  EXPECT_EQ(tensor.device().type, DeviceType::kCuda);
  EXPECT_EQ(tensor.data_type(), DataType::kFloat32);
  std::vector<float> values(tensor.element_count());
  EXPECT_EQ(cudaMemcpy(values.data(), tensor.data(), tensor.bytes(),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  return values;
}

std::vector<int64_t> CopyCudaIndices(const Tensor& tensor) {
  EXPECT_EQ(tensor.device().type, DeviceType::kCuda);
  EXPECT_EQ(tensor.data_type(), DataType::kInt64);
  std::vector<int64_t> values(tensor.element_count());
  EXPECT_EQ(cudaMemcpy(values.data(), tensor.data(), tensor.bytes(),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  return values;
}

TEST(ClassificationPostprocessTest, SoftmaxDispatchesCudaTensor) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "CUDA postprocess is unavailable";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  Tensor logits = MakeCudaFloatTensor(
      {2, 3}, {1.0F, 2.0F, 3.0F, 3.0F, 2.0F, 1.0F}, "logits");

  Tensor probabilities = Softmax(logits);

  EXPECT_EQ(probabilities.name(), "softmax");
  EXPECT_EQ(probabilities.shape(), std::vector<int64_t>({2, 3}));
  std::vector<float> values = CopyCudaFloats(probabilities);
  ExpectRowSumsToOne(values, 2, 3);
  EXPECT_LT(values[0], values[1]);
  EXPECT_LT(values[1], values[2]);
  EXPECT_GT(values[3], values[4]);
  EXPECT_GT(values[4], values[5]);
}

TEST(ClassificationPostprocessTest, TopKDispatchesCudaTensor) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "CUDA postprocess is unavailable";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  Tensor scores = MakeCudaFloatTensor(
      {2, 4}, {0.1F, 0.8F, 0.4F, 0.9F, 0.7F, 0.2F, 0.7F, 0.1F}, "scores");

  TopKResult result = TopK(scores, 2);

  EXPECT_EQ(result.scores.name(), "topk_scores");
  EXPECT_EQ(result.indices.name(), "topk_indices");
  EXPECT_EQ(result.scores.shape(), std::vector<int64_t>({2, 2}));
  EXPECT_EQ(result.indices.shape(), std::vector<int64_t>({2, 2}));
  EXPECT_EQ(CopyCudaFloats(result.scores),
            std::vector<float>({0.9F, 0.8F, 0.7F, 0.7F}));
  EXPECT_EQ(CopyCudaIndices(result.indices),
            std::vector<int64_t>({3, 1, 0, 2}));
}

TEST(ClassificationPostprocessTest, DispatchesDeviceTensorByInputDevice) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "CUDA postprocess is unavailable";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  Tensor logits =
      MakeCudaFloatTensor({2, 3}, {1.0F, 2.0F, 3.0F, 3.0F, 2.0F, 1.0F});

  Tensor probabilities = Softmax(logits);
  TopKResult result = TopK(probabilities, 1);

  EXPECT_EQ(probabilities.device().type, DeviceType::kCuda);
  EXPECT_EQ(result.indices.device().type, DeviceType::kCuda);
  EXPECT_EQ(CopyCudaIndices(result.indices), std::vector<int64_t>({2, 0}));
}

#endif

}  // namespace
}  // namespace mw::infer
