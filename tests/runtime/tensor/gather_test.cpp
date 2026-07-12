#include "mw/infer/runtime/tensor/gather.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mw/infer/runtime/execution_stream.h"

#if defined(MW_INFER_HAS_CUDA_TENSOR_OPS)
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
  if (tensor.bytes() > 0) {
    std::memcpy(tensor.data(), data.data(), tensor.bytes());
  }
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
  if (tensor.bytes() > 0) {
    std::memcpy(tensor.data(), data.data(), tensor.bytes());
  }
  return tensor;
}

std::vector<float> CopyCpuFloats(const Tensor& tensor) {
  EXPECT_EQ(tensor.device().type, DeviceType::kCpu);
  EXPECT_EQ(tensor.data_type(), DataType::kFloat32);
  const auto* data = static_cast<const float*>(tensor.data());
  return std::vector<float>(data, data + tensor.element_count());
}

std::vector<int64_t> CopyCpuInt64(const Tensor& tensor) {
  EXPECT_EQ(tensor.device().type, DeviceType::kCpu);
  EXPECT_EQ(tensor.data_type(), DataType::kInt64);
  const auto* data = static_cast<const int64_t*>(tensor.data());
  return std::vector<int64_t>(data, data + tensor.element_count());
}

TEST(GatherRowsTest, GathersRowsFromRankOneTensor) {
  Tensor scores = MakeCpuFloatTensor({4}, {0.1F, 0.8F, 0.4F, 0.9F}, "scores");
  Tensor indices = MakeCpuInt64Tensor({2}, {3, 1}, "indices");

  Tensor selected = scores.GatherRows(indices);

  EXPECT_EQ(selected.name(), "scores_gathered");
  EXPECT_EQ(selected.shape(), std::vector<int64_t>({2}));
  EXPECT_EQ(CopyCpuFloats(selected), std::vector<float>({0.9F, 0.8F}));
}

TEST(GatherRowsTest, GathersRowsFromRankTwoTensor) {
  Tensor boxes = MakeCpuFloatTensor({3, 4},
                                    {
                                        0.0F,
                                        0.0F,
                                        10.0F,
                                        10.0F,  //
                                        1.0F,
                                        1.0F,
                                        11.0F,
                                        11.0F,  //
                                        20.0F,
                                        20.0F,
                                        30.0F,
                                        30.0F,
                                    },
                                    "boxes");
  Tensor indices = MakeCpuInt64Tensor({2}, {0, 2}, "indices");

  Tensor selected = GatherRows(boxes, indices);

  EXPECT_EQ(selected.shape(), std::vector<int64_t>({2, 4}));
  EXPECT_EQ(CopyCpuFloats(selected),
            std::vector<float>(
                {0.0F, 0.0F, 10.0F, 10.0F, 20.0F, 20.0F, 30.0F, 30.0F}));
}

TEST(GatherRowsTest, PreservesInputDataType) {
  Tensor values = MakeCpuInt64Tensor({3, 2}, {1, 2, 3, 4, 5, 6}, "values");
  Tensor indices = MakeCpuInt64Tensor({2}, {2, 0}, "indices");

  Tensor selected = GatherRows(values, indices);

  EXPECT_EQ(selected.name(), "values_gathered");
  EXPECT_EQ(selected.shape(), std::vector<int64_t>({2, 2}));
  EXPECT_EQ(CopyCpuInt64(selected), std::vector<int64_t>({5, 6, 1, 2}));
}

TEST(GatherRowsTest, AllowsEmptyIndexTensor) {
  Tensor values = MakeCpuFloatTensor(
      {3, 2}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}, "values");
  Tensor indices = MakeCpuInt64Tensor({0}, {}, "indices");

  Tensor selected = GatherRows(values, indices);

  EXPECT_FALSE(selected.empty());
  EXPECT_EQ(selected.shape(), std::vector<int64_t>({0, 2}));
  EXPECT_EQ(selected.element_count(), 0U);
  EXPECT_EQ(selected.bytes(), 0U);
  EXPECT_EQ(CopyCpuFloats(selected), std::vector<float>({}));
}

TEST(GatherRowsTest, RejectsInvalidInputs) {
  Tensor data = MakeCpuFloatTensor({3}, {1.0F, 2.0F, 3.0F}, "data");
  Tensor indices = MakeCpuInt64Tensor({2}, {0, 2}, "indices");
  Tensor bad_type = MakeCpuFloatTensor({2}, {0.0F, 1.0F}, "indices");
  Tensor bad_rank = MakeCpuInt64Tensor({1, 2}, {0, 1}, "indices");
  Tensor out_of_range = MakeCpuInt64Tensor({1}, {3}, "indices");

  EXPECT_THROW(static_cast<void>(GatherRows(Tensor{}, indices)),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(GatherRows(data, Tensor{})),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(GatherRows(data, bad_type)),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(GatherRows(data, bad_rank)),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(GatherRows(data, out_of_range)),
               std::invalid_argument);
}

#if defined(MW_INFER_HAS_CUDA_TENSOR_OPS)

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
  if (tensor.bytes() > 0) {
    EXPECT_EQ(cudaMemcpy(tensor.data(), data.data(), tensor.bytes(),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
  }
  return tensor;
}

Tensor MakeCudaInt64Tensor(std::vector<int64_t> shape,
                           const std::vector<int64_t>& data,
                           std::string name = {}) {
  TensorDesc desc;
  desc.info.name = std::move(name);
  desc.info.data_type = DataType::kInt64;
  desc.info.shape = std::move(shape);
  desc.device = Device{DeviceType::kCuda, 0};
  Tensor tensor = Tensor::Allocate(std::move(desc));
  if (tensor.bytes() > 0) {
    EXPECT_EQ(cudaMemcpy(tensor.data(), data.data(), tensor.bytes(),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
  }
  return tensor;
}

std::vector<float> CopyCudaFloats(const Tensor& tensor) {
  EXPECT_EQ(tensor.device().type, DeviceType::kCuda);
  EXPECT_EQ(tensor.data_type(), DataType::kFloat32);
  std::vector<float> values(tensor.element_count());
  if (tensor.bytes() > 0) {
    EXPECT_EQ(cudaMemcpy(values.data(), tensor.data(), tensor.bytes(),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
  }
  return values;
}

TEST(GatherRowsTest, DispatchesCudaTensorByInputDevice) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "CUDA tensor ops are unavailable";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  Tensor boxes = MakeCudaFloatTensor({3, 4},
                                     {
                                         0.0F,
                                         0.0F,
                                         10.0F,
                                         10.0F,  //
                                         1.0F,
                                         1.0F,
                                         11.0F,
                                         11.0F,  //
                                         20.0F,
                                         20.0F,
                                         30.0F,
                                         30.0F,
                                     },
                                     "boxes");
  Tensor indices = MakeCudaInt64Tensor({2}, {0, 2}, "indices");

  Tensor selected = GatherRows(boxes, indices);

  EXPECT_EQ(selected.name(), "boxes_gathered");
  EXPECT_EQ(selected.device().type, DeviceType::kCuda);
  EXPECT_EQ(selected.shape(), std::vector<int64_t>({2, 4}));
  EXPECT_EQ(CopyCudaFloats(selected),
            std::vector<float>(
                {0.0F, 0.0F, 10.0F, 10.0F, 20.0F, 20.0F, 30.0F, 30.0F}));
}

TEST(GatherRowsTest, UsesProvidedCudaExecutionStream) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "CUDA tensor ops are unavailable";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  Tensor values = MakeCudaFloatTensor({3}, {1.0F, 2.0F, 3.0F}, "values");
  Tensor indices = MakeCudaInt64Tensor({2}, {2, 0}, "indices");
  ExecutionStream stream(Device{DeviceType::kCuda, 0});

  Tensor selected =
      GatherRows(values, indices, TensorAllocator::Default(), &stream);

  EXPECT_EQ(CopyCudaFloats(selected), std::vector<float>({3.0F, 1.0F}));
}

TEST(GatherRowsTest, RejectsCudaOutOfRangeIndex) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "CUDA tensor ops are unavailable";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  Tensor data = MakeCudaFloatTensor({3}, {1.0F, 2.0F, 3.0F}, "data");
  Tensor indices = MakeCudaInt64Tensor({1}, {3}, "indices");

  EXPECT_THROW(static_cast<void>(GatherRows(data, indices)),
               std::invalid_argument);
}

#endif

}  // namespace
}  // namespace mw::infer
