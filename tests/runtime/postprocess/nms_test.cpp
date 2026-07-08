#include "mw/infer/runtime/postprocess/nms.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "mw/infer/runtime/tensor/tensor_allocator.h"

#if defined(MW_INFER_HAS_CUDA_NMS)
#include <cuda_runtime_api.h>

#include "mw/infer/runtime/postprocess/cuda_nms.h"
#endif

namespace mw::infer {
namespace {

Tensor MakeCpuTensor(std::vector<int64_t> shape, const std::vector<float>& data,
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

std::vector<int64_t> CopyCpuIndices(const Tensor& tensor) {
  EXPECT_EQ(tensor.device().type, DeviceType::kCpu);
  EXPECT_EQ(tensor.data_type(), DataType::kInt64);
  const auto* data = static_cast<const int64_t*>(tensor.data());
  return std::vector<int64_t>(data, data + tensor.element_count());
}

std::vector<float> TestBoxes() {
  return {
      0.0F,  0.0F,  10.0F, 10.0F,  //
      1.0F,  1.0F,  11.0F, 11.0F,  //
      20.0F, 20.0F, 30.0F, 30.0F,
  };
}

std::vector<float> TestScores() { return {0.9F, 0.8F, 0.7F}; }

TEST(NmsTest, CpuNmsSuppressesOverlappingBoxes) {
  Tensor boxes = MakeCpuTensor({3, 4}, TestBoxes(), "boxes");
  Tensor scores = MakeCpuTensor({3}, TestScores(), "scores");

  Tensor keep = CpuNms(boxes, scores, 0.5F);

  EXPECT_EQ(keep.name(), "nms_indices");
  EXPECT_EQ(keep.shape(), std::vector<int64_t>({2}));
  EXPECT_EQ(CopyCpuIndices(keep), std::vector<int64_t>({0, 2}));
}

TEST(NmsTest, DispatchesCpuNmsByInputDevice) {
  Tensor boxes = MakeCpuTensor({3, 4}, TestBoxes(), "boxes");
  Tensor scores = MakeCpuTensor({3}, TestScores(), "scores");

  Tensor keep = Nms(boxes, scores, 0.5F);

  EXPECT_EQ(CopyCpuIndices(keep), std::vector<int64_t>({0, 2}));
}

TEST(NmsTest, RespectsMaxOutputBoxes) {
  Tensor boxes = MakeCpuTensor({3, 4}, TestBoxes(), "boxes");
  Tensor scores = MakeCpuTensor({3}, TestScores(), "scores");
  NmsOptions options;
  options.iou_threshold = 0.5F;
  options.max_output_boxes = 1;

  Tensor keep = CpuNms(boxes, scores, options);

  EXPECT_EQ(keep.shape(), std::vector<int64_t>({1}));
  EXPECT_EQ(CopyCpuIndices(keep), std::vector<int64_t>({0}));
}

TEST(NmsTest, RejectsInvalidInputs) {
  Tensor boxes = MakeCpuTensor({3, 4}, TestBoxes(), "boxes");
  Tensor scores = MakeCpuTensor({3}, TestScores(), "scores");

  NmsOptions options;
  options.iou_threshold = -0.1F;
  EXPECT_THROW(static_cast<void>(CpuNms(boxes, scores, options)),
               std::invalid_argument);

  Tensor bad_scores = MakeCpuTensor({2}, {0.9F, 0.8F}, "scores");
  EXPECT_THROW(static_cast<void>(CpuNms(boxes, bad_scores, 0.5F)),
               std::invalid_argument);
}

#if defined(MW_INFER_HAS_CUDA_NMS)

bool HasUsableCudaDevice() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

Tensor MakeCudaTensor(std::vector<int64_t> shape,
                      const std::vector<float>& data, std::string name = {}) {
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

std::vector<int64_t> CopyCudaIndices(const Tensor& tensor) {
  EXPECT_EQ(tensor.device().type, DeviceType::kCuda);
  EXPECT_EQ(tensor.data_type(), DataType::kInt64);
  std::vector<int64_t> values(tensor.element_count());
  EXPECT_EQ(cudaMemcpy(values.data(), tensor.data(), tensor.bytes(),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  return values;
}

TEST(NmsTest, CudaNmsSuppressesOverlappingBoxes) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "CUDA NMS is unavailable";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

  Tensor boxes = MakeCudaTensor({3, 4}, TestBoxes(), "boxes");
  Tensor scores = MakeCudaTensor({3}, TestScores(), "scores");

  Tensor keep = CudaNms(boxes, scores, 0.5F);

  EXPECT_TRUE(CudaNmsAvailable());
  EXPECT_EQ(keep.name(), "nms_indices");
  EXPECT_EQ(keep.shape(), std::vector<int64_t>({2}));
  EXPECT_EQ(CopyCudaIndices(keep), std::vector<int64_t>({0, 2}));
}

TEST(NmsTest, DispatchesCudaNmsByInputDevice) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "CUDA NMS is unavailable";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

  Tensor boxes = MakeCudaTensor({3, 4}, TestBoxes(), "boxes");
  Tensor scores = MakeCudaTensor({3}, TestScores(), "scores");

  Tensor keep = Nms(boxes, scores, 0.5F);

  EXPECT_EQ(CopyCudaIndices(keep), std::vector<int64_t>({0, 2}));
}

#endif

}  // namespace
}  // namespace mw::infer
