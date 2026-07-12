#include "mw/infer/runtime/postprocess/nms.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include "mw/infer/runtime/execution_stream.h"
#include "mw/infer/runtime/tensor/tensor_allocator.h"

#if defined(MW_INFER_HAS_CUDA_POSTPROCESS)
#include <cuda_runtime_api.h>
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
  if (tensor.bytes() > 0) {
    std::memcpy(tensor.data(), data.data(), tensor.bytes());
  }
  return tensor;
}

std::vector<int64_t> CopyCpuIndices(const Tensor& tensor) {
  EXPECT_EQ(tensor.device().type, DeviceType::kCpu);
  EXPECT_EQ(tensor.data_type(), DataType::kInt64);
  const auto* data = static_cast<const int64_t*>(tensor.data());
  return std::vector<int64_t>(data, data + tensor.element_count());
}

std::vector<float> CopyCpuFloats(const Tensor& tensor) {
  EXPECT_EQ(tensor.device().type, DeviceType::kCpu);
  EXPECT_EQ(tensor.data_type(), DataType::kFloat32);
  const auto* data = static_cast<const float*>(tensor.data());
  return std::vector<float>(data, data + tensor.element_count());
}

std::vector<float> TestBoxes() {
  return {
      0.0F,  0.0F,  10.0F, 10.0F,  //
      1.0F,  1.0F,  11.0F, 11.0F,  //
      20.0F, 20.0F, 30.0F, 30.0F,
  };
}

std::vector<float> TestScores() { return {0.9F, 0.8F, 0.7F}; }

TEST(NmsTest, SuppressesOverlappingBoxes) {
  Tensor boxes = MakeCpuTensor({3, 4}, TestBoxes(), "boxes");
  Tensor scores = MakeCpuTensor({3}, TestScores(), "scores");

  Tensor keep = Nms(boxes, scores, 0.5F);

  EXPECT_EQ(keep.name(), "nms_indices");
  EXPECT_EQ(keep.shape(), std::vector<int64_t>({2}));
  EXPECT_EQ(CopyCpuIndices(keep), std::vector<int64_t>({0, 2}));
}

TEST(NmsTest, DispatchesHostTensorByInputDevice) {
  Tensor boxes = MakeCpuTensor({3, 4}, TestBoxes(), "boxes");
  Tensor scores = MakeCpuTensor({3}, TestScores(), "scores");

  Tensor keep = Nms(boxes, scores, 0.5F);

  EXPECT_EQ(CopyCpuIndices(keep), std::vector<int64_t>({0, 2}));
}

TEST(NmsTest, AcceptsMatchingCpuExecutionStream) {
  Tensor boxes = MakeCpuTensor({3, 4}, TestBoxes(), "boxes");
  Tensor scores = MakeCpuTensor({3}, TestScores(), "scores");
  ExecutionStream stream(Device{DeviceType::kCpu, 0});

  Tensor keep =
      Nms(boxes, scores, 0.5F, 0.0F, 0, TensorAllocator::Default(), &stream);

  EXPECT_EQ(CopyCpuIndices(keep), std::vector<int64_t>({0, 2}));
}

TEST(NmsTest, RespectsMaxOutputBoxes) {
  Tensor boxes = MakeCpuTensor({3, 4}, TestBoxes(), "boxes");
  Tensor scores = MakeCpuTensor({3}, TestScores(), "scores");
  Tensor keep = Nms(boxes, scores, 0.5F, 0.0F, 1);

  EXPECT_EQ(keep.shape(), std::vector<int64_t>({1}));
  EXPECT_EQ(CopyCpuIndices(keep), std::vector<int64_t>({0}));
}

TEST(NmsTest, AllowsEmptyInputs) {
  Tensor boxes = MakeCpuTensor({0, 4}, {}, "boxes");
  Tensor scores = MakeCpuTensor({0}, {}, "scores");

  Tensor keep = Nms(boxes, scores, 0.5F);

  EXPECT_FALSE(keep.empty());
  EXPECT_EQ(keep.shape(), std::vector<int64_t>({0}));
  EXPECT_EQ(keep.element_count(), 0U);
  EXPECT_EQ(keep.bytes(), 0U);
  EXPECT_EQ(CopyCpuIndices(keep), std::vector<int64_t>({}));
}

TEST(NmsTest, IndicesGatherSelectedRows) {
  Tensor boxes = MakeCpuTensor({3, 4}, TestBoxes(), "boxes");
  Tensor scores = MakeCpuTensor({3}, TestScores(), "scores");

  Tensor keep = Nms(boxes, scores, 0.5F);
  Tensor selected_boxes = boxes.GatherRows(keep);
  Tensor selected_scores = scores.GatherRows(keep);

  EXPECT_EQ(selected_boxes.shape(), std::vector<int64_t>({2, 4}));
  EXPECT_EQ(selected_scores.shape(), std::vector<int64_t>({2}));
  EXPECT_EQ(CopyCpuFloats(selected_boxes),
            std::vector<float>(
                {0.0F, 0.0F, 10.0F, 10.0F, 20.0F, 20.0F, 30.0F, 30.0F}));
  EXPECT_EQ(CopyCpuFloats(selected_scores), std::vector<float>({0.9F, 0.7F}));
}

BatchNmsOptions BatchOptions(int64_t max_detections = 5) {
  BatchNmsOptions options;
  options.score_threshold = 0.25F;
  options.iou_threshold = 0.5F;
  options.max_detections = max_detections;
  return options;
}

void ExpectBatchResult(const BatchNmsResult& result,
                       const std::vector<int64_t>& counts,
                       const std::vector<float>& scores,
                       const std::vector<int64_t>& class_ids,
                       const std::vector<int64_t>& indices) {
  EXPECT_EQ(result.counts.CopyToHostVector<int64_t>(), counts);
  EXPECT_EQ(result.scores.CopyToHostVector<float>(), scores);
  EXPECT_EQ(result.class_ids.CopyToHostVector<int64_t>(), class_ids);
  EXPECT_EQ(result.indices.CopyToHostVector<int64_t>(), indices);
}

TEST(BatchNmsTest, KeepsClassesAndBatchesIndependentWithStablePadding) {
  const std::vector<float> one_batch_boxes = TestBoxes();
  std::vector<float> boxes = one_batch_boxes;
  boxes.insert(boxes.end(), one_batch_boxes.begin(), one_batch_boxes.end());
  Tensor box_tensor = MakeCpuTensor({2, 3, 4}, boxes, "boxes");
  Tensor score_tensor = MakeCpuTensor(
      {2, 3, 2},
      {0.9F, 0.9F, 0.8F, 0.7F, 0.6F, 0.1F,
       0.4F, 0.3F, 0.95F, 0.2F, 0.5F, 0.1F},
      "scores");

  BatchNmsResult result =
      BatchNms(box_tensor, score_tensor, BatchOptions(4));

  EXPECT_EQ(result.boxes.shape(), std::vector<int64_t>({2, 4, 4}));
  ExpectBatchResult(result, {3, 3},
                    {0.9F, 0.9F, 0.6F, 0.0F,
                     0.95F, 0.5F, 0.3F, 0.0F},
                    {0, 1, 0, -1, 0, 0, 1, -1},
                    {0, 0, 2, -1, 1, 2, 0, -1});
}

TEST(BatchNmsTest, ClassAgnosticUsesBestClassAndSuppressesAcrossClasses) {
  Tensor boxes = MakeCpuTensor({1, 3, 4}, TestBoxes(), "boxes");
  Tensor scores = MakeCpuTensor(
      {1, 3, 2}, {0.9F, 0.9F, 0.8F, 0.95F, 0.7F, 0.1F}, "scores");
  BatchNmsOptions options = BatchOptions(3);
  options.class_agnostic = true;

  BatchNmsResult result = BatchNms(boxes, scores, options);

  ExpectBatchResult(result, {2}, {0.95F, 0.7F, 0.0F}, {1, 0, -1},
                    {1, 2, -1});
}

TEST(BatchNmsTest, IncludesThresholdBoundariesAndIgnoresInvalidValues) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  Tensor boxes = MakeCpuTensor(
      {1, 5, 4},
      {0.0F, 0.0F, 2.0F, 2.0F, 0.0F, 0.0F, 1.0F, 2.0F,
       3.0F, 3.0F, 2.0F, 4.0F, nan, 0.0F, 1.0F, 1.0F,
       3.0F, 3.0F, 4.0F, 4.0F},
      "boxes");
  Tensor scores =
      MakeCpuTensor({1, 5, 1}, {0.9F, 0.8F, 0.99F, 0.99F, 0.5F}, "scores");
  BatchNmsOptions options = BatchOptions(5);
  options.score_threshold = 0.5F;

  BatchNmsResult result = BatchNms(boxes, scores, options);

  ExpectBatchResult(result, {3}, {0.9F, 0.8F, 0.5F, 0.0F, 0.0F},
                    {0, 0, 0, -1, -1}, {0, 1, 4, -1, -1});
}

TEST(BatchNmsTest, SupportsZeroBatchAndZeroCandidates) {
  BatchNmsResult zero_batch = BatchNms(
      MakeCpuTensor({0, 3, 4}, {}), MakeCpuTensor({0, 3, 2}, {}),
      BatchOptions(2));
  EXPECT_EQ(zero_batch.counts.shape(), std::vector<int64_t>({0}));
  EXPECT_EQ(zero_batch.boxes.shape(), std::vector<int64_t>({0, 2, 4}));

  BatchNmsResult zero_candidates = BatchNms(
      MakeCpuTensor({2, 0, 4}, {}), MakeCpuTensor({2, 0, 2}, {}),
      BatchOptions(2));
  ExpectBatchResult(zero_candidates, {0, 0}, {0.0F, 0.0F, 0.0F, 0.0F},
                    {-1, -1, -1, -1}, {-1, -1, -1, -1});
}

TEST(NmsTest, RejectsInvalidInputs) {
  Tensor boxes = MakeCpuTensor({3, 4}, TestBoxes(), "boxes");
  Tensor scores = MakeCpuTensor({3}, TestScores(), "scores");

  EXPECT_THROW(static_cast<void>(Nms(boxes, scores, -0.1F)),
               std::invalid_argument);

  Tensor bad_scores = MakeCpuTensor({2}, {0.9F, 0.8F}, "scores");
  EXPECT_THROW(static_cast<void>(Nms(boxes, bad_scores, 0.5F)),
               std::invalid_argument);

  Tensor batch_boxes = MakeCpuTensor({1, 3, 4}, TestBoxes());
  Tensor empty_classes = MakeCpuTensor({1, 3, 0}, {});
  EXPECT_THROW(static_cast<void>(BatchNms(batch_boxes, empty_classes)),
               std::invalid_argument);
  BatchNmsOptions bad_options;
  bad_options.max_detections = 0;
  EXPECT_THROW(static_cast<void>(
                   BatchNms(batch_boxes, MakeCpuTensor({1, 3, 1}, TestScores()),
                            bad_options)),
               std::invalid_argument);
}

#if defined(MW_INFER_HAS_CUDA_POSTPROCESS)

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
  if (tensor.bytes() > 0) {
    EXPECT_EQ(cudaMemcpy(tensor.data(), data.data(), tensor.bytes(),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
  }
  return tensor;
}

std::vector<int64_t> CopyCudaIndices(const Tensor& tensor) {
  EXPECT_EQ(tensor.device().type, DeviceType::kCuda);
  EXPECT_EQ(tensor.data_type(), DataType::kInt64);
  std::vector<int64_t> values(tensor.element_count());
  if (tensor.bytes() > 0) {
    EXPECT_EQ(cudaMemcpy(values.data(), tensor.data(), tensor.bytes(),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
  }
  return values;
}

TEST(NmsTest, DispatchesCudaTensor) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "CUDA postprocess is unavailable";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

  Tensor boxes = MakeCudaTensor({3, 4}, TestBoxes(), "boxes");
  Tensor scores = MakeCudaTensor({3}, TestScores(), "scores");

  Tensor keep = Nms(boxes, scores, 0.5F);

  EXPECT_EQ(keep.name(), "nms_indices");
  EXPECT_EQ(keep.shape(), std::vector<int64_t>({2}));
  EXPECT_EQ(CopyCudaIndices(keep), std::vector<int64_t>({0, 2}));
}

TEST(NmsTest, DispatchesDeviceTensorByInputDevice) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "CUDA postprocess is unavailable";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

  Tensor boxes = MakeCudaTensor({3, 4}, TestBoxes(), "boxes");
  Tensor scores = MakeCudaTensor({3}, TestScores(), "scores");

  Tensor keep = Nms(boxes, scores, 0.5F);

  EXPECT_EQ(CopyCudaIndices(keep), std::vector<int64_t>({0, 2}));
}

TEST(BatchNmsTest, CudaMatchesCpuDenseResult) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "CUDA postprocess is unavailable";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  Tensor boxes = MakeCudaTensor({1, 3, 4}, TestBoxes(), "boxes");
  Tensor scores = MakeCudaTensor(
      {1, 3, 2}, {0.9F, 0.9F, 0.8F, 0.7F, 0.6F, 0.1F}, "scores");
  ExecutionStream stream(Device{DeviceType::kCuda, 0});

  BatchNmsResult result = BatchNms(boxes, scores, BatchOptions(4),
                                   TensorAllocator::Default(), &stream);

  EXPECT_EQ(result.boxes.device().type, DeviceType::kCuda);
  ExpectBatchResult(result, {3}, {0.9F, 0.9F, 0.6F, 0.0F},
                    {0, 1, 0, -1}, {0, 0, 2, -1});
}

#endif

}  // namespace
}  // namespace mw::infer
