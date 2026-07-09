#include <cuda_runtime_api.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>

#include <cfloat>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mw/infer/runtime/postprocess/yolo_decode.h"

namespace mw::infer::postprocess_internal {
namespace {

constexpr int kThreadsPerBlock = 256;

std::string CudaErrorMessage(cudaError_t status, const char* operation) {
  return std::string(operation) + " failed: " + cudaGetErrorString(status);
}

void CheckCuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(CudaErrorMessage(status, operation));
  }
}

int CheckedInt64ToInt(int64_t value, const char* name) {
  if (value < 0 || value > std::numeric_limits<int>::max()) {
    throw std::invalid_argument(std::string(name) + " exceeds int range");
  }
  return static_cast<int>(value);
}

bool HasObjectness(YoloVersion version) {
  switch (version) {
    case YoloVersion::kYoloV5:
      return true;
    case YoloVersion::kYoloV8:
    case YoloVersion::kYoloV11:
      return false;
  }
  throw std::invalid_argument("YOLO version is unsupported");
}

int ClassStart(YoloVersion version) { return HasObjectness(version) ? 5 : 4; }

TensorDesc MakeFloatDesc(std::string name, std::vector<int64_t> shape,
                         Device device) {
  TensorDesc desc;
  desc.info.name = std::move(name);
  desc.info.data_type = DataType::kFloat32;
  desc.info.shape = std::move(shape);
  desc.device = device;
  return desc;
}

TensorDesc MakeClassIdDesc(std::vector<int64_t> shape, Device device) {
  TensorDesc desc;
  desc.info.name = "yolo_class_ids";
  desc.info.data_type = DataType::kInt64;
  desc.info.shape = std::move(shape);
  desc.device = device;
  return desc;
}

Tensor ViewTensor(const Tensor& tensor, std::vector<int64_t> shape) {
  TensorDesc desc = tensor.desc();
  desc.info.shape = std::move(shape);
  return tensor.View(std::move(desc));
}

__device__ float PredictionValue(const float* predictions, bool channel_first,
                                 int channel_count, int candidate_count,
                                 int channel, int candidate) {
  if (channel_first) {
    return predictions[channel * candidate_count + candidate];
  }
  return predictions[candidate * channel_count + channel];
}

__device__ float CandidateScore(const float* predictions, bool channel_first,
                                int channel_count, int candidate_count,
                                int class_start, bool has_objectness,
                                int candidate, int* best_class) {
  const int class_count = channel_count - class_start;
  float best_class_score = -FLT_MAX;
  int selected_class = 0;
  for (int class_index = 0; class_index < class_count; ++class_index) {
    const float class_score =
        PredictionValue(predictions, channel_first, channel_count,
                        candidate_count, class_start + class_index, candidate);
    if (class_score > best_class_score) {
      best_class_score = class_score;
      selected_class = class_index;
    }
  }

  *best_class = selected_class;
  const float objectness =
      has_objectness
          ? PredictionValue(predictions, channel_first, channel_count,
                            candidate_count, 4, candidate)
          : 1.0F;
  return objectness * best_class_score;
}

__global__ void YoloMarkCandidatesKernel(const float* predictions,
                                         bool channel_first, int channel_count,
                                         int candidate_count, int class_start,
                                         bool has_objectness,
                                         float score_threshold, int* flags) {
  const int candidate = blockIdx.x * blockDim.x + threadIdx.x;
  if (candidate >= candidate_count) {
    return;
  }

  int best_class = 0;
  const float score =
      CandidateScore(predictions, channel_first, channel_count, candidate_count,
                     class_start, has_objectness, candidate, &best_class);
  if (score < score_threshold) {
    flags[candidate] = 0;
    return;
  }

  const float width = PredictionValue(predictions, channel_first, channel_count,
                                      candidate_count, 2, candidate);
  const float height = PredictionValue(
      predictions, channel_first, channel_count, candidate_count, 3, candidate);
  flags[candidate] = (width > 0.0F && height > 0.0F) ? 1 : 0;
}

__global__ void YoloCompactKernel(const float* predictions, bool channel_first,
                                  int channel_count, int candidate_count,
                                  int class_start, bool has_objectness,
                                  float class_offset, const int* flags,
                                  const int* offsets, float* boxes,
                                  float* nms_boxes, float* scores,
                                  int64_t* class_ids) {
  const int candidate = blockIdx.x * blockDim.x + threadIdx.x;
  if (candidate >= candidate_count || flags[candidate] == 0) {
    return;
  }

  int best_class = 0;
  const float score =
      CandidateScore(predictions, channel_first, channel_count, candidate_count,
                     class_start, has_objectness, candidate, &best_class);

  const int output = offsets[candidate];
  const float center_x = PredictionValue(
      predictions, channel_first, channel_count, candidate_count, 0, candidate);
  const float center_y = PredictionValue(
      predictions, channel_first, channel_count, candidate_count, 1, candidate);
  const float width = PredictionValue(predictions, channel_first, channel_count,
                                      candidate_count, 2, candidate);
  const float height = PredictionValue(
      predictions, channel_first, channel_count, candidate_count, 3, candidate);
  const float left = center_x - width * 0.5F;
  const float top = center_y - height * 0.5F;
  const float right = center_x + width * 0.5F;
  const float bottom = center_y + height * 0.5F;
  const float class_shift = static_cast<float>(best_class) * class_offset;

  boxes[output * 4] = left;
  boxes[output * 4 + 1] = top;
  boxes[output * 4 + 2] = right;
  boxes[output * 4 + 3] = bottom;
  nms_boxes[output * 4] = left + class_shift;
  nms_boxes[output * 4 + 1] = top + class_shift;
  nms_boxes[output * 4 + 2] = right + class_shift;
  nms_boxes[output * 4 + 3] = bottom + class_shift;
  scores[output] = score;
  class_ids[output] = best_class;
}

}  // namespace

YoloDecodeResult RunYoloDecodeOnDevice(
    const Tensor& predictions, int64_t channel_count, int64_t candidate_count,
    bool channel_first, YoloDecodeOptions options, TensorAllocator& allocator) {
  const int channels = CheckedInt64ToInt(channel_count, "YOLO channel count");
  const int candidates =
      CheckedInt64ToInt(candidate_count, "YOLO candidate count");
  const bool has_objectness = HasObjectness(options.version);
  const int class_start = ClassStart(options.version);
  CheckCuda(cudaSetDevice(predictions.device().id), "cudaSetDevice");

  Tensor boxes = Tensor::Allocate(
      MakeFloatDesc("yolo_boxes", {candidate_count, 4}, predictions.device()),
      allocator);
  Tensor nms_boxes =
      Tensor::Allocate(MakeFloatDesc("yolo_nms_boxes", {candidate_count, 4},
                                     predictions.device()),
                       allocator);
  Tensor scores = Tensor::Allocate(
      MakeFloatDesc("yolo_scores", {candidate_count}, predictions.device()),
      allocator);
  Tensor class_ids = Tensor::Allocate(
      MakeClassIdDesc({candidate_count}, predictions.device()), allocator);

  thrust::device_vector<int> flags(candidates, 0);
  const int blocks = (candidates + kThreadsPerBlock - 1) / kThreadsPerBlock;
  YoloMarkCandidatesKernel<<<blocks, kThreadsPerBlock>>>(
      static_cast<const float*>(predictions.data()), channel_first, channels,
      candidates, class_start, has_objectness, options.score_threshold,
      thrust::raw_pointer_cast(flags.data()));
  CheckCuda(cudaGetLastError(), "YoloMarkCandidatesKernel");

  thrust::device_vector<int> offsets(candidates, 0);
  thrust::exclusive_scan(flags.begin(), flags.end(), offsets.begin());

  std::vector<int> tail_flag(1);
  std::vector<int> tail_offset(1);
  thrust::copy(flags.end() - 1, flags.end(), tail_flag.begin());
  thrust::copy(offsets.end() - 1, offsets.end(), tail_offset.begin());
  const int64_t output_count =
      static_cast<int64_t>(tail_flag[0] + tail_offset[0]);

  if (output_count > 0) {
    YoloCompactKernel<<<blocks, kThreadsPerBlock>>>(
        static_cast<const float*>(predictions.data()), channel_first, channels,
        candidates, class_start, has_objectness, options.class_offset,
        thrust::raw_pointer_cast(flags.data()),
        thrust::raw_pointer_cast(offsets.data()),
        static_cast<float*>(boxes.data()),
        static_cast<float*>(nms_boxes.data()),
        static_cast<float*>(scores.data()),
        static_cast<int64_t*>(class_ids.data()));
    CheckCuda(cudaGetLastError(), "YoloCompactKernel");
  }

  return YoloDecodeResult{
      ViewTensor(boxes, {output_count, 4}),
      ViewTensor(nms_boxes, {output_count, 4}),
      ViewTensor(scores, {output_count}),
      ViewTensor(class_ids, {output_count}),
  };
}

}  // namespace mw::infer::postprocess_internal
