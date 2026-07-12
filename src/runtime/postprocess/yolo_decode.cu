#include <cuda_runtime_api.h>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mw/infer/runtime/execution_stream.h"
#include "mw/infer/runtime/postprocess/yolo_decode.h"

namespace mw::infer::postprocess_internal {
namespace {

constexpr int kThreadsPerBlock = 256;

void CheckCuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) +
                             " failed: " + cudaGetErrorString(status));
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
    case YoloVersion::kYoloV26:
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

struct OutputOwner {
  Tensor boxes_storage;
  Tensor scores_storage;
  Tensor predictions;
};

YoloDecodeResult AllocateResult(const Tensor& predictions, int64_t batch_count,
                                int64_t candidate_count, int64_t class_count,
                                TensorAllocator& allocator) {
  const TensorDesc boxes_desc =
      MakeFloatDesc("yolo_boxes", {batch_count, candidate_count, 4},
                    predictions.device());
  const TensorDesc scores_desc =
      MakeFloatDesc("yolo_scores", {batch_count, candidate_count, class_count},
                    predictions.device());
  auto owner = std::make_shared<OutputOwner>();
  owner->boxes_storage = Tensor::Allocate(boxes_desc, allocator);
  owner->scores_storage = Tensor::Allocate(scores_desc, allocator);
  owner->predictions = predictions;
  std::shared_ptr<void> boxes_owner = owner;
  std::shared_ptr<void> scores_owner = owner;
  return YoloDecodeResult{
      Tensor::FromExternal(boxes_desc, owner->boxes_storage.data(),
                           owner->boxes_storage.bytes(),
                           std::move(boxes_owner)),
      Tensor::FromExternal(scores_desc, owner->scores_storage.data(),
                           owner->scores_storage.bytes(),
                           std::move(scores_owner)),
  };
}

__device__ float PredictionValue(const float* predictions, bool channel_first,
                                 int channel_count, int candidate_count,
                                 int batch, int channel, int candidate) {
  if (channel_first) {
    return predictions[(batch * channel_count + channel) * candidate_count +
                       candidate];
  }
  return predictions[(batch * candidate_count + candidate) * channel_count +
                     channel];
}

__global__ void YoloDecodeKernel(
    const float* predictions, bool channel_first, int batch_count,
    int channel_count, int candidate_count, int class_start,
    bool has_objectness, float* boxes, float* scores) {
  const int global = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_candidates = batch_count * candidate_count;
  if (global >= total_candidates) {
    return;
  }
  const int batch = global / candidate_count;
  const int candidate = global % candidate_count;
  const int class_count = channel_count - class_start;

  const float center_x =
      PredictionValue(predictions, channel_first, channel_count,
                      candidate_count, batch, 0, candidate);
  const float center_y =
      PredictionValue(predictions, channel_first, channel_count,
                      candidate_count, batch, 1, candidate);
  const float width = PredictionValue(predictions, channel_first, channel_count,
                                      candidate_count, batch, 2, candidate);
  const float height =
      PredictionValue(predictions, channel_first, channel_count,
                      candidate_count, batch, 3, candidate);
  const float left = center_x - width * 0.5F;
  const float top = center_y - height * 0.5F;
  const float right = center_x + width * 0.5F;
  const float bottom = center_y + height * 0.5F;

  float* output_box = boxes + static_cast<std::size_t>(global) * 4U;
  output_box[0] = left;
  output_box[1] = top;
  output_box[2] = right;
  output_box[3] = bottom;

  const float objectness =
      has_objectness
          ? PredictionValue(predictions, channel_first, channel_count,
                            candidate_count, batch, 4, candidate)
          : 1.0F;
  float* output_scores =
      scores + static_cast<std::size_t>(global) * class_count;
  for (int class_index = 0; class_index < class_count; ++class_index) {
    const float class_score = PredictionValue(
        predictions, channel_first, channel_count, candidate_count, batch,
        class_start + class_index, candidate);
    output_scores[class_index] = objectness * class_score;
  }
}

void SynchronizeNoThrow(ExecutionStream* execution_stream,
                        cudaStream_t cuda_stream) noexcept {
  if (execution_stream != nullptr) {
    execution_stream->SynchronizeNoThrow();
    return;
  }
  static_cast<void>(cudaStreamSynchronize(cuda_stream));
}

}  // namespace

YoloDecodeResult RunYoloDecodeOnDevice(
    const Tensor& predictions, int64_t batch_count, int64_t channel_count,
    int64_t candidate_count, bool channel_first, YoloDecodeOptions options,
    TensorAllocator& allocator, ExecutionStream* execution_stream) {
  const int batches = CheckedInt64ToInt(batch_count, "YOLO batch count");
  const int channels = CheckedInt64ToInt(channel_count, "YOLO channel count");
  const int candidates =
      CheckedInt64ToInt(candidate_count, "YOLO candidate count");
  if (batches != 0 && candidates > std::numeric_limits<int>::max() / batches) {
    throw std::invalid_argument("YOLO candidate count exceeds int range");
  }
  const int total_candidates = batches * candidates;
  const bool has_objectness = HasObjectness(options.version);
  const int class_start = ClassStart(options.version);
  const int class_count = channels - class_start;
  CheckCuda(cudaSetDevice(predictions.device().id), "cudaSetDevice");
  const cudaStream_t cuda_stream = execution_stream == nullptr
                                       ? cudaStream_t{}
                                       : execution_stream->cuda_handle();

  YoloDecodeResult result = AllocateResult(
      predictions, batch_count, candidate_count, class_count, allocator);
  if (total_candidates == 0) {
    return result;
  }

  try {
    const int blocks =
        (total_candidates + kThreadsPerBlock - 1) / kThreadsPerBlock;
    YoloDecodeKernel<<<blocks, kThreadsPerBlock, 0, cuda_stream>>>(
        static_cast<const float*>(predictions.data()), channel_first, batches,
        channels, candidates, class_start, has_objectness,
        static_cast<float*>(result.boxes.data()),
        static_cast<float*>(result.scores.data()));
    CheckCuda(cudaGetLastError(), "YoloDecodeKernel");
  } catch (...) {
    SynchronizeNoThrow(execution_stream, cuda_stream);
    throw;
  }
  return result;
}

}  // namespace mw::infer::postprocess_internal
