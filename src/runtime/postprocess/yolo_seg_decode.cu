#include <cuda_runtime_api.h>
#include <thrust/scan.h>
#include <thrust/system/cuda/execution_policy.h>

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mw/infer/runtime/execution_stream.h"
#include "yolo_seg_decode_internal.h"

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

template <typename T>
class DeviceBuffer final {
 public:
  explicit DeviceBuffer(std::size_t count) {
    if (count > 0) {
      CheckCuda(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)),
                "cudaMalloc");
    }
  }

  ~DeviceBuffer() {
    if (data_ != nullptr) {
      static_cast<void>(cudaFree(data_));
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  T* data() { return data_; }

 private:
  T* data_ = nullptr;
};

int CheckedInt64ToInt(int64_t value, const char* name) {
  if (value < 0 || value > std::numeric_limits<int>::max()) {
    throw std::invalid_argument(std::string(name) + " exceeds int range");
  }
  return static_cast<int>(value);
}

int CheckedProductToInt(int64_t lhs, int64_t rhs, const char* name) {
  if (lhs < 0 || rhs < 0 ||
      (rhs != 0 && lhs > std::numeric_limits<int>::max() / rhs)) {
    throw std::invalid_argument(std::string(name) + " exceeds int range");
  }
  return static_cast<int>(lhs * rhs);
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
  throw std::invalid_argument("YOLO segmentation version is unsupported");
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

TensorDesc MakeInt64Desc(std::string name, std::vector<int64_t> shape,
                         Device device) {
  TensorDesc desc;
  desc.info.name = std::move(name);
  desc.info.data_type = DataType::kInt64;
  desc.info.shape = std::move(shape);
  desc.device = device;
  return desc;
}

TensorDesc MakeMaskDesc(std::vector<int64_t> shape, Device device) {
  TensorDesc desc;
  desc.info.name = "yolo_seg_masks";
  desc.info.data_type = DataType::kUInt8;
  desc.info.shape = std::move(shape);
  desc.device = device;
  return desc;
}

Tensor ViewTensor(const Tensor& tensor, std::vector<int64_t> shape) {
  TensorDesc desc = tensor.desc();
  desc.info.shape = std::move(shape);
  return tensor.View(std::move(desc));
}

YoloSegCandidates AllocateCandidates(int64_t count, Device device,
                                     TensorAllocator& allocator) {
  YoloSegCandidates result;
  result.boxes = Tensor::Allocate(
      MakeFloatDesc("yolo_seg_boxes", {count, 4}, device), allocator);
  result.scores = Tensor::Allocate(
      MakeFloatDesc("yolo_seg_scores", {count}, device), allocator);
  result.class_ids = Tensor::Allocate(
      MakeInt64Desc("yolo_seg_class_ids", {count}, device), allocator);
  result.batch_ids = Tensor::Allocate(
      MakeInt64Desc("yolo_seg_batch_ids", {count}, device), allocator);
  result.candidate_ids = Tensor::Allocate(
      MakeInt64Desc("yolo_seg_candidate_ids", {count}, device), allocator);
  return result;
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

__device__ float Sigmoid(float value) {
  if (value >= 0.0F) {
    return 1.0F / (1.0F + expf(-value));
  }
  const float exponential = expf(value);
  return exponential / (1.0F + exponential);
}

__device__ float RawCandidateScore(const float* predictions, bool channel_first,
                                   int channel_count, int candidate_count,
                                   int class_start, int class_count,
                                   bool has_objectness, int batch,
                                   int candidate, int* best_class) {
  float best_class_score = -FLT_MAX;
  int selected_class = 0;
  for (int class_index = 0; class_index < class_count; ++class_index) {
    const float class_score = PredictionValue(
        predictions, channel_first, channel_count, candidate_count, batch,
        class_start + class_index, candidate);
    if (class_score > best_class_score) {
      best_class_score = class_score;
      selected_class = class_index;
    }
  }
  *best_class = selected_class;
  const float objectness =
      has_objectness
          ? PredictionValue(predictions, channel_first, channel_count,
                            candidate_count, batch, 4, candidate)
          : 1.0F;
  return objectness * best_class_score;
}

__global__ void RawMarkCandidatesKernel(const float* predictions,
                                        bool channel_first, int batch_count,
                                        int channel_count, int candidate_count,
                                        int class_start, int class_count,
                                        bool has_objectness,
                                        float score_threshold, int* flags) {
  const int global = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_candidates = batch_count * candidate_count;
  if (global >= total_candidates) {
    return;
  }
  const int batch = global / candidate_count;
  const int candidate = global % candidate_count;
  int best_class = 0;
  const float score = RawCandidateScore(
      predictions, channel_first, channel_count, candidate_count, class_start,
      class_count, has_objectness, batch, candidate, &best_class);
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
  flags[global] = isfinite(score) && score >= score_threshold &&
                          isfinite(center_x) && isfinite(center_y) &&
                          isfinite(width) && isfinite(height) && width > 0.0F &&
                          height > 0.0F && isfinite(left) && isfinite(top) &&
                          isfinite(right) && isfinite(bottom)
                      ? 1
                      : 0;
}

__global__ void RawCompactKernel(const float* predictions, bool channel_first,
                                 int batch_count, int channel_count,
                                 int candidate_count, int class_start,
                                 int class_count, bool has_objectness,
                                 const int* flags, const int* offsets,
                                 float* boxes, float* scores,
                                 int64_t* class_ids, int64_t* batch_ids,
                                 int64_t* candidate_ids) {
  const int global = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_candidates = batch_count * candidate_count;
  if (global >= total_candidates || flags[global] == 0) {
    return;
  }
  const int batch = global / candidate_count;
  const int candidate = global % candidate_count;
  int best_class = 0;
  const float score = RawCandidateScore(
      predictions, channel_first, channel_count, candidate_count, class_start,
      class_count, has_objectness, batch, candidate, &best_class);
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
  const int output = offsets[global];

  boxes[output * 4] = left;
  boxes[output * 4 + 1] = top;
  boxes[output * 4 + 2] = right;
  boxes[output * 4 + 3] = bottom;
  scores[output] = score;
  class_ids[output] = best_class;
  batch_ids[output] = batch;
  candidate_ids[output] = candidate;
}

__global__ void SelectedMarkCandidatesKernel(
    const float* predictions, bool channel_first, int batch_count,
    int channel_count, int candidate_count, float score_threshold, int* flags) {
  const int global = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_candidates = batch_count * candidate_count;
  if (global >= total_candidates) {
    return;
  }
  const int batch = global / candidate_count;
  const int candidate = global % candidate_count;
  const float left = PredictionValue(predictions, channel_first, channel_count,
                                     candidate_count, batch, 0, candidate);
  const float top = PredictionValue(predictions, channel_first, channel_count,
                                    candidate_count, batch, 1, candidate);
  const float right = PredictionValue(predictions, channel_first, channel_count,
                                      candidate_count, batch, 2, candidate);
  const float bottom =
      PredictionValue(predictions, channel_first, channel_count,
                      candidate_count, batch, 3, candidate);
  const float score = PredictionValue(predictions, channel_first, channel_count,
                                      candidate_count, batch, 4, candidate);
  const float class_id =
      PredictionValue(predictions, channel_first, channel_count,
                      candidate_count, batch, 5, candidate);
  flags[global] = isfinite(left) && isfinite(top) && isfinite(right) &&
                          isfinite(bottom) && right > left && bottom > top &&
                          isfinite(score) && score >= score_threshold &&
                          isfinite(class_id) && class_id >= 0.0F &&
                          class_id <= static_cast<float>(INT_MAX) &&
                          floorf(class_id) == class_id
                      ? 1
                      : 0;
}

__global__ void SelectedCompactKernel(const float* predictions,
                                      bool channel_first, int batch_count,
                                      int channel_count, int candidate_count,
                                      const int* flags, const int* offsets,
                                      float* boxes, float* scores,
                                      int64_t* class_ids, int64_t* batch_ids,
                                      int64_t* candidate_ids) {
  const int global = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_candidates = batch_count * candidate_count;
  if (global >= total_candidates || flags[global] == 0) {
    return;
  }
  const int batch = global / candidate_count;
  const int candidate = global % candidate_count;
  const int output = offsets[global];
  for (int coordinate = 0; coordinate < 4; ++coordinate) {
    boxes[output * 4 + coordinate] =
        PredictionValue(predictions, channel_first, channel_count,
                        candidate_count, batch, coordinate, candidate);
  }
  scores[output] = PredictionValue(predictions, channel_first, channel_count,
                                   candidate_count, batch, 4, candidate);
  class_ids[output] = static_cast<int64_t>(
      PredictionValue(predictions, channel_first, channel_count,
                      candidate_count, batch, 5, candidate));
  batch_ids[output] = batch;
  candidate_ids[output] = candidate;
}

int64_t CompactCount(const int* flags, const int* offsets, int count,
                     cudaStream_t stream) {
  if (count == 0) {
    return 0;
  }
  int tail_flag = 0;
  int tail_offset = 0;
  CheckCuda(cudaMemcpyAsync(&tail_flag, flags + count - 1, sizeof(int),
                            cudaMemcpyDeviceToHost, stream),
            "cudaMemcpyAsync");
  CheckCuda(cudaMemcpyAsync(&tail_offset, offsets + count - 1, sizeof(int),
                            cudaMemcpyDeviceToHost, stream),
            "cudaMemcpyAsync");
  CheckCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
  return static_cast<int64_t>(tail_flag + tail_offset);
}

__global__ void ComposeCropMaskKernel(
    const float* predictions, const float* prototypes, const float* boxes,
    const int64_t* batch_ids, const int64_t* candidate_ids, bool channel_first,
    int channel_count, int candidate_count, int mask_count,
    int coefficient_start, bool scale_coefficients_by_objectness,
    int prototype_height, int prototype_width, int input_height,
    int input_width, int selected_count, bool output_probabilities,
    float* values) {
  const int global = blockIdx.x * blockDim.x + threadIdx.x;
  const int prototype_plane = prototype_height * prototype_width;
  const int total = selected_count * prototype_plane;
  if (global >= total) {
    return;
  }
  const int selected = global / prototype_plane;
  const int pixel = global % prototype_plane;
  const int y = pixel / prototype_width;
  const int x = pixel % prototype_width;
  const int batch = static_cast<int>(batch_ids[selected]);
  const int candidate = static_cast<int>(candidate_ids[selected]);
  const float left = boxes[selected * 4] * prototype_width / input_width;
  const float top = boxes[selected * 4 + 1] * prototype_height / input_height;
  const float right = boxes[selected * 4 + 2] * prototype_width / input_width;
  const float bottom =
      boxes[selected * 4 + 3] * prototype_height / input_height;
  if (static_cast<float>(x) < left || static_cast<float>(x) >= right ||
      static_cast<float>(y) < top || static_cast<float>(y) >= bottom) {
    values[global] = 0.0F;
    return;
  }
  const float objectness =
      scale_coefficients_by_objectness
          ? PredictionValue(predictions, channel_first, channel_count,
                            candidate_count, batch, 4, candidate)
          : 1.0F;
  float value = 0.0F;
  for (int channel = 0; channel < mask_count; ++channel) {
    const float coefficient =
        PredictionValue(predictions, channel_first, channel_count,
                        candidate_count, batch, coefficient_start + channel,
                        candidate) *
        objectness;
    value +=
        coefficient *
        prototypes[(batch * mask_count + channel) * prototype_plane + pixel];
  }
  values[global] = output_probabilities ? Sigmoid(value) : value;
}

__global__ void ResizeThresholdMaskKernel(
    const float* values, const float* boxes, int selected_count,
    int prototype_height, int prototype_width, int input_height,
    int input_width, float value_threshold, std::uint8_t* masks) {
  const int global = blockIdx.x * blockDim.x + threadIdx.x;
  const int output_plane = input_height * input_width;
  const int total = selected_count * output_plane;
  if (global >= total) {
    return;
  }
  const int selected = global / output_plane;
  const int pixel = global % output_plane;
  const int y = pixel / input_width;
  const int x = pixel % input_width;
  if (static_cast<float>(x) < boxes[selected * 4] ||
      static_cast<float>(x) >= boxes[selected * 4 + 2] ||
      static_cast<float>(y) < boxes[selected * 4 + 1] ||
      static_cast<float>(y) >= boxes[selected * 4 + 3]) {
    masks[global] = 0;
    return;
  }

  const float source_x =
      (static_cast<float>(x) + 0.5F) * prototype_width / input_width - 0.5F;
  const float source_y =
      (static_cast<float>(y) + 0.5F) * prototype_height / input_height - 0.5F;
  const float clamped_x =
      fminf(fmaxf(source_x, 0.0F), static_cast<float>(prototype_width - 1));
  const float clamped_y =
      fminf(fmaxf(source_y, 0.0F), static_cast<float>(prototype_height - 1));
  const int x0 = static_cast<int>(floorf(clamped_x));
  const int y0 = static_cast<int>(floorf(clamped_y));
  const int x1 = min(x0 + 1, prototype_width - 1);
  const int y1 = min(y0 + 1, prototype_height - 1);
  const float x_weight = clamped_x - x0;
  const float y_weight = clamped_y - y0;
  const int prototype_plane = prototype_height * prototype_width;
  const float* selected_values = values + selected * prototype_plane;
  const float top_value =
      selected_values[y0 * prototype_width + x0] * (1.0F - x_weight) +
      selected_values[y0 * prototype_width + x1] * x_weight;
  const float bottom_value =
      selected_values[y1 * prototype_width + x0] * (1.0F - x_weight) +
      selected_values[y1 * prototype_width + x1] * x_weight;
  const float value = top_value * (1.0F - y_weight) + bottom_value * y_weight;
  masks[global] = value > value_threshold ? 1 : 0;
}

}  // namespace

YoloSegCandidates RunRawYoloSegDecodeOnDevice(
    const Tensor& predictions, int64_t batch_count, int64_t channel_count,
    int64_t candidate_count, bool channel_first, int64_t class_count,
    YoloSegDecodeOptions options, TensorAllocator& allocator,
    ExecutionStream* execution_stream) {
  const int batches =
      CheckedInt64ToInt(batch_count, "YOLO segmentation batch count");
  const int channels =
      CheckedInt64ToInt(channel_count, "YOLO segmentation channel count");
  const int candidates =
      CheckedInt64ToInt(candidate_count, "YOLO segmentation candidate count");
  const int classes =
      CheckedInt64ToInt(class_count, "YOLO segmentation class count");
  const int total_candidates = CheckedProductToInt(
      batches, candidates, "YOLO segmentation candidate total");
  CheckCuda(cudaSetDevice(predictions.device().id), "cudaSetDevice");
  const cudaStream_t cuda_stream = execution_stream == nullptr
                                       ? cudaStream_t{}
                                       : execution_stream->cuda_handle();

  YoloSegCandidates storage =
      AllocateCandidates(total_candidates, predictions.device(), allocator);
  if (total_candidates == 0) {
    return storage;
  }
  DeviceBuffer<int> flags(total_candidates);
  DeviceBuffer<int> offsets(total_candidates);
  const int blocks =
      (total_candidates + kThreadsPerBlock - 1) / kThreadsPerBlock;
  const int class_start = ClassStart(options.version);
  const bool has_objectness = HasObjectness(options.version);
  RawMarkCandidatesKernel<<<blocks, kThreadsPerBlock, 0, cuda_stream>>>(
      static_cast<const float*>(predictions.data()), channel_first, batches,
      channels, candidates, class_start, classes, has_objectness,
      options.score_threshold, flags.data());
  CheckCuda(cudaGetLastError(), "RawMarkCandidatesKernel");
  auto policy = thrust::cuda::par.on(cuda_stream);
  thrust::exclusive_scan(policy, flags.data(), flags.data() + total_candidates,
                         offsets.data());
  const int64_t output_count =
      CompactCount(flags.data(), offsets.data(), total_candidates, cuda_stream);
  if (output_count > 0) {
    RawCompactKernel<<<blocks, kThreadsPerBlock, 0, cuda_stream>>>(
        static_cast<const float*>(predictions.data()), channel_first, batches,
        channels, candidates, class_start, classes, has_objectness,
        flags.data(), offsets.data(), static_cast<float*>(storage.boxes.data()),
        static_cast<float*>(storage.scores.data()),
        static_cast<int64_t*>(storage.class_ids.data()),
        static_cast<int64_t*>(storage.batch_ids.data()),
        static_cast<int64_t*>(storage.candidate_ids.data()));
    CheckCuda(cudaGetLastError(), "RawCompactKernel");
    CheckCuda(cudaStreamSynchronize(cuda_stream), "cudaStreamSynchronize");
  }
  return YoloSegCandidates{
      ViewTensor(storage.boxes, {output_count, 4}),
      ViewTensor(storage.scores, {output_count}),
      ViewTensor(storage.class_ids, {output_count}),
      ViewTensor(storage.batch_ids, {output_count}),
      ViewTensor(storage.candidate_ids, {output_count}),
  };
}

YoloSegCandidates RunSelectedYoloSegDecodeOnDevice(
    const Tensor& predictions, int64_t batch_count, int64_t channel_count,
    int64_t candidate_count, bool channel_first, YoloSegDecodeOptions options,
    TensorAllocator& allocator, ExecutionStream* execution_stream) {
  const int batches =
      CheckedInt64ToInt(batch_count, "YOLO segmentation batch count");
  const int channels =
      CheckedInt64ToInt(channel_count, "YOLO segmentation channel count");
  const int candidates =
      CheckedInt64ToInt(candidate_count, "YOLO segmentation candidate count");
  const int total_candidates = CheckedProductToInt(
      batches, candidates, "YOLO segmentation candidate total");
  CheckCuda(cudaSetDevice(predictions.device().id), "cudaSetDevice");
  const cudaStream_t cuda_stream = execution_stream == nullptr
                                       ? cudaStream_t{}
                                       : execution_stream->cuda_handle();

  YoloSegCandidates storage =
      AllocateCandidates(total_candidates, predictions.device(), allocator);
  if (total_candidates == 0) {
    return storage;
  }
  DeviceBuffer<int> flags(total_candidates);
  DeviceBuffer<int> offsets(total_candidates);
  const int blocks =
      (total_candidates + kThreadsPerBlock - 1) / kThreadsPerBlock;
  SelectedMarkCandidatesKernel<<<blocks, kThreadsPerBlock, 0, cuda_stream>>>(
      static_cast<const float*>(predictions.data()), channel_first, batches,
      channels, candidates, options.score_threshold, flags.data());
  CheckCuda(cudaGetLastError(), "SelectedMarkCandidatesKernel");
  auto policy = thrust::cuda::par.on(cuda_stream);
  thrust::exclusive_scan(policy, flags.data(), flags.data() + total_candidates,
                         offsets.data());
  const int64_t output_count =
      CompactCount(flags.data(), offsets.data(), total_candidates, cuda_stream);
  if (output_count > 0) {
    SelectedCompactKernel<<<blocks, kThreadsPerBlock, 0, cuda_stream>>>(
        static_cast<const float*>(predictions.data()), channel_first, batches,
        channels, candidates, flags.data(), offsets.data(),
        static_cast<float*>(storage.boxes.data()),
        static_cast<float*>(storage.scores.data()),
        static_cast<int64_t*>(storage.class_ids.data()),
        static_cast<int64_t*>(storage.batch_ids.data()),
        static_cast<int64_t*>(storage.candidate_ids.data()));
    CheckCuda(cudaGetLastError(), "SelectedCompactKernel");
    CheckCuda(cudaStreamSynchronize(cuda_stream), "cudaStreamSynchronize");
  }
  return YoloSegCandidates{
      ViewTensor(storage.boxes, {output_count, 4}),
      ViewTensor(storage.scores, {output_count}),
      ViewTensor(storage.class_ids, {output_count}),
      ViewTensor(storage.batch_ids, {output_count}),
      ViewTensor(storage.candidate_ids, {output_count}),
  };
}

Tensor RunYoloSegMasksOnDevice(
    const Tensor& predictions, const Tensor& prototypes, const Tensor& boxes,
    const Tensor& batch_ids, const Tensor& candidate_ids, int64_t channel_count,
    int64_t candidate_count, bool channel_first, int64_t coefficient_start,
    bool scale_coefficients_by_objectness, ImageSize input_size,
    float mask_threshold, TensorAllocator& allocator,
    ExecutionStream* execution_stream) {
  const int selected =
      CheckedInt64ToInt(boxes.shape()[0], "YOLO segmentation selected count");
  Tensor output = Tensor::Allocate(
      MakeMaskDesc({selected, input_size.height, input_size.width},
                   predictions.device()),
      allocator);
  if (selected == 0) {
    return output;
  }
  const int channels =
      CheckedInt64ToInt(channel_count, "YOLO segmentation channel count");
  const int candidates =
      CheckedInt64ToInt(candidate_count, "YOLO segmentation candidate count");
  const int masks =
      CheckedInt64ToInt(prototypes.shape()[1], "YOLO mask channel count");
  const int coefficient =
      CheckedInt64ToInt(coefficient_start, "YOLO coefficient start");
  const int prototype_height =
      CheckedInt64ToInt(prototypes.shape()[2], "YOLO prototype height");
  const int prototype_width =
      CheckedInt64ToInt(prototypes.shape()[3], "YOLO prototype width");
  const int prototype_plane = CheckedProductToInt(
      prototype_height, prototype_width, "YOLO prototype plane");
  const int value_count =
      CheckedProductToInt(selected, prototype_plane, "YOLO mask values");
  const int output_plane = CheckedProductToInt(
      input_size.height, input_size.width, "YOLO output mask plane");
  const int output_count =
      CheckedProductToInt(selected, output_plane, "YOLO output masks");
  const bool output_probabilities = mask_threshold != 0.5F;
  const float value_threshold = output_probabilities ? mask_threshold : 0.0F;
  CheckCuda(cudaSetDevice(predictions.device().id), "cudaSetDevice");
  const cudaStream_t cuda_stream = execution_stream == nullptr
                                       ? cudaStream_t{}
                                       : execution_stream->cuda_handle();

  Tensor mask_values = Tensor::Allocate(
      MakeFloatDesc("yolo_seg_mask_values",
                    {selected, prototype_height, prototype_width},
                    predictions.device()),
      allocator);
  const int value_blocks =
      (value_count + kThreadsPerBlock - 1) / kThreadsPerBlock;
  ComposeCropMaskKernel<<<value_blocks, kThreadsPerBlock, 0, cuda_stream>>>(
      static_cast<const float*>(predictions.data()),
      static_cast<const float*>(prototypes.data()),
      static_cast<const float*>(boxes.data()),
      static_cast<const int64_t*>(batch_ids.data()),
      static_cast<const int64_t*>(candidate_ids.data()), channel_first,
      channels, candidates, masks, coefficient,
      scale_coefficients_by_objectness, prototype_height, prototype_width,
      input_size.height, input_size.width, selected, output_probabilities,
      static_cast<float*>(mask_values.data()));
  CheckCuda(cudaGetLastError(), "ComposeCropMaskKernel");

  const int output_blocks =
      (output_count + kThreadsPerBlock - 1) / kThreadsPerBlock;
  ResizeThresholdMaskKernel<<<output_blocks, kThreadsPerBlock, 0,
                              cuda_stream>>>(
      static_cast<const float*>(mask_values.data()),
      static_cast<const float*>(boxes.data()), selected, prototype_height,
      prototype_width, input_size.height, input_size.width, value_threshold,
      static_cast<std::uint8_t*>(output.data()));
  CheckCuda(cudaGetLastError(), "ResizeThresholdMaskKernel");
  CheckCuda(cudaStreamSynchronize(cuda_stream), "cudaStreamSynchronize");
  return output;
}

}  // namespace mw::infer::postprocess_internal
