#include <cuda_runtime_api.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mw/infer/runtime/execution_stream.h"
#include "nms_internal.h"

namespace mw::infer::postprocess_internal {
namespace {

constexpr int kThreadsPerBlock = 64;

std::string CudaErrorMessage(cudaError_t status, const char* operation) {
  return std::string(operation) + " failed: " + cudaGetErrorString(status);
}

void CheckCuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(CudaErrorMessage(status, operation));
  }
}

int CheckedBoxCount(const Tensor& boxes) {
  if (boxes.shape()[0] > std::numeric_limits<int>::max()) {
    throw std::invalid_argument("CUDA NMS boxes count exceeds int range");
  }
  return static_cast<int>(boxes.shape()[0]);
}

template <typename T>
class DeviceBuffer final {
 public:
  explicit DeviceBuffer(std::size_t count) {
    if (count > 0) {
      if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
        throw std::invalid_argument("CUDA NMS buffer size overflows size_t");
      }
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

template <typename T>
class PinnedBuffer final {
 public:
  explicit PinnedBuffer(std::size_t count) {
    if (count > 0) {
      CheckCuda(cudaMallocHost(reinterpret_cast<void**>(&data_),
                               count * sizeof(T)),
                "cudaMallocHost");
    }
  }

  ~PinnedBuffer() {
    if (data_ != nullptr) {
      static_cast<void>(cudaFreeHost(data_));
    }
  }

  PinnedBuffer(const PinnedBuffer&) = delete;
  PinnedBuffer& operator=(const PinnedBuffer&) = delete;

  T* data() { return data_; }
  const T* data() const { return data_; }

 private:
  T* data_ = nullptr;
};

__device__ float DeviceMax(float lhs, float rhs) {
  return lhs > rhs ? lhs : rhs;
}

__device__ float DeviceMin(float lhs, float rhs) {
  return lhs < rhs ? lhs : rhs;
}

struct ScoreDescending {
  const float* scores = nullptr;

  __host__ __device__ bool operator()(int64_t lhs, int64_t rhs) const {
    return scores[lhs] > scores[rhs];
  }
};

__device__ float DeviceBoxIoU(const float* lhs, const float* rhs,
                              float offset) {
  const float left = DeviceMax(lhs[0], rhs[0]);
  const float top = DeviceMax(lhs[1], rhs[1]);
  const float right = DeviceMin(lhs[2], rhs[2]);
  const float bottom = DeviceMin(lhs[3], rhs[3]);
  const float width = DeviceMax(right - left + offset, 0.0F);
  const float height = DeviceMax(bottom - top + offset, 0.0F);
  const float intersection = width * height;

  const float lhs_area = DeviceMax(lhs[2] - lhs[0] + offset, 0.0F) *
                         DeviceMax(lhs[3] - lhs[1] + offset, 0.0F);
  const float rhs_area = DeviceMax(rhs[2] - rhs[0] + offset, 0.0F) *
                         DeviceMax(rhs[3] - rhs[1] + offset, 0.0F);
  const float denominator = lhs_area + rhs_area - intersection;
  if (denominator <= 0.0F) {
    return 0.0F;
  }
  return intersection / denominator;
}

__global__ void GatherSortedInputsKernel(
    int count, const int64_t* order, const float* boxes,
    const int64_t* class_ids, const int64_t* batch_ids, float* sorted_boxes,
    int64_t* sorted_class_ids, int64_t* sorted_batch_ids) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }

  const int64_t source_index = order[index];
  for (int coord = 0; coord < 4; ++coord) {
    sorted_boxes[index * 4 + coord] = boxes[source_index * 4 + coord];
  }
  if (class_ids != nullptr) {
    sorted_class_ids[index] = class_ids[source_index];
  }
  if (batch_ids != nullptr) {
    sorted_batch_ids[index] = batch_ids[source_index];
  }
}

__global__ void NmsMaskKernel(int count, float iou_threshold, float offset,
                              const float* boxes, const int64_t* class_ids,
                              const int64_t* batch_ids,
                              unsigned long long* mask, int column_blocks) {
  const int row_block = blockIdx.y;
  const int column_block = blockIdx.x;
  if (row_block > column_block) {
    return;
  }

  const int row_start = row_block * kThreadsPerBlock;
  const int column_start = column_block * kThreadsPerBlock;
  const int row_size = min(count - row_start, kThreadsPerBlock);
  const int column_size = min(count - column_start, kThreadsPerBlock);

  __shared__ float column_boxes[kThreadsPerBlock * 4];
  if (threadIdx.x < column_size) {
    for (int coord = 0; coord < 4; ++coord) {
      column_boxes[threadIdx.x * 4 + coord] =
          boxes[(column_start + threadIdx.x) * 4 + coord];
    }
  }
  __syncthreads();

  if (threadIdx.x >= row_size) {
    return;
  }

  const int current_index = row_start + threadIdx.x;
  const float* current_box = boxes + current_index * 4;
  const int compare_start = row_block == column_block ? threadIdx.x + 1 : 0;

  unsigned long long suppress_mask = 0;
  for (int compare_index = compare_start; compare_index < column_size;
       ++compare_index) {
    const int candidate_index = column_start + compare_index;
    if ((batch_ids != nullptr &&
         batch_ids[current_index] != batch_ids[candidate_index]) ||
        (class_ids != nullptr &&
         class_ids[current_index] != class_ids[candidate_index])) {
      continue;
    }
    if (DeviceBoxIoU(current_box, column_boxes + compare_index * 4, offset) >
        iou_threshold) {
      suppress_mask |= 1ULL << compare_index;
    }
  }

  mask[current_index * column_blocks + column_block] = suppress_mask;
}

TensorDesc MakeBatchOutputDesc(std::string name, DataType data_type,
                               std::vector<int64_t> shape, Device device) {
  TensorDesc desc;
  desc.info.name = std::move(name);
  desc.info.data_type = data_type;
  desc.info.shape = std::move(shape);
  desc.device = device;
  return desc;
}

BatchNmsResult AllocateBatchResult(const Tensor& boxes,
                                   const BatchNmsOptions& options,
                                   TensorAllocator& allocator) {
  const int64_t batch_count = boxes.shape()[0];
  const int64_t max_detections = options.max_detections;
  const Device device = boxes.device();
  Tensor counts = Tensor::Allocate(
      MakeBatchOutputDesc("batch_nms_counts", DataType::kInt64, {batch_count},
                          device),
      allocator);
  Tensor output_boxes = Tensor::Allocate(
      MakeBatchOutputDesc("batch_nms_boxes", DataType::kFloat32,
                          {batch_count, max_detections, 4}, device),
      allocator);
  Tensor output_scores = Tensor::Allocate(
      MakeBatchOutputDesc("batch_nms_scores", DataType::kFloat32,
                          {batch_count, max_detections}, device),
      allocator);
  Tensor class_ids = Tensor::Allocate(
      MakeBatchOutputDesc("batch_nms_class_ids", DataType::kInt64,
                          {batch_count, max_detections}, device),
      allocator);
  Tensor indices = Tensor::Allocate(
      MakeBatchOutputDesc("batch_nms_indices", DataType::kInt64,
                          {batch_count, max_detections}, device),
      allocator);
  return BatchNmsResult{std::move(counts), std::move(output_boxes),
                        std::move(output_scores), std::move(class_ids),
                        std::move(indices)};
}

}  // namespace

Tensor RunNmsOnDevice(const Tensor& boxes, const Tensor& scores,
                      const Tensor* class_ids, const Tensor* batch_ids,
                      NmsParameters parameters, TensorAllocator& allocator,
                      ExecutionStream* execution_stream) {
  const int count = CheckedBoxCount(boxes);
  if (count == 0) {
    TensorDesc desc;
    desc.info.name = "nms_indices";
    desc.info.data_type = DataType::kInt64;
    desc.info.shape = {0};
    desc.device = boxes.device();
    return Tensor::Allocate(std::move(desc), allocator);
  }

  CheckCuda(cudaSetDevice(boxes.device().id), "cudaSetDevice");
  const cudaStream_t cuda_stream = execution_stream == nullptr
                                       ? cudaStream_t{}
                                       : execution_stream->cuda_handle();
  auto policy = thrust::cuda::par.on(cuda_stream);

  const auto* boxes_data = static_cast<const float*>(boxes.data());
  const auto* scores_data = static_cast<const float*>(scores.data());

  DeviceBuffer<int64_t> order(count);
  thrust::sequence(policy, order.data(), order.data() + count, int64_t{0});
  thrust::stable_sort(policy, order.data(), order.data() + count,
                      ScoreDescending{scores_data});

  DeviceBuffer<float> sorted_boxes(static_cast<std::size_t>(count) * 4U);
  DeviceBuffer<int64_t> sorted_class_ids(class_ids == nullptr ? 0 : count);
  DeviceBuffer<int64_t> sorted_batch_ids(batch_ids == nullptr ? 0 : count);
  const int gather_blocks = (count + kThreadsPerBlock - 1) / kThreadsPerBlock;
  GatherSortedInputsKernel<<<gather_blocks, kThreadsPerBlock, 0, cuda_stream>>>(
      count, order.data(), boxes_data,
      class_ids == nullptr ? nullptr
                           : static_cast<const int64_t*>(class_ids->data()),
      batch_ids == nullptr ? nullptr
                           : static_cast<const int64_t*>(batch_ids->data()),
      sorted_boxes.data(), sorted_class_ids.data(), sorted_batch_ids.data());
  CheckCuda(cudaGetLastError(), "GatherSortedInputsKernel");

  const int column_blocks = (count + kThreadsPerBlock - 1) / kThreadsPerBlock;
  const std::size_t mask_count =
      static_cast<std::size_t>(count) * column_blocks;
  DeviceBuffer<unsigned long long> mask(mask_count);
  dim3 mask_blocks(column_blocks, column_blocks);
  NmsMaskKernel<<<mask_blocks, kThreadsPerBlock, 0, cuda_stream>>>(
      count, parameters.iou_threshold, parameters.coordinate_offset,
      sorted_boxes.data(), sorted_class_ids.data(), sorted_batch_ids.data(),
      mask.data(), column_blocks);
  CheckCuda(cudaGetLastError(), "NmsMaskKernel");

  std::vector<unsigned long long> host_mask(mask_count);
  std::vector<int64_t> host_order(static_cast<std::size_t>(count));
  std::vector<int64_t> host_batch_ids(
      batch_ids == nullptr ? 0 : static_cast<std::size_t>(count));
  CheckCuda(cudaMemcpyAsync(host_mask.data(), mask.data(),
                            host_mask.size() * sizeof(unsigned long long),
                            cudaMemcpyDeviceToHost, cuda_stream),
            "cudaMemcpyAsync");
  CheckCuda(cudaMemcpyAsync(host_order.data(), order.data(),
                            host_order.size() * sizeof(int64_t),
                            cudaMemcpyDeviceToHost, cuda_stream),
            "cudaMemcpyAsync");
  if (batch_ids != nullptr) {
    CheckCuda(cudaMemcpyAsync(host_batch_ids.data(), sorted_batch_ids.data(),
                              host_batch_ids.size() * sizeof(int64_t),
                              cudaMemcpyDeviceToHost, cuda_stream),
              "cudaMemcpyAsync");
  }
  CheckCuda(cudaStreamSynchronize(cuda_stream), "cudaStreamSynchronize");

  std::vector<unsigned long long> suppressed(column_blocks, 0);
  std::vector<int64_t> keep;
  keep.reserve(static_cast<std::size_t>(count));
  std::unordered_map<int64_t, int64_t> batch_output_counts;
  for (int sorted_index = 0; sorted_index < count; ++sorted_index) {
    const int block_index = sorted_index / kThreadsPerBlock;
    const int bit_index = sorted_index % kThreadsPerBlock;
    if ((suppressed[block_index] & (1ULL << bit_index)) != 0) {
      continue;
    }

    if (batch_ids != nullptr && parameters.max_output_boxes > 0 &&
        batch_output_counts[host_batch_ids[sorted_index]] >=
            parameters.max_output_boxes) {
      continue;
    }
    keep.push_back(host_order[sorted_index]);
    if (batch_ids != nullptr) {
      ++batch_output_counts[host_batch_ids[sorted_index]];
    } else if (parameters.max_output_boxes > 0 &&
        keep.size() == static_cast<std::size_t>(parameters.max_output_boxes)) {
      break;
    }
    const auto* current_mask =
        host_mask.data() +
        static_cast<std::size_t>(sorted_index) * column_blocks;
    for (int block = block_index; block < column_blocks; ++block) {
      suppressed[block] |= current_mask[block];
    }
  }

  TensorDesc desc;
  desc.info.name = "nms_indices";
  desc.info.data_type = DataType::kInt64;
  desc.info.shape = {static_cast<int64_t>(keep.size())};
  desc.device = boxes.device();
  Tensor output = Tensor::Allocate(std::move(desc), allocator);
  if (output.bytes() > 0) {
    CheckCuda(cudaMemcpyAsync(output.data(), keep.data(), output.bytes(),
                              cudaMemcpyHostToDevice, cuda_stream),
              "cudaMemcpyAsync");
    CheckCuda(cudaStreamSynchronize(cuda_stream), "cudaStreamSynchronize");
  }
  return output;
}

BatchNmsResult RunBatchNmsOnDevice(
    const Tensor& boxes, const Tensor& scores,
    const BatchNmsOptions& options, TensorAllocator& allocator,
    ExecutionStream* execution_stream) {
  CheckCuda(cudaSetDevice(boxes.device().id), "cudaSetDevice");
  const cudaStream_t cuda_stream = execution_stream == nullptr
                                       ? cudaStream_t{}
                                       : execution_stream->cuda_handle();
  PinnedBuffer<float> host_boxes(boxes.element_count());
  PinnedBuffer<float> host_scores(scores.element_count());

  try {
    if (boxes.bytes() > 0) {
      CheckCuda(cudaMemcpyAsync(host_boxes.data(), boxes.data(), boxes.bytes(),
                                cudaMemcpyDeviceToHost, cuda_stream),
                "cudaMemcpyAsync");
    }
    if (scores.bytes() > 0) {
      CheckCuda(cudaMemcpyAsync(host_scores.data(), scores.data(),
                                scores.bytes(), cudaMemcpyDeviceToHost,
                                cuda_stream),
                "cudaMemcpyAsync");
    }
    if (boxes.bytes() > 0 || scores.bytes() > 0) {
      CheckCuda(cudaStreamSynchronize(cuda_stream), "cudaStreamSynchronize");
    }

    BatchNmsHostResult host_result = RunBatchNmsOnHostBuffers(
        host_boxes.data(), host_scores.data(), boxes.shape()[0],
        boxes.shape()[1], scores.shape()[2], options);
    BatchNmsResult output = AllocateBatchResult(boxes, options, allocator);

    PinnedBuffer<int64_t> output_counts(host_result.counts.size());
    PinnedBuffer<float> output_boxes(host_result.boxes.size());
    PinnedBuffer<float> output_scores(host_result.scores.size());
    PinnedBuffer<int64_t> output_class_ids(host_result.class_ids.size());
    PinnedBuffer<int64_t> output_indices(host_result.indices.size());
    std::copy(host_result.counts.begin(), host_result.counts.end(),
              output_counts.data());
    std::copy(host_result.boxes.begin(), host_result.boxes.end(),
              output_boxes.data());
    std::copy(host_result.scores.begin(), host_result.scores.end(),
              output_scores.data());
    std::copy(host_result.class_ids.begin(), host_result.class_ids.end(),
              output_class_ids.data());
    std::copy(host_result.indices.begin(), host_result.indices.end(),
              output_indices.data());

    bool has_output_copy = false;
    const auto copy_output = [&](void* destination, const void* source,
                                 std::size_t bytes) {
      if (bytes == 0) {
        return;
      }
      CheckCuda(cudaMemcpyAsync(destination, source, bytes,
                                cudaMemcpyHostToDevice, cuda_stream),
                "cudaMemcpyAsync");
      has_output_copy = true;
    };
    copy_output(output.counts.data(), output_counts.data(),
                output.counts.bytes());
    copy_output(output.boxes.data(), output_boxes.data(), output.boxes.bytes());
    copy_output(output.scores.data(), output_scores.data(),
                output.scores.bytes());
    copy_output(output.class_ids.data(), output_class_ids.data(),
                output.class_ids.bytes());
    copy_output(output.indices.data(), output_indices.data(),
                output.indices.bytes());
    if (has_output_copy) {
      CheckCuda(cudaStreamSynchronize(cuda_stream), "cudaStreamSynchronize");
    }
    return output;
  } catch (...) {
    static_cast<void>(cudaStreamSynchronize(cuda_stream));
    throw;
  }
}

}  // namespace mw::infer::postprocess_internal
