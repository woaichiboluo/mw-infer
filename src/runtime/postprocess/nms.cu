#include <cuda_runtime_api.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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

__global__ void GatherSortedBoxesKernel(int count, const int64_t* order,
                                        const float* boxes,
                                        float* sorted_boxes) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }

  const int64_t source_index = order[index];
  for (int coord = 0; coord < 4; ++coord) {
    sorted_boxes[index * 4 + coord] = boxes[source_index * 4 + coord];
  }
}

__global__ void NmsMaskKernel(int count, float iou_threshold, float offset,
                              const float* boxes, unsigned long long* mask,
                              int column_blocks) {
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
    if (DeviceBoxIoU(current_box, column_boxes + compare_index * 4, offset) >
        iou_threshold) {
      suppress_mask |= 1ULL << compare_index;
    }
  }

  mask[current_index * column_blocks + column_block] = suppress_mask;
}

}  // namespace

Tensor RunNmsOnDevice(const Tensor& boxes, const Tensor& scores,
                      NmsParameters parameters) {
  const int count = CheckedBoxCount(boxes);
  if (count == 0) {
    throw std::invalid_argument("NMS boxes tensor count must be positive");
  }

  CheckCuda(cudaSetDevice(boxes.device().id), "cudaSetDevice");

  const auto* boxes_data = static_cast<const float*>(boxes.data());
  const auto* scores_data = static_cast<const float*>(scores.data());

  thrust::device_vector<int64_t> order(count);
  thrust::sequence(order.begin(), order.end(), int64_t{0});
  thrust::sort(order.begin(), order.end(), ScoreDescending{scores_data});

  thrust::device_vector<float> sorted_boxes(static_cast<std::size_t>(count) *
                                            4U);
  const int gather_blocks = (count + kThreadsPerBlock - 1) / kThreadsPerBlock;
  GatherSortedBoxesKernel<<<gather_blocks, kThreadsPerBlock>>>(
      count, thrust::raw_pointer_cast(order.data()), boxes_data,
      thrust::raw_pointer_cast(sorted_boxes.data()));
  CheckCuda(cudaGetLastError(), "GatherSortedBoxesKernel");

  const int column_blocks = (count + kThreadsPerBlock - 1) / kThreadsPerBlock;
  thrust::device_vector<unsigned long long> mask(
      static_cast<std::size_t>(count) * column_blocks, 0);
  dim3 mask_blocks(column_blocks, column_blocks);
  NmsMaskKernel<<<mask_blocks, kThreadsPerBlock>>>(
      count, parameters.iou_threshold, parameters.coordinate_offset,
      thrust::raw_pointer_cast(sorted_boxes.data()),
      thrust::raw_pointer_cast(mask.data()), column_blocks);
  CheckCuda(cudaGetLastError(), "NmsMaskKernel");

  std::vector<unsigned long long> host_mask(mask.size());
  thrust::copy(mask.begin(), mask.end(), host_mask.begin());
  std::vector<int64_t> host_order(order.size());
  thrust::copy(order.begin(), order.end(), host_order.begin());

  std::vector<unsigned long long> suppressed(column_blocks, 0);
  std::vector<int64_t> keep;
  keep.reserve(static_cast<std::size_t>(count));
  for (int sorted_index = 0; sorted_index < count; ++sorted_index) {
    const int block_index = sorted_index / kThreadsPerBlock;
    const int bit_index = sorted_index % kThreadsPerBlock;
    if ((suppressed[block_index] & (1ULL << bit_index)) != 0) {
      continue;
    }

    keep.push_back(host_order[sorted_index]);
    if (parameters.max_output_boxes > 0 &&
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
  Tensor output = Tensor::Allocate(std::move(desc));
  CheckCuda(cudaMemcpy(output.data(), keep.data(), output.bytes(),
                       cudaMemcpyHostToDevice),
            "cudaMemcpy");
  return output;
}

}  // namespace mw::infer::postprocess_internal
