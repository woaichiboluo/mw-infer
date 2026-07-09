#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mw/infer/runtime/tensor/gather.h"

namespace mw::infer::tensor_internal {
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

std::string MakeOutputName(const Tensor& data) {
  if (data.name().empty()) {
    return "gather_rows";
  }
  return data.name() + "_gathered";
}

std::size_t RowBytes(const Tensor& data) {
  TensorDesc desc;
  desc.info.data_type = data.data_type();
  desc.info.shape.assign(data.shape().begin() + 1, data.shape().end());
  return TensorBytes(desc);
}

TensorDesc MakeOutputDesc(const Tensor& data, int64_t selected_count) {
  TensorDesc desc;
  desc.info.name = MakeOutputName(data);
  desc.info.data_type = data.data_type();
  desc.info.shape = data.shape();
  desc.info.shape[0] = selected_count;
  desc.device = data.device();
  return desc;
}

int CheckedCount(int64_t value, const char* name) {
  if (value < 0) {
    throw std::invalid_argument(std::string(name) + " is negative");
  }
  if (value > std::numeric_limits<int>::max()) {
    throw std::invalid_argument(std::string(name) + " exceeds int range");
  }
  return static_cast<int>(value);
}

__global__ void GatherRowsKernel(const std::uint8_t* input,
                                 const int64_t* indices, std::uint8_t* output,
                                 int row_count, std::size_t row_bytes,
                                 int selected_count, int* has_error) {
  const int output_row = blockIdx.x;
  if (output_row >= selected_count) {
    return;
  }

  const int64_t input_row = indices[output_row];
  if (input_row < 0 || input_row >= row_count) {
    if (threadIdx.x == 0) {
      atomicExch(has_error, 1);
    }
    return;
  }

  const std::size_t output_offset =
      static_cast<std::size_t>(output_row) * row_bytes;
  const std::size_t input_offset =
      static_cast<std::size_t>(input_row) * row_bytes;
  for (std::size_t byte = threadIdx.x; byte < row_bytes; byte += blockDim.x) {
    output[output_offset + byte] = input[input_offset + byte];
  }
}

}  // namespace

Tensor RunGatherRowsOnDevice(const Tensor& data, const Tensor& indices,
                             TensorAllocator& allocator) {
  const int row_count = CheckedCount(data.shape()[0], "GatherRows row count");
  const int selected_count =
      CheckedCount(indices.shape()[0], "GatherRows selected count");
  const std::size_t row_bytes = RowBytes(data);

  CheckCuda(cudaSetDevice(data.device().id), "cudaSetDevice");
  Tensor output =
      Tensor::Allocate(MakeOutputDesc(data, indices.shape()[0]), allocator);
  if (selected_count == 0) {
    return output;
  }

  int* device_error = nullptr;
  CheckCuda(cudaMalloc(&device_error, sizeof(int)), "cudaMalloc");
  try {
    CheckCuda(cudaMemset(device_error, 0, sizeof(int)), "cudaMemset");
    GatherRowsKernel<<<selected_count, kThreadsPerBlock>>>(
        static_cast<const std::uint8_t*>(data.data()),
        static_cast<const int64_t*>(indices.data()),
        static_cast<std::uint8_t*>(output.data()), row_count, row_bytes,
        selected_count, device_error);
    CheckCuda(cudaGetLastError(), "GatherRowsKernel");

    int host_error = 0;
    CheckCuda(cudaMemcpy(&host_error, device_error, sizeof(int),
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy");
    CheckCuda(cudaFree(device_error), "cudaFree");
    device_error = nullptr;
    if (host_error != 0) {
      throw std::invalid_argument("GatherRows index is out of range");
    }
  } catch (...) {
    if (device_error != nullptr) {
      static_cast<void>(cudaFree(device_error));
    }
    throw;
  }

  return output;
}

}  // namespace mw::infer::tensor_internal
