#include <cuda_runtime_api.h>

#include <cfloat>
#include <limits>
#include <stdexcept>
#include <string>

#include "mw/infer/runtime/postprocess/softmax.h"

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

struct MatrixShape {
  int rows = 0;
  int columns = 0;
};

MatrixShape CheckedMatrixShape(const Tensor& tensor) {
  const int64_t rows = tensor.shape().size() == 2 ? tensor.shape()[0] : 1;
  const int64_t columns =
      tensor.shape().size() == 2 ? tensor.shape()[1] : tensor.shape()[0];
  if (rows > std::numeric_limits<int>::max()) {
    throw std::invalid_argument(
        "CUDA postprocess batch size exceeds int range");
  }
  if (columns > std::numeric_limits<int>::max()) {
    throw std::invalid_argument(
        "CUDA postprocess class count exceeds int range");
  }
  return MatrixShape{static_cast<int>(rows), static_cast<int>(columns)};
}

TensorDesc MakeOutputDesc(const Tensor& logits) {
  TensorDesc desc;
  desc.info.name = "softmax";
  desc.info.data_type = DataType::kFloat32;
  desc.info.shape = logits.shape();
  desc.device = logits.device();
  return desc;
}

__device__ float DeviceMax(float lhs, float rhs) {
  return lhs > rhs ? lhs : rhs;
}

__global__ void SoftmaxKernel(const float* logits, float* output, int rows,
                              int columns) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  __shared__ float shared[kThreadsPerBlock];
  const int row_offset = row * columns;

  float thread_max = -FLT_MAX;
  for (int column = threadIdx.x; column < columns; column += blockDim.x) {
    thread_max = DeviceMax(thread_max, logits[row_offset + column]);
  }
  shared[threadIdx.x] = thread_max;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] =
          DeviceMax(shared[threadIdx.x], shared[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  const float row_max = shared[0];

  float thread_sum = 0.0F;
  for (int column = threadIdx.x; column < columns; column += blockDim.x) {
    const float value = expf(logits[row_offset + column] - row_max);
    output[row_offset + column] = value;
    thread_sum += value;
  }
  shared[threadIdx.x] = thread_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float row_sum = shared[0];

  for (int column = threadIdx.x; column < columns; column += blockDim.x) {
    output[row_offset + column] /= row_sum;
  }
}

}  // namespace

Tensor RunSoftmaxOnDevice(const Tensor& logits) {
  const MatrixShape shape = CheckedMatrixShape(logits);
  CheckCuda(cudaSetDevice(logits.device().id), "cudaSetDevice");

  Tensor output = Tensor::Allocate(MakeOutputDesc(logits));
  SoftmaxKernel<<<shape.rows, kThreadsPerBlock>>>(
      static_cast<const float*>(logits.data()),
      static_cast<float*>(output.data()), shape.rows, shape.columns);
  CheckCuda(cudaGetLastError(), "SoftmaxKernel");
  return output;
}

}  // namespace mw::infer::postprocess_internal
