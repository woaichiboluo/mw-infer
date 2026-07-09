#include <cuda_runtime_api.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "mw/infer/runtime/process/channel.h"

namespace mw::infer::process_internal {
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
class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t count) : count_(count) {
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)),
              "cudaMalloc");
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  ~DeviceBuffer() {
    if (data_ != nullptr) {
      static_cast<void>(cudaFree(data_));
    }
  }

  void CopyFromHost(const T* source) {
    CheckCuda(
        cudaMemcpy(data_, source, count_ * sizeof(T), cudaMemcpyHostToDevice),
        "cudaMemcpy");
  }

  T* data() { return data_; }

 private:
  T* data_ = nullptr;
  std::size_t count_ = 0;
};

struct ImageTensorShape {
  int64_t batch = 0;
  int64_t channels = 0;
  int64_t height = 0;
  int64_t width = 0;
};

ImageTensorShape TensorShape(const Tensor& tensor, TensorLayout layout) {
  ImageTensorShape shape;
  shape.batch = tensor.shape()[0];
  switch (layout) {
    case TensorLayout::kBchw:
      shape.channels = tensor.shape()[1];
      shape.height = tensor.shape()[2];
      shape.width = tensor.shape()[3];
      break;
    case TensorLayout::kBhwc:
      shape.height = tensor.shape()[1];
      shape.width = tensor.shape()[2];
      shape.channels = tensor.shape()[3];
      break;
  }
  return shape;
}

TensorDesc MakeOutputDesc(const Tensor& input) {
  TensorDesc desc = input.desc();
  return desc;
}

int GridBlocks(std::size_t element_count) {
  const std::size_t blocks =
      (element_count + kThreadsPerBlock - 1) / kThreadsPerBlock;
  if (blocks > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("CUDA preprocess tensor is too large");
  }
  return static_cast<int>(blocks);
}

__global__ void ReorderChannelsKernel(const float* input, float* output,
                                      const int64_t* order,
                                      int64_t element_count, int64_t channels,
                                      int64_t height, int64_t width,
                                      int layout) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= element_count) {
    return;
  }

  if (layout == 0) {
    const int64_t spatial = height * width;
    const int64_t batch = index / (channels * spatial);
    const int64_t rem = index % (channels * spatial);
    const int64_t channel = rem / spatial;
    const int64_t pixel = rem % spatial;
    output[index] =
        input[(batch * channels + order[channel]) * spatial + pixel];
    return;
  }

  const int64_t channel = index % channels;
  output[index] = input[index - channel + order[channel]];
}

}  // namespace

Tensor RunReorderChannelsOnDevice(const Tensor& input,
                                  const std::vector<int64_t>& order,
                                  TensorLayout layout,
                                  TensorAllocator& allocator) {
  CheckCuda(cudaSetDevice(input.device().id), "cudaSetDevice");

  DeviceBuffer<int64_t> device_order(order.size());
  device_order.CopyFromHost(order.data());

  Tensor output = Tensor::Allocate(MakeOutputDesc(input), allocator);
  const ImageTensorShape shape = TensorShape(input, layout);
  const int layout_id = layout == TensorLayout::kBchw ? 0 : 1;
  ReorderChannelsKernel<<<GridBlocks(input.element_count()),
                          kThreadsPerBlock>>>(
      static_cast<const float*>(input.data()),
      static_cast<float*>(output.data()), device_order.data(),
      static_cast<int64_t>(input.element_count()), shape.channels, shape.height,
      shape.width, layout_id);
  CheckCuda(cudaGetLastError(), "ReorderChannelsKernel");
  return output;
}

}  // namespace mw::infer::process_internal
