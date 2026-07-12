#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
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

std::size_t AlignUp(std::size_t value, std::size_t alignment) {
  const std::size_t remainder = value % alignment;
  if (remainder == 0) {
    return value;
  }
  const std::size_t padding = alignment - remainder;
  if (value > std::numeric_limits<std::size_t>::max() - padding) {
    throw std::invalid_argument("CUDA preprocess storage size overflows");
  }
  return value + padding;
}

Tensor AllocateOutputAndOrder(const Tensor& input, std::size_t order_count,
                              TensorAllocator& allocator,
                              const std::vector<int64_t>& order,
                              int64_t** device_order,
                              const int64_t** host_order) {
  const std::size_t order_offset = AlignUp(input.bytes(), alignof(int64_t));
  if (order_count > std::numeric_limits<std::size_t>::max() / sizeof(int64_t)) {
    throw std::invalid_argument("CUDA channel order size overflows");
  }
  const std::size_t order_bytes = order_count * sizeof(int64_t);
  if (order_offset > std::numeric_limits<std::size_t>::max() - order_bytes) {
    throw std::invalid_argument("CUDA preprocess storage size overflows");
  }
  const std::size_t storage_bytes = order_offset + order_bytes;
  if (storage_bytes >
      static_cast<std::size_t>(std::numeric_limits<int64_t>::max())) {
    throw std::invalid_argument("CUDA preprocess storage exceeds int64 range");
  }

  TensorDesc storage_desc = input.desc();
  storage_desc.info.data_type = DataType::kUInt8;
  storage_desc.info.shape = {static_cast<int64_t>(storage_bytes)};
  struct OutputOwner {
    Tensor storage;
    Tensor input;
    std::vector<int64_t> order;
  };
  auto owner = std::make_shared<OutputOwner>();
  owner->storage = Tensor::Allocate(std::move(storage_desc), allocator);
  owner->input = input;
  owner->order = order;
  *device_order = reinterpret_cast<int64_t*>(
      static_cast<std::byte*>(owner->storage.data()) + order_offset);
  *host_order = owner->order.data();
  std::shared_ptr<void> output_owner = owner;
  return Tensor::FromExternal(MakeOutputDesc(input), owner->storage.data(),
                              input.bytes(),
                              std::move(output_owner));
}

void SynchronizeNoThrow(ExecutionStream* stream,
                        cudaStream_t cuda_stream) noexcept {
  if (stream != nullptr) {
    stream->SynchronizeNoThrow();
    return;
  }
  static_cast<void>(cudaStreamSynchronize(cuda_stream));
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
                                  TensorLayout layout, ExecutionStream* stream,
                                  TensorAllocator& allocator) {
  CheckCuda(cudaSetDevice(input.device().id), "cudaSetDevice");
  const cudaStream_t cuda_stream =
      stream == nullptr ? nullptr : stream->cuda_handle();
  int64_t* device_order = nullptr;
  const int64_t* host_order = nullptr;
  Tensor output = AllocateOutputAndOrder(input, order.size(), allocator, order,
                                         &device_order, &host_order);
  try {
    CheckCuda(cudaMemcpyAsync(device_order, host_order,
                              order.size() * sizeof(int64_t),
                              cudaMemcpyHostToDevice, cuda_stream),
              "cudaMemcpyAsync");
    const ImageTensorShape shape = TensorShape(input, layout);
    const int layout_id = layout == TensorLayout::kBchw ? 0 : 1;
    ReorderChannelsKernel<<<GridBlocks(input.element_count()),
                            kThreadsPerBlock, 0, cuda_stream>>>(
        static_cast<const float*>(input.data()),
        static_cast<float*>(output.data()), device_order,
        static_cast<int64_t>(input.element_count()), shape.channels,
        shape.height, shape.width, layout_id);
    CheckCuda(cudaGetLastError(), "ReorderChannelsKernel");
    if (stream == nullptr) {
      CheckCuda(cudaStreamSynchronize(cuda_stream), "cudaStreamSynchronize");
    }
  } catch (...) {
    SynchronizeNoThrow(stream, cuda_stream);
    throw;
  }
  return output;
}

}  // namespace mw::infer::process_internal
