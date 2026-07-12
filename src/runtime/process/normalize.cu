#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mw/infer/runtime/process/normalize.h"

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

Tensor AllocateOutputAndParameters(const Tensor& input,
                                   const std::vector<float>& mean,
                                   const std::vector<float>& stddev,
                                   TensorAllocator& allocator,
                                   float** device_parameters,
                                   const float** host_parameters) {
  const std::size_t parameter_count = mean.size() + stddev.size();
  if (parameter_count >
      std::numeric_limits<std::size_t>::max() / sizeof(float)) {
    throw std::invalid_argument("CUDA normalize parameter size overflows");
  }
  const std::size_t parameter_bytes = parameter_count * sizeof(float);
  if (input.bytes() >
      std::numeric_limits<std::size_t>::max() - parameter_bytes) {
    throw std::invalid_argument("CUDA preprocess storage size overflows");
  }
  const std::size_t storage_bytes = input.bytes() + parameter_bytes;
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
    std::vector<float> parameters;
  };
  auto owner = std::make_shared<OutputOwner>();
  owner->storage = Tensor::Allocate(std::move(storage_desc), allocator);
  owner->input = input;
  owner->parameters.reserve(parameter_count);
  owner->parameters.insert(owner->parameters.end(), mean.begin(), mean.end());
  owner->parameters.insert(owner->parameters.end(), stddev.begin(),
                           stddev.end());
  *device_parameters = reinterpret_cast<float*>(
      static_cast<std::byte*>(owner->storage.data()) + input.bytes());
  *host_parameters = owner->parameters.data();
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

__global__ void NormalizeKernel(const float* input, float* output,
                                const float* mean, const float* stddev,
                                float scale, int64_t element_count,
                                int64_t channels, int64_t height, int64_t width,
                                int layout) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= element_count) {
    return;
  }

  int64_t channel = 0;
  if (layout == 0) {
    const int64_t spatial = height * width;
    channel = (index / spatial) % channels;
  } else {
    channel = index % channels;
  }

  output[index] = (input[index] * scale - mean[channel]) / stddev[channel];
}

}  // namespace

Tensor RunNormalizeOnDevice(const Tensor& input, const std::vector<float>& mean,
                            const std::vector<float>& stddev, float scale,
                            TensorLayout layout, ExecutionStream* stream,
                            TensorAllocator& allocator) {
  CheckCuda(cudaSetDevice(input.device().id), "cudaSetDevice");
  const cudaStream_t cuda_stream =
      stream == nullptr ? nullptr : stream->cuda_handle();
  float* device_mean = nullptr;
  const float* host_parameters = nullptr;
  Tensor output = AllocateOutputAndParameters(input, mean, stddev, allocator,
                                               &device_mean, &host_parameters);
  float* device_stddev = device_mean + mean.size();
  try {
    CheckCuda(cudaMemcpyAsync(device_mean, host_parameters,
                              mean.size() * sizeof(float),
                              cudaMemcpyHostToDevice, cuda_stream),
              "cudaMemcpyAsync");
    CheckCuda(cudaMemcpyAsync(device_stddev, host_parameters + mean.size(),
                              stddev.size() * sizeof(float),
                              cudaMemcpyHostToDevice, cuda_stream),
              "cudaMemcpyAsync");
    const ImageTensorShape shape = TensorShape(input, layout);
    const int layout_id = layout == TensorLayout::kBchw ? 0 : 1;
    NormalizeKernel<<<GridBlocks(input.element_count()), kThreadsPerBlock, 0,
                      cuda_stream>>>(
        static_cast<const float*>(input.data()),
        static_cast<float*>(output.data()), device_mean, device_stddev, scale,
        static_cast<int64_t>(input.element_count()), shape.channels,
        shape.height, shape.width, layout_id);
    CheckCuda(cudaGetLastError(), "NormalizeKernel");
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
