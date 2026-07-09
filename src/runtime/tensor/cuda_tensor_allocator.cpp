#include <cuda_runtime_api.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mw/infer/runtime/tensor/tensor_allocator.h"

namespace mw::infer {
namespace {

std::string CudaErrorMessage(cudaError_t status, const char* operation) {
  return std::string(operation) + " failed: " + cudaGetErrorString(status);
}

void CheckCuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(CudaErrorMessage(status, operation));
  }
}

Tensor AllocateCudaTensor(TensorDesc desc) {
  if (desc.device.type != DeviceType::kCuda) {
    throw std::invalid_argument(
        "CUDA tensor allocation requires a CUDA device");
  }
  if (desc.device.id < 0) {
    throw std::invalid_argument("CUDA tensor device id is negative");
  }

  const int device_id = desc.device.id;
  CheckCuda(cudaSetDevice(device_id), "cudaSetDevice");

  const std::size_t bytes = TensorBytes(desc);
  if (bytes == 0) {
    return Tensor::FromBuffer(std::move(desc), nullptr, 0, [](void*) {});
  }

  void* data = nullptr;
  CheckCuda(cudaMalloc(&data, bytes), "cudaMalloc");

  try {
    return Tensor::FromBuffer(std::move(desc), data, bytes,
                              [device_id](void* ptr) {
                                if (ptr == nullptr) {
                                  return;
                                }
                                static_cast<void>(cudaSetDevice(device_id));
                                static_cast<void>(cudaFree(ptr));
                              });
  } catch (...) {
    static_cast<void>(cudaFree(data));
    throw;
  }
}

class CudaTensorAllocationAdapter final : public TensorAllocationAdapter {
 public:
  bool Supports(Device device) const override {
    return device.type == DeviceType::kCuda;
  }

  Tensor Allocate(TensorDesc desc) override {
    return AllocateCudaTensor(std::move(desc));
  }
};

}  // namespace

std::unique_ptr<TensorAllocationAdapter> CreateCudaTensorAllocationAdapter() {
  return std::make_unique<CudaTensorAllocationAdapter>();
}

}  // namespace mw::infer
