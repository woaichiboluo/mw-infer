#include "mw/infer/runtime/tensor/tensor_allocator.h"

#if defined(MW_INFER_HAS_CUDA_RUNTIME)
#include <cuda_runtime_api.h>
#endif

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace mw::infer {

namespace {

#if defined(MW_INFER_HAS_CUDA_RUNTIME)
std::string CudaErrorMessage(cudaError_t status, const char* operation) {
  return std::string(operation) + " failed: " + cudaGetErrorString(status);
}

void CheckCuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(CudaErrorMessage(status, operation));
  }
}
#endif

void CopyTensorBytes(const Tensor& source, Tensor* target) {
  if (source.bytes() != target->bytes()) {
    throw std::invalid_argument("Tensor copy byte size mismatch");
  }

  const Device source_device = source.device();
  const Device target_device = target->device();
  if (source_device.type == DeviceType::kCpu &&
      target_device.type == DeviceType::kCpu) {
    std::memcpy(target->data(), source.data(), source.bytes());
    return;
  }

#if defined(MW_INFER_HAS_CUDA_RUNTIME)
  if (source_device.type == DeviceType::kCpu &&
      target_device.type == DeviceType::kCuda) {
    CheckCuda(cudaSetDevice(target_device.id), "cudaSetDevice");
    CheckCuda(cudaMemcpy(target->data(), source.data(), source.bytes(),
                         cudaMemcpyHostToDevice),
              "cudaMemcpy");
    return;
  }
  if (source_device.type == DeviceType::kCuda &&
      target_device.type == DeviceType::kCpu) {
    CheckCuda(cudaSetDevice(source_device.id), "cudaSetDevice");
    CheckCuda(cudaMemcpy(target->data(), source.data(), source.bytes(),
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy");
    return;
  }
  if (source_device.type == DeviceType::kCuda &&
      target_device.type == DeviceType::kCuda) {
    if (source_device.id == target_device.id) {
      CheckCuda(cudaSetDevice(target_device.id), "cudaSetDevice");
      CheckCuda(cudaMemcpy(target->data(), source.data(), source.bytes(),
                           cudaMemcpyDeviceToDevice),
                "cudaMemcpy");
      return;
    }

    CheckCuda(cudaMemcpyPeer(target->data(), target_device.id, source.data(),
                             source_device.id, source.bytes()),
              "cudaMemcpyPeer");
    return;
  }
#else
  if (source_device.type == DeviceType::kCuda ||
      target_device.type == DeviceType::kCuda) {
    throw std::runtime_error("CUDA tensor copy is unavailable in this build");
  }
#endif

  throw std::invalid_argument("Tensor copy device is unsupported");
}

}  // namespace

std::unique_ptr<TensorAllocationAdapter> CreateHostTensorAllocationAdapter();

#if defined(MW_INFER_HAS_CUDA_TENSOR_ADAPTER)
std::unique_ptr<TensorAllocationAdapter> CreateCudaTensorAllocationAdapter();
#endif

TensorAllocator::TensorAllocator() {
  AddAdapter(CreateHostTensorAllocationAdapter());
#if defined(MW_INFER_HAS_CUDA_TENSOR_ADAPTER)
  AddAdapter(CreateCudaTensorAllocationAdapter());
#endif
}

TensorAllocator::TensorAllocator(
    std::vector<std::unique_ptr<TensorAllocationAdapter>> adapters)
    : adapters_(std::move(adapters)) {
  if (adapters_.empty()) {
    throw std::invalid_argument("Tensor allocator has no adapters");
  }
  for (const auto& adapter : adapters_) {
    if (!adapter) {
      throw std::invalid_argument("Tensor allocator adapter is null");
    }
  }
}

void TensorAllocator::AddAdapter(
    std::unique_ptr<TensorAllocationAdapter> adapter) {
  if (!adapter) {
    throw std::invalid_argument("Tensor allocator adapter is null");
  }
  adapters_.push_back(std::move(adapter));
}

bool TensorAllocator::Supports(Device device) const {
  for (const auto& adapter : adapters_) {
    if (adapter->Supports(device)) {
      return true;
    }
  }
  return false;
}

Tensor TensorAllocator::Allocate(TensorDesc desc) const {
  for (const auto& adapter : adapters_) {
    if (adapter->Supports(desc.device)) {
      return adapter->Allocate(std::move(desc));
    }
  }

  throw std::invalid_argument("No tensor allocation adapter supports device");
}

Tensor Tensor::Allocate(TensorDesc desc) {
  return TensorAllocator().Allocate(std::move(desc));
}

Tensor Tensor::Allocate(TensorDesc desc, const TensorAllocator& allocator) {
  return allocator.Allocate(std::move(desc));
}

Tensor Tensor::CopyTo(Device target_device) const {
  if (empty()) {
    throw std::invalid_argument("Cannot copy an empty tensor");
  }

  TensorDesc target_desc = desc_;
  target_desc.device = target_device;
  Tensor target = Tensor::Allocate(std::move(target_desc));
  CopyTensorBytes(*this, &target);
  return target;
}

TensorBuffer::TensorBuffer(TensorAllocator allocator)
    : allocator_(std::move(allocator)) {}

const Tensor& TensorBuffer::tensor() const { return tensor_; }

std::size_t TensorBuffer::capacity_bytes() const {
  return tensor_.capacity_bytes();
}

Tensor TensorBuffer::Ensure(TensorDesc desc) {
  const bool same_device = !tensor_.empty() &&
                           tensor_.device().type == desc.device.type &&
                           tensor_.device().id == desc.device.id;
  if (!same_device || tensor_.capacity_bytes() < TensorBytes(desc)) {
    tensor_ = Tensor::Allocate(std::move(desc), allocator_);
    return tensor_;
  }

  tensor_ = tensor_.View(std::move(desc));
  return tensor_;
}

}  // namespace mw::infer
