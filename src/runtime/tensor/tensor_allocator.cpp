#include "mw/infer/runtime/tensor/tensor_allocator.h"

#include <memory>
#include <stdexcept>
#include <utility>

namespace mw::infer {

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
