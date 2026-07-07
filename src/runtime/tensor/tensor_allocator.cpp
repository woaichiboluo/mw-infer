#include "mw/infer/runtime/tensor/tensor_allocator.h"

#include <new>
#include <stdexcept>
#include <utility>

namespace mw::infer {
namespace {

class HostTensorAllocationAdapter final : public TensorAllocationAdapter {
 public:
  bool Supports(Device device) const override {
    return device.type == DeviceType::kCpu;
  }

  Tensor Allocate(TensorDesc desc) const override {
    return AllocateHostTensor(std::move(desc));
  }
};

}  // namespace

Tensor AllocateHostTensor(TensorDesc desc) {
  if (desc.device.type != DeviceType::kCpu) {
    throw std::invalid_argument("Host tensor allocation requires a CPU device");
  }

  const std::size_t bytes = TensorBytes(desc);
  void* data = ::operator new(bytes);
  try {
    return Tensor::FromBuffer(std::move(desc), data, bytes,
                              [](void* ptr) { ::operator delete(ptr); });
  } catch (...) {
    ::operator delete(data);
    throw;
  }
}

const TensorAllocationAdapter& GetHostTensorAllocationAdapter() {
  static const HostTensorAllocationAdapter adapter;
  return adapter;
}

TensorAllocator::TensorAllocator()
    : TensorAllocator({&GetHostTensorAllocationAdapter()}) {}

TensorAllocator::TensorAllocator(
    std::vector<const TensorAllocationAdapter*> adapters)
    : adapters_(std::move(adapters)) {
  if (adapters_.empty()) {
    throw std::invalid_argument("Tensor allocator has no adapters");
  }
  for (const TensorAllocationAdapter* adapter : adapters_) {
    if (adapter == nullptr) {
      throw std::invalid_argument("Tensor allocator adapter is null");
    }
  }
}

bool TensorAllocator::Supports(Device device) const {
  for (const TensorAllocationAdapter* adapter : adapters_) {
    if (adapter->Supports(device)) {
      return true;
    }
  }
  return false;
}

Tensor TensorAllocator::Allocate(TensorDesc desc) const {
  for (const TensorAllocationAdapter* adapter : adapters_) {
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
