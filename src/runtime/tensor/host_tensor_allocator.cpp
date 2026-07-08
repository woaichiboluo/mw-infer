#include <memory>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

#include "mw/infer/runtime/tensor/tensor_allocator.h"

namespace mw::infer {
namespace {

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

std::unique_ptr<TensorAllocationAdapter> CreateHostTensorAllocationAdapter() {
  return std::make_unique<HostTensorAllocationAdapter>();
}

}  // namespace mw::infer
