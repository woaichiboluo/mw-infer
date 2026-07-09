#include "mw/infer/runtime/tensor/tensor_allocator.h"

#if defined(MW_INFER_HAS_CUDA_RUNTIME)
#include <cuda_runtime_api.h>
#endif

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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
  if (source.bytes() == 0) {
    return;
  }
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

DirectTensorAllocator::DirectTensorAllocator() {
  AddAdapter(CreateHostTensorAllocationAdapter());
#if defined(MW_INFER_HAS_CUDA_TENSOR_ADAPTER)
  AddAdapter(CreateCudaTensorAllocationAdapter());
#endif
}

DirectTensorAllocator::DirectTensorAllocator(
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

void DirectTensorAllocator::AddAdapter(
    std::unique_ptr<TensorAllocationAdapter> adapter) {
  if (!adapter) {
    throw std::invalid_argument("Tensor allocator adapter is null");
  }
  adapters_.push_back(std::move(adapter));
}

bool DirectTensorAllocator::Supports(Device device) const {
  for (const auto& adapter : adapters_) {
    if (adapter->Supports(device)) {
      return true;
    }
  }
  return false;
}

Tensor DirectTensorAllocator::Allocate(TensorDesc desc) {
  for (const auto& adapter : adapters_) {
    if (adapter->Supports(desc.device)) {
      return adapter->Allocate(std::move(desc));
    }
  }

  throw std::invalid_argument("No tensor allocation adapter supports device");
}

TensorAllocator& TensorAllocator::Default() {
  static DirectTensorAllocator allocator;
  return allocator;
}

Tensor Tensor::Allocate(TensorDesc desc, TensorAllocator& allocator) {
  return allocator.Allocate(std::move(desc));
}

Tensor Tensor::CopyTo(Device target_device, TensorAllocator& allocator) const {
  if (empty()) {
    throw std::invalid_argument("Cannot copy an empty tensor");
  }

  TensorDesc target_desc = desc_;
  target_desc.device = target_device;
  Tensor target = Tensor::Allocate(std::move(target_desc), allocator);
  CopyTensorBytes(*this, &target);
  return target;
}

void Tensor::CopyElementToHost(std::size_t element_offset, void* output,
                               std::size_t element_bytes) const {
  if (empty()) {
    throw std::invalid_argument("Tensor is empty");
  }
  if (output == nullptr) {
    throw std::invalid_argument("Tensor output data is null");
  }
  if (element_bytes == 0) {
    throw std::invalid_argument("Tensor element byte size is zero");
  }
  if (element_offset >= element_count()) {
    throw std::invalid_argument("Tensor index is out of range");
  }
  if (element_offset >
      std::numeric_limits<std::size_t>::max() / element_bytes) {
    throw std::invalid_argument("Tensor element byte offset overflows size_t");
  }

  const std::size_t byte_offset = element_offset * element_bytes;
  if (element_bytes > bytes_ || byte_offset > bytes_ - element_bytes) {
    throw std::invalid_argument("Tensor element byte range is out of range");
  }

  if (device().type == DeviceType::kCpu) {
    const auto* source = static_cast<const std::uint8_t*>(data()) + byte_offset;
    std::memcpy(output, source, element_bytes);
    return;
  }

#if defined(MW_INFER_HAS_CUDA_RUNTIME)
  if (device().type == DeviceType::kCuda) {
    CheckCuda(cudaSetDevice(device().id), "cudaSetDevice");
    const auto* source = static_cast<const std::uint8_t*>(data()) + byte_offset;
    CheckCuda(cudaMemcpy(output, source, element_bytes, cudaMemcpyDeviceToHost),
              "cudaMemcpy");
    return;
  }
#else
  if (device().type == DeviceType::kCuda) {
    throw std::runtime_error("CUDA tensor copy is unavailable in this build");
  }
#endif

  throw std::invalid_argument("Tensor device is unsupported");
}

struct PooledTensorAllocator::State {
  struct Block {
    Tensor owner;
    Device device;
    void* data = nullptr;
    std::size_t capacity_bytes = 0;
  };

  std::vector<std::shared_ptr<Block>> free_blocks;
};

namespace {

bool SameDevice(Device lhs, Device rhs) {
  return lhs.type == rhs.type && lhs.id == rhs.id;
}

}  // namespace

PooledTensorAllocator::PooledTensorAllocator()
    : PooledTensorAllocator(std::make_unique<DirectTensorAllocator>()) {}

PooledTensorAllocator::PooledTensorAllocator(
    std::unique_ptr<TensorAllocator> upstream)
    : upstream_(std::move(upstream)), state_(std::make_shared<State>()) {
  if (!upstream_) {
    throw std::invalid_argument("Pooled tensor allocator upstream is null");
  }
}

PooledTensorAllocator::~PooledTensorAllocator() = default;

bool PooledTensorAllocator::Supports(Device device) const {
  return upstream_->Supports(device);
}

Tensor PooledTensorAllocator::Allocate(TensorDesc desc) {
  if (!Supports(desc.device)) {
    throw std::invalid_argument("No tensor allocation adapter supports device");
  }

  const std::size_t bytes = TensorBytes(desc);
  auto& free_blocks = state_->free_blocks;
  auto block = std::shared_ptr<State::Block>();
  const auto block_it = std::find_if(
      free_blocks.begin(), free_blocks.end(),
      [device = desc.device, bytes](const std::shared_ptr<State::Block>& item) {
        return item && SameDevice(item->device, device) &&
               item->capacity_bytes >= bytes;
      });
  if (block_it != free_blocks.end()) {
    block = std::move(*block_it);
    free_blocks.erase(block_it);
  } else {
    Tensor owner = upstream_->Allocate(desc);
    block = std::make_shared<State::Block>();
    block->device = owner.device();
    block->data = owner.data();
    block->capacity_bytes = owner.capacity_bytes();
    block->owner = std::move(owner);
  }

  void* data = block->data;
  const std::size_t capacity_bytes = block->capacity_bytes;
  std::shared_ptr<State> state = state_;
  return Tensor::FromBuffer(std::move(desc), data, capacity_bytes,
                            [state, block = std::move(block)](void*) {
                              state->free_blocks.push_back(block);
                            });
}

void PooledTensorAllocator::Clear() { state_->free_blocks.clear(); }

}  // namespace mw::infer
