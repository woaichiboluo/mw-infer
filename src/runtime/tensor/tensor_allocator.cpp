#include "mw/infer/runtime/tensor/tensor_allocator.h"

#if defined(MW_INFER_HAS_CUDA_RUNTIME)
#include <cuda_runtime_api.h>
#endif

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

struct PooledTensorAllocator::Block {
  explicit Block(Tensor storage) : storage(std::move(storage)) {}

  Tensor storage;
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
    : upstream_(std::move(upstream)) {
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
  if (bytes == 0) {
    return upstream_->Allocate(std::move(desc));
  }

  auto best_fit = blocks_.end();
  auto replacement = blocks_.end();
  for (auto block_it = blocks_.begin(); block_it != blocks_.end(); ++block_it) {
    // The pool owns one reference. Additional references belong to tensors or
    // views that still use this block.
    if (block_it->use_count() != 1 ||
        !SameDevice((*block_it)->storage.device(), desc.device)) {
      continue;
    }

    const std::size_t capacity_bytes = (*block_it)->storage.capacity_bytes();
    if (capacity_bytes >= bytes) {
      if (best_fit == blocks_.end() ||
          capacity_bytes < (*best_fit)->storage.capacity_bytes()) {
        best_fit = block_it;
      }
    } else if (replacement == blocks_.end() ||
               capacity_bytes > (*replacement)->storage.capacity_bytes()) {
      replacement = block_it;
    }
  }

  std::shared_ptr<Block> block;
  if (best_fit != blocks_.end()) {
    block = *best_fit;
  } else {
    if (replacement != blocks_.end()) {
      // Release an idle undersized block before allocating its replacement so
      // growing dynamic shapes do not require both buffers to coexist.
      blocks_.erase(replacement);
    }
    block = std::make_shared<Block>(upstream_->Allocate(desc));
    // An idle undersized block is replaced rather than accumulated. Therefore,
    // each device's block count grows only when all of its blocks are active.
    blocks_.push_back(block);
  }

  std::shared_ptr<void> owner = block;
  return Tensor::FromExternal(std::move(desc), block->storage.data(),
                              block->storage.capacity_bytes(),
                              std::move(owner));
}

void PooledTensorAllocator::Clear() {
  std::vector<std::shared_ptr<Block>> blocks;
  blocks.swap(blocks_);
}

}  // namespace mw::infer
