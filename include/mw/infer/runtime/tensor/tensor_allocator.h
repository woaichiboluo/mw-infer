#ifndef MW_INFER_RUNTIME_TENSOR_TENSOR_ALLOCATOR_H_
#define MW_INFER_RUNTIME_TENSOR_TENSOR_ALLOCATOR_H_

#include <cstddef>
#include <memory>
#include <vector>

#include "mw/infer/runtime/tensor/tensor.h"

namespace mw::infer {

class TensorAllocationAdapter {
 public:
  virtual ~TensorAllocationAdapter() = default;

  virtual bool Supports(Device device) const = 0;
  virtual Tensor Allocate(TensorDesc desc) const = 0;
};

class TensorAllocator {
 public:
  TensorAllocator();
  explicit TensorAllocator(
      std::vector<std::unique_ptr<TensorAllocationAdapter>> adapters);

  bool Supports(Device device) const;
  Tensor Allocate(TensorDesc desc) const;

 private:
  void AddAdapter(std::unique_ptr<TensorAllocationAdapter> adapter);

  std::vector<std::unique_ptr<TensorAllocationAdapter>> adapters_;
};

class TensorBuffer {
 public:
  TensorBuffer() = default;
  explicit TensorBuffer(TensorAllocator allocator);

  const Tensor& tensor() const;
  std::size_t capacity_bytes() const;

  Tensor Ensure(TensorDesc desc);

 private:
  TensorAllocator allocator_;
  Tensor tensor_;
};

}  // namespace mw::infer

#endif  // MW_INFER_RUNTIME_TENSOR_TENSOR_ALLOCATOR_H_
