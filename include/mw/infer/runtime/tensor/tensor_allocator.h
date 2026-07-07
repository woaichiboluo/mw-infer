#ifndef MW_INFER_RUNTIME_TENSOR_TENSOR_ALLOCATOR_H_
#define MW_INFER_RUNTIME_TENSOR_TENSOR_ALLOCATOR_H_

#include <cstddef>
#include <vector>

#include "mw/infer/runtime/tensor/tensor.h"

namespace mw::infer {

Tensor AllocateHostTensor(TensorDesc desc);

class TensorAllocationAdapter {
 public:
  virtual ~TensorAllocationAdapter() = default;

  virtual bool Supports(Device device) const = 0;
  virtual Tensor Allocate(TensorDesc desc) const = 0;
};

const TensorAllocationAdapter& GetHostTensorAllocationAdapter();

class TensorAllocator {
 public:
  TensorAllocator();
  explicit TensorAllocator(
      std::vector<const TensorAllocationAdapter*> adapters);

  bool Supports(Device device) const;
  Tensor Allocate(TensorDesc desc) const;

 private:
  std::vector<const TensorAllocationAdapter*> adapters_;
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
