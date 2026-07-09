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
  virtual Tensor Allocate(TensorDesc desc) = 0;
};

class DirectTensorAllocator final : public TensorAllocator {
 public:
  DirectTensorAllocator();
  explicit DirectTensorAllocator(
      std::vector<std::unique_ptr<TensorAllocationAdapter>> adapters);

  bool Supports(Device device) const override;
  Tensor Allocate(TensorDesc desc) override;

 private:
  void AddAdapter(std::unique_ptr<TensorAllocationAdapter> adapter);

  std::vector<std::unique_ptr<TensorAllocationAdapter>> adapters_;
};

class PooledTensorAllocator final : public TensorAllocator {
 public:
  PooledTensorAllocator();
  explicit PooledTensorAllocator(std::unique_ptr<TensorAllocator> upstream);
  ~PooledTensorAllocator() override;
  PooledTensorAllocator(const PooledTensorAllocator&) = delete;
  PooledTensorAllocator& operator=(const PooledTensorAllocator&) = delete;
  PooledTensorAllocator(PooledTensorAllocator&&) = delete;
  PooledTensorAllocator& operator=(PooledTensorAllocator&&) = delete;

  bool Supports(Device device) const override;
  Tensor Allocate(TensorDesc desc) override;
  void Clear();

 private:
  struct State;

  std::unique_ptr<TensorAllocator> upstream_;
  std::shared_ptr<State> state_;
};

}  // namespace mw::infer

#endif  // MW_INFER_RUNTIME_TENSOR_TENSOR_ALLOCATOR_H_
