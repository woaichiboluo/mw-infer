#ifndef MW_INFER_RUNTIME_TENSOR_GATHER_H_
#define MW_INFER_RUNTIME_TENSOR_GATHER_H_

#include "mw/infer/runtime/tensor/tensor.h"
#include "mw/infer/runtime/tensor/tensor_allocator.h"

namespace mw::infer {

// A provided CUDA stream orders gather after work already queued on that
// stream. Gather synchronizes before returning to report invalid indices.
Tensor GatherRows(const Tensor& data, const Tensor& indices,
                  TensorAllocator& allocator = TensorAllocator::Default(),
                  ExecutionStream* execution_stream = nullptr);

}  // namespace mw::infer

#endif  // MW_INFER_RUNTIME_TENSOR_GATHER_H_
