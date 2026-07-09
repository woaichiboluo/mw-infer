#ifndef MW_INFER_RUNTIME_POSTPROCESS_SOFTMAX_H_
#define MW_INFER_RUNTIME_POSTPROCESS_SOFTMAX_H_

#include "mw/infer/runtime/tensor/tensor.h"
#include "mw/infer/runtime/tensor/tensor_allocator.h"

namespace mw::infer {

Tensor Softmax(const Tensor& logits,
               TensorAllocator& allocator = TensorAllocator::Default());

}  // namespace mw::infer

#endif  // MW_INFER_RUNTIME_POSTPROCESS_SOFTMAX_H_
