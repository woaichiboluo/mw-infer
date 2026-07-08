#ifndef MW_INFER_RUNTIME_POSTPROCESS_SOFTMAX_H_
#define MW_INFER_RUNTIME_POSTPROCESS_SOFTMAX_H_

#include "mw/infer/runtime/tensor/tensor.h"

namespace mw::infer {

Tensor Softmax(const Tensor& logits);

}  // namespace mw::infer

#endif  // MW_INFER_RUNTIME_POSTPROCESS_SOFTMAX_H_
