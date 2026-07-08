#ifndef MW_INFER_RUNTIME_TENSOR_GATHER_H_
#define MW_INFER_RUNTIME_TENSOR_GATHER_H_

#include "mw/infer/runtime/tensor/tensor.h"

namespace mw::infer {

Tensor GatherRows(const Tensor& data, const Tensor& indices);

}  // namespace mw::infer

#endif  // MW_INFER_RUNTIME_TENSOR_GATHER_H_
