#ifndef MW_INFER_RUNTIME_POSTPROCESS_TOPK_H_
#define MW_INFER_RUNTIME_POSTPROCESS_TOPK_H_

#include <cstdint>

#include "mw/infer/runtime/tensor/tensor.h"

namespace mw::infer {

struct TopKResult {
  Tensor scores;
  Tensor indices;
};

TopKResult TopK(const Tensor& scores, int64_t k);

}  // namespace mw::infer

#endif  // MW_INFER_RUNTIME_POSTPROCESS_TOPK_H_
