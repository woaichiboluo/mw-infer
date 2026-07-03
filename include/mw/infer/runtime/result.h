#ifndef MW_INFER_RESULT_H_
#define MW_INFER_RESULT_H_

#include <vector>

#include "mw/infer/common/tensor.h"

namespace mw::infer {

struct InferenceResult {
  std::vector<Tensor> outputs;
};

}  // namespace mw::infer

#endif  // MW_INFER_RESULT_H_
