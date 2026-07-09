#ifndef MW_INFER_RUNTIME_PROCESS_NORMALIZE_H_
#define MW_INFER_RUNTIME_PROCESS_NORMALIZE_H_

#include <vector>

#include "mw/infer/runtime/process/image_to_tensor.h"
#include "mw/infer/runtime/tensor/tensor.h"

namespace mw::infer {

Tensor Normalize(const Tensor& input, const std::vector<float>& mean,
                 const std::vector<float>& stddev, float scale = 1.0F,
                 TensorLayout layout = TensorLayout::kBchw,
                 TensorAllocator& allocator = TensorAllocator::Default());

}  // namespace mw::infer

#endif  // MW_INFER_RUNTIME_PROCESS_NORMALIZE_H_
