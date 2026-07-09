#ifndef MW_INFER_RUNTIME_POSTPROCESS_NMS_H_
#define MW_INFER_RUNTIME_POSTPROCESS_NMS_H_

#include <cstdint>

#include "mw/infer/runtime/tensor/tensor.h"
#include "mw/infer/runtime/tensor/tensor_allocator.h"

namespace mw::infer {

Tensor Nms(const Tensor& boxes, const Tensor& scores, float iou_threshold,
           float coordinate_offset = 0.0F, int64_t max_output_boxes = 0,
           TensorAllocator& allocator = TensorAllocator::Default());

}  // namespace mw::infer

#endif  // MW_INFER_RUNTIME_POSTPROCESS_NMS_H_
