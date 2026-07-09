#ifndef MW_INFER_SRC_RUNTIME_POSTPROCESS_NMS_INTERNAL_H_
#define MW_INFER_SRC_RUNTIME_POSTPROCESS_NMS_INTERNAL_H_

#include <cstdint>

#include "mw/infer/runtime/postprocess/nms.h"

namespace mw::infer::postprocess_internal {

struct NmsParameters {
  float iou_threshold = 0.5F;
  float coordinate_offset = 0.0F;
  int64_t max_output_boxes = 0;
};

#if defined(MW_INFER_HAS_CUDA_POSTPROCESS)

Tensor RunNmsOnDevice(const Tensor& boxes, const Tensor& scores,
                      NmsParameters parameters, TensorAllocator& allocator);

#endif

}  // namespace mw::infer::postprocess_internal

#endif  // MW_INFER_SRC_RUNTIME_POSTPROCESS_NMS_INTERNAL_H_
