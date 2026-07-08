#ifndef MW_INFER_RUNTIME_POSTPROCESS_NMS_H_
#define MW_INFER_RUNTIME_POSTPROCESS_NMS_H_

#include <cstdint>
#include <vector>

#include "mw/infer/runtime/tensor/tensor.h"

namespace mw::infer {

struct NmsOptions {
  float iou_threshold = 0.5F;
  float coordinate_offset = 0.0F;
  int64_t max_output_boxes = 0;
};

Tensor CpuNms(const Tensor& boxes, const Tensor& scores, NmsOptions options);
Tensor CpuNms(const Tensor& boxes, const Tensor& scores, float iou_threshold);

bool CudaNmsAvailable();

Tensor Nms(const Tensor& boxes, const Tensor& scores, NmsOptions options);
Tensor Nms(const Tensor& boxes, const Tensor& scores, float iou_threshold);

}  // namespace mw::infer

#endif  // MW_INFER_RUNTIME_POSTPROCESS_NMS_H_
