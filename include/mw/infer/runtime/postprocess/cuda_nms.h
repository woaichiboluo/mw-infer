#ifndef MW_INFER_RUNTIME_POSTPROCESS_CUDA_NMS_H_
#define MW_INFER_RUNTIME_POSTPROCESS_CUDA_NMS_H_

#include "mw/infer/runtime/postprocess/nms.h"

namespace mw::infer {

#if defined(MW_INFER_HAS_CUDA_NMS)

Tensor CudaNms(const Tensor& boxes, const Tensor& scores, NmsOptions options);
Tensor CudaNms(const Tensor& boxes, const Tensor& scores, float iou_threshold);

#endif

}  // namespace mw::infer

#endif  // MW_INFER_RUNTIME_POSTPROCESS_CUDA_NMS_H_
