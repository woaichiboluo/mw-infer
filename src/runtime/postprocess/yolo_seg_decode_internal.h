#ifndef MW_INFER_SRC_RUNTIME_POSTPROCESS_YOLO_SEG_DECODE_INTERNAL_H_
#define MW_INFER_SRC_RUNTIME_POSTPROCESS_YOLO_SEG_DECODE_INTERNAL_H_

#include <cstdint>

#include "mw/infer/runtime/postprocess/yolo_seg_decode.h"

namespace mw::infer::postprocess_internal {

struct YoloSegCandidates {
  Tensor boxes;
  Tensor nms_boxes;
  Tensor scores;
  Tensor class_ids;
  Tensor batch_ids;
  Tensor candidate_ids;
};

YoloSegCandidates RunRawYoloSegDecodeOnDevice(
    const Tensor& predictions, int64_t batch_count, int64_t channel_count,
    int64_t candidate_count, bool channel_first, int64_t class_count,
    YoloSegDecodeOptions options, TensorAllocator& allocator);

YoloSegCandidates RunSelectedYoloSegDecodeOnDevice(
    const Tensor& predictions, int64_t batch_count, int64_t channel_count,
    int64_t candidate_count, bool channel_first, YoloSegDecodeOptions options,
    TensorAllocator& allocator);

Tensor RunYoloSegMasksOnDevice(
    const Tensor& predictions, const Tensor& prototypes, const Tensor& boxes,
    const Tensor& batch_ids, const Tensor& candidate_ids, int64_t channel_count,
    int64_t candidate_count, bool channel_first, int64_t coefficient_start,
    bool scale_coefficients_by_objectness, ImageSize input_size,
    float mask_threshold, TensorAllocator& allocator);

}  // namespace mw::infer::postprocess_internal

#endif  // MW_INFER_SRC_RUNTIME_POSTPROCESS_YOLO_SEG_DECODE_INTERNAL_H_
