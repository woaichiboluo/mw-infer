#ifndef MW_INFER_RUNTIME_POSTPROCESS_YOLO_DECODE_H_
#define MW_INFER_RUNTIME_POSTPROCESS_YOLO_DECODE_H_

#include "mw/infer/runtime/tensor/tensor.h"
#include "mw/infer/runtime/tensor/tensor_allocator.h"

namespace mw::infer {

enum class YoloVersion {
  kYoloV5,
  kYoloV8,
  kYoloV11,
};

struct YoloDecodeOptions {
  YoloVersion version = YoloVersion::kYoloV8;
  float score_threshold = 0.25F;
  float class_offset = 4096.0F;
};

struct YoloDecodeResult {
  Tensor boxes;
  Tensor nms_boxes;
  Tensor scores;
  Tensor class_ids;
};

YoloDecodeResult YoloDecode(
    const Tensor& predictions, YoloDecodeOptions options = {},
    TensorAllocator& allocator = TensorAllocator::Default());

}  // namespace mw::infer

#endif  // MW_INFER_RUNTIME_POSTPROCESS_YOLO_DECODE_H_
