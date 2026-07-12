#ifndef MW_INFER_RUNTIME_POSTPROCESS_YOLO_DECODE_H_
#define MW_INFER_RUNTIME_POSTPROCESS_YOLO_DECODE_H_

#include "mw/infer/runtime/tensor/tensor.h"
#include "mw/infer/runtime/tensor/tensor_allocator.h"

namespace mw::infer {

class ExecutionStream;

enum class YoloVersion {
  kYoloV5,
  kYoloV8,
  kYoloV11,
  kYoloV26,
};

enum class YoloTensorLayout {
  // Uses the official export layout when both candidate axes are plausible:
  // YOLOv5 is candidate-first, while newer versions are channel-first.
  kAuto,
  kChannelFirst,
  kCandidateFirst,
};

struct YoloDecodeOptions {
  YoloVersion version = YoloVersion::kYoloV8;
  YoloTensorLayout tensor_layout = YoloTensorLayout::kAuto;
};

struct YoloDecodeResult {
  Tensor boxes;   // float32 [B, N, 4], xyxy.
  Tensor scores;  // float32 [B, N, C].
};

// A provided CUDA stream orders decode after work already queued on that
// stream. Keep predictions and the result alive until the stream is
// synchronized.
YoloDecodeResult YoloDecode(
    const Tensor& predictions, YoloDecodeOptions options = {},
    TensorAllocator& allocator = TensorAllocator::Default(),
    ExecutionStream* execution_stream = nullptr);

}  // namespace mw::infer

#endif  // MW_INFER_RUNTIME_POSTPROCESS_YOLO_DECODE_H_
