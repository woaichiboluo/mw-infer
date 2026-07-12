#ifndef MW_INFER_RUNTIME_POSTPROCESS_YOLO_SEG_DECODE_H_
#define MW_INFER_RUNTIME_POSTPROCESS_YOLO_SEG_DECODE_H_

#include <cstdint>

#include "mw/infer/runtime/input/input.h"
#include "mw/infer/runtime/postprocess/yolo_decode.h"
#include "mw/infer/runtime/tensor/tensor.h"
#include "mw/infer/runtime/tensor/tensor_allocator.h"

namespace mw::infer {

enum class YoloSegPredictionLayout {
  // [B, C, N] or [B, N, C]: xywh, optional objectness, class scores,
  // followed by mask coefficients. Class-aware NMS is applied.
  kRaw,
  // [B, N, 6 + M] or [B, 6 + M, N]: xyxy, score, class id, followed
  // by mask coefficients. Candidates are already selected and score-sorted
  // within each batch, so NMS is skipped.
  kSelected,
};

enum class YoloSegTensorLayout {
  // Uses the official export layout when both candidate axes are plausible:
  // YOLOv5 raw is candidate-first, while newer raw outputs are channel-first;
  // selected outputs are candidate-first.
  kAuto,
  kChannelFirst,
  kCandidateFirst,
};

struct YoloSegDecodeOptions {
  YoloVersion version = YoloVersion::kYoloV8;
  YoloSegPredictionLayout prediction_layout = YoloSegPredictionLayout::kRaw;
  YoloSegTensorLayout tensor_layout = YoloSegTensorLayout::kAuto;
  float score_threshold = 0.25F;
  float iou_threshold = 0.45F;
  float mask_threshold = 0.5F;
  // Maximum detections per batch. Zero keeps every selected detection.
  int64_t max_detections = 300;
};

struct YoloSegDecodeResult {
  // Boxes and masks use the letterboxed model input coordinate system.
  Tensor boxes;      // [K, 4], float32 xyxy.
  Tensor scores;     // [K], float32.
  Tensor class_ids;  // [K], int64.
  Tensor batch_ids;  // [K], int64.
  // Dense masks can be large when K, batch size, or input resolution is large.
  Tensor masks;  // [K, input_height, input_width], uint8 values 0 or 1.
};

// Without an execution stream, CUDA inputs must be ready before this call. A
// provided stream orders decode after work already queued on that stream. The
// function synchronizes before returning, so results are ready on any stream.
YoloSegDecodeResult YoloSegDecode(
    const Tensor& predictions, const Tensor& prototypes, ImageSize input_size,
    YoloSegDecodeOptions options = {},
    TensorAllocator& allocator = TensorAllocator::Default(),
    ExecutionStream* execution_stream = nullptr);

}  // namespace mw::infer

#endif  // MW_INFER_RUNTIME_POSTPROCESS_YOLO_SEG_DECODE_H_
