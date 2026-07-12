#ifndef MW_INFER_RUNTIME_POSTPROCESS_NMS_H_
#define MW_INFER_RUNTIME_POSTPROCESS_NMS_H_

#include <cstdint>

#include "mw/infer/runtime/tensor/tensor.h"
#include "mw/infer/runtime/tensor/tensor_allocator.h"

namespace mw::infer {

class ExecutionStream;

// A provided CUDA stream orders NMS after work already queued on that stream.
// NMS synchronizes before returning because suppression is finalized on host.
Tensor Nms(const Tensor& boxes, const Tensor& scores, float iou_threshold,
           float coordinate_offset = 0.0F, int64_t max_output_boxes = 0,
           TensorAllocator& allocator = TensorAllocator::Default(),
           ExecutionStream* execution_stream = nullptr);

struct BatchNmsOptions {
  float score_threshold = 0.25F;
  float iou_threshold = 0.45F;
  int64_t max_detections = 300;
  bool class_agnostic = false;
};

struct BatchNmsResult {
  Tensor counts;     // [B], int64.
  Tensor boxes;      // [B, M, 4], float32.
  Tensor scores;     // [B, M], float32.
  Tensor class_ids;  // [B, M], int64; padding is -1.
  Tensor indices;    // [B, M], int64 indices into N; padding is -1.
};

// Runs NMS independently for each batch. boxes must be float32 [B, N, 4]
// and scores must be float32 [B, N, C]. M equals options.max_detections.
// Valid output rows are stable score-descending; remaining rows are padded.
// A provided CUDA stream orders NMS after work already queued on that stream.
// NMS synchronizes before returning because suppression is finalized on host.
BatchNmsResult BatchNms(
    const Tensor& boxes, const Tensor& scores,
    const BatchNmsOptions& options = {},
    TensorAllocator& allocator = TensorAllocator::Default(),
    ExecutionStream* execution_stream = nullptr);

}  // namespace mw::infer

#endif  // MW_INFER_RUNTIME_POSTPROCESS_NMS_H_
