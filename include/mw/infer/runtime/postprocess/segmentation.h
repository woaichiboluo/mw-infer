#ifndef MW_INFER_RUNTIME_POSTPROCESS_SEGMENTATION_H_
#define MW_INFER_RUNTIME_POSTPROCESS_SEGMENTATION_H_

#include <vector>

#include "mw/infer/runtime/process/geometry.h"
#include "mw/infer/runtime/tensor/tensor.h"
#include "mw/infer/runtime/tensor/tensor_allocator.h"

namespace mw::infer {

struct SemanticSegmentationOptions {
  float binary_threshold = 0.5F;
};

struct SemanticSegmentationResult {
  Tensor class_ids;
  // Selected class probabilities with shape [N, H, W]. For C > 1 logits this
  // is the selected channel softmax probability. For C == 1 logits this is the
  // selected binary class probability after sigmoid and binary_threshold.
  Tensor scores;
};

// Restores [N, C, H, W] logits to the original image size recorded in traces.
// The returned tensor is dense, so every sample in the batch must restore to
// the same H/W. When original image sizes differ, split the logits into N == 1
// tensors and call this with one matching trace per tensor.
Tensor RestoreSegmentationLogits(
    const Tensor& logits, const std::vector<GeometryTrace>& traces,
    TensorAllocator& allocator = TensorAllocator::Default());

SemanticSegmentationResult SemanticSegmentation(
    const Tensor& logits, const SemanticSegmentationOptions& options = {},
    TensorAllocator& allocator = TensorAllocator::Default());

SemanticSegmentationResult SemanticSegmentation(
    const Tensor& logits, const std::vector<GeometryTrace>& traces,
    const SemanticSegmentationOptions& options = {},
    TensorAllocator& allocator = TensorAllocator::Default());

}  // namespace mw::infer

#endif  // MW_INFER_RUNTIME_POSTPROCESS_SEGMENTATION_H_
