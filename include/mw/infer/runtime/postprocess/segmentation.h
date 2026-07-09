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
  Tensor scores;
};

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
