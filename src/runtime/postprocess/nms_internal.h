#ifndef MW_INFER_SRC_RUNTIME_POSTPROCESS_NMS_INTERNAL_H_
#define MW_INFER_SRC_RUNTIME_POSTPROCESS_NMS_INTERNAL_H_

#include <cstdint>
#include <vector>

#include "mw/infer/runtime/postprocess/nms.h"

namespace mw::infer::postprocess_internal {

struct NmsParameters {
  float iou_threshold = 0.5F;
  float coordinate_offset = 0.0F;
  int64_t max_output_boxes = 0;
};

struct BatchNmsHostResult {
  std::vector<int64_t> counts;
  std::vector<float> boxes;
  std::vector<float> scores;
  std::vector<int64_t> class_ids;
  std::vector<int64_t> indices;
};

BatchNmsHostResult RunBatchNmsOnHostBuffers(
    const float* boxes, const float* scores, int64_t batch_count,
    int64_t box_count, int64_t class_count,
    const BatchNmsOptions& options);

Tensor RunClassAwareBatchNms(
    const Tensor& boxes, const Tensor& scores, const Tensor& class_ids,
    const Tensor& batch_ids, float iou_threshold,
    float coordinate_offset = 0.0F, int64_t max_output_boxes = 0,
    TensorAllocator& allocator = TensorAllocator::Default(),
    ExecutionStream* execution_stream = nullptr);

#if defined(MW_INFER_HAS_CUDA_POSTPROCESS)

Tensor RunNmsOnDevice(const Tensor& boxes, const Tensor& scores,
                      const Tensor* class_ids, const Tensor* batch_ids,
                      NmsParameters parameters, TensorAllocator& allocator,
                      ExecutionStream* execution_stream);

BatchNmsResult RunBatchNmsOnDevice(
    const Tensor& boxes, const Tensor& scores,
    const BatchNmsOptions& options, TensorAllocator& allocator,
    ExecutionStream* execution_stream);

#endif

}  // namespace mw::infer::postprocess_internal

#endif  // MW_INFER_SRC_RUNTIME_POSTPROCESS_NMS_INTERNAL_H_
