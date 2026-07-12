#include "mw/infer/runtime/postprocess/nms.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mw/infer/runtime/execution_stream.h"
#include "nms_internal.h"

namespace mw::infer {

namespace {

using postprocess_internal::NmsParameters;

struct BatchNmsShape {
  int64_t batch_count = 0;
  int64_t box_count = 0;
  int64_t class_count = 0;
};

struct BatchCandidate {
  int64_t box_index = 0;
  int64_t class_id = 0;
  float score = 0.0F;
  std::size_t stable_index = 0;
};

void ValidateParameters(NmsParameters parameters) {
  if (!std::isfinite(parameters.iou_threshold) ||
      parameters.iou_threshold < 0.0F || parameters.iou_threshold > 1.0F) {
    throw std::invalid_argument("NMS IoU threshold must be in [0, 1]");
  }
  if (!std::isfinite(parameters.coordinate_offset) ||
      parameters.coordinate_offset < 0.0F) {
    throw std::invalid_argument("NMS coordinate offset must be non-negative");
  }
  if (parameters.max_output_boxes < 0) {
    throw std::invalid_argument("NMS max output boxes must be non-negative");
  }
}

void ValidateInputs(const Tensor& boxes, const Tensor& scores) {
  if (boxes.empty()) {
    throw std::invalid_argument("NMS boxes tensor is empty");
  }
  if (scores.empty()) {
    throw std::invalid_argument("NMS scores tensor is empty");
  }
  if (boxes.data_type() != DataType::kFloat32) {
    throw std::invalid_argument("NMS boxes tensor must be float32");
  }
  if (scores.data_type() != DataType::kFloat32) {
    throw std::invalid_argument("NMS scores tensor must be float32");
  }
  if (boxes.shape().size() != 2 || boxes.shape()[1] != 4) {
    throw std::invalid_argument("NMS boxes tensor shape must be [N, 4]");
  }
  if (scores.shape().size() != 1) {
    throw std::invalid_argument("NMS scores tensor shape must be [N]");
  }
  if (boxes.shape()[0] != scores.shape()[0]) {
    throw std::invalid_argument("NMS boxes and scores count mismatch");
  }
  if (boxes.device().type != scores.device().type ||
      boxes.device().id != scores.device().id) {
    throw std::invalid_argument("NMS boxes and scores device mismatch");
  }
}

void ValidateGroupIds(const Tensor& boxes, const Tensor& ids) {
  if (ids.empty()) {
    throw std::invalid_argument("NMS group id tensor is empty");
  }
  if (ids.data_type() != DataType::kInt64) {
    throw std::invalid_argument("NMS group id tensor must be int64");
  }
  if (ids.shape().size() != 1 || ids.shape()[0] != boxes.shape()[0]) {
    throw std::invalid_argument("NMS group id tensor shape must be [N]");
  }
  if (ids.device().type != boxes.device().type ||
      ids.device().id != boxes.device().id) {
    throw std::invalid_argument("NMS group id tensor device mismatch");
  }
}

BatchNmsShape ValidateBatchInputs(const Tensor& boxes, const Tensor& scores) {
  if (boxes.empty()) {
    throw std::invalid_argument("BatchNms boxes tensor is empty");
  }
  if (scores.empty()) {
    throw std::invalid_argument("BatchNms scores tensor is empty");
  }
  if (boxes.data_type() != DataType::kFloat32) {
    throw std::invalid_argument("BatchNms boxes tensor must be float32");
  }
  if (scores.data_type() != DataType::kFloat32) {
    throw std::invalid_argument("BatchNms scores tensor must be float32");
  }
  if (boxes.shape().size() != 3 || boxes.shape()[2] != 4) {
    throw std::invalid_argument("BatchNms boxes tensor shape must be [B, N, 4]");
  }
  if (scores.shape().size() != 3) {
    throw std::invalid_argument(
        "BatchNms scores tensor shape must be [B, N, C]");
  }
  if (boxes.shape()[0] != scores.shape()[0] ||
      boxes.shape()[1] != scores.shape()[1]) {
    throw std::invalid_argument("BatchNms boxes and scores shape mismatch");
  }
  if (scores.shape()[2] <= 0) {
    throw std::invalid_argument("BatchNms class count must be positive");
  }
  if (boxes.device().type != scores.device().type ||
      boxes.device().id != scores.device().id) {
    throw std::invalid_argument("BatchNms boxes and scores device mismatch");
  }
  return BatchNmsShape{boxes.shape()[0], boxes.shape()[1], scores.shape()[2]};
}

void ValidateBatchOptions(const BatchNmsOptions& options) {
  if (!std::isfinite(options.score_threshold)) {
    throw std::invalid_argument("BatchNms score threshold must be finite");
  }
  if (!std::isfinite(options.iou_threshold) || options.iou_threshold < 0.0F ||
      options.iou_threshold > 1.0F) {
    throw std::invalid_argument("BatchNms IoU threshold must be in [0, 1]");
  }
  if (options.max_detections <= 0) {
    throw std::invalid_argument("BatchNms max detections must be positive");
  }
}

void ValidateExecutionStream(const Tensor& tensor,
                             ExecutionStream* execution_stream) {
  if (execution_stream == nullptr) {
    return;
  }
  const Device stream_device = execution_stream->device();
  if (stream_device.type != tensor.device().type ||
      stream_device.id != tensor.device().id) {
    throw std::invalid_argument(
        "NMS execution stream device does not match tensor device");
  }
}

std::size_t BoxCount(const Tensor& boxes) {
  return static_cast<std::size_t>(boxes.shape()[0]);
}

float BoxArea(const float* boxes, int64_t index, float offset) {
  const float width =
      std::max(0.0F, boxes[index * 4 + 2] - boxes[index * 4] + offset);
  const float height =
      std::max(0.0F, boxes[index * 4 + 3] - boxes[index * 4 + 1] + offset);
  return width * height;
}

bool IsValidBatchBox(const float* boxes, int64_t index) {
  const float left = boxes[index * 4];
  const float top = boxes[index * 4 + 1];
  const float right = boxes[index * 4 + 2];
  const float bottom = boxes[index * 4 + 3];
  return std::isfinite(left) && std::isfinite(top) && std::isfinite(right) &&
         std::isfinite(bottom) && right > left && bottom > top;
}

float IoU(const float* boxes, const std::vector<float>& areas, int64_t lhs,
          int64_t rhs, float offset) {
  const float xx1 = std::max(boxes[lhs * 4], boxes[rhs * 4]);
  const float yy1 = std::max(boxes[lhs * 4 + 1], boxes[rhs * 4 + 1]);
  const float xx2 = std::min(boxes[lhs * 4 + 2], boxes[rhs * 4 + 2]);
  const float yy2 = std::min(boxes[lhs * 4 + 3], boxes[rhs * 4 + 3]);

  const float width = std::max(0.0F, xx2 - xx1 + offset);
  const float height = std::max(0.0F, yy2 - yy1 + offset);
  const float intersection = width * height;
  const float denominator = areas[lhs] + areas[rhs] - intersection;
  if (denominator <= 0.0F) {
    return 0.0F;
  }
  return intersection / denominator;
}

bool CandidateScoreDescending(const BatchCandidate& lhs,
                              const BatchCandidate& rhs) {
  if (lhs.score == rhs.score) {
    return lhs.stable_index < rhs.stable_index;
  }
  return lhs.score > rhs.score;
}

std::vector<BatchCandidate> SelectCandidateGroup(
    const float* boxes, const std::vector<float>& areas,
    std::vector<BatchCandidate> candidates, float iou_threshold,
    int64_t max_detections) {
  std::stable_sort(candidates.begin(), candidates.end(),
                   CandidateScoreDescending);
  std::vector<uint8_t> suppressed(candidates.size(), 0);
  std::vector<BatchCandidate> selected;
  selected.reserve(std::min(candidates.size(),
                            static_cast<std::size_t>(max_detections)));
  for (std::size_t index = 0; index < candidates.size(); ++index) {
    if (suppressed[index] != 0) {
      continue;
    }
    selected.push_back(candidates[index]);
    if (selected.size() == static_cast<std::size_t>(max_detections)) {
      break;
    }
    for (std::size_t next = index + 1; next < candidates.size(); ++next) {
      if (suppressed[next] != 0) {
        continue;
      }
      if (IoU(boxes, areas, candidates[index].box_index,
              candidates[next].box_index, 0.0F) > iou_threshold) {
        suppressed[next] = 1;
      }
    }
  }
  return selected;
}

TensorDesc MakeOutputDesc(std::string name, DataType data_type,
                          std::vector<int64_t> shape, Device device) {
  TensorDesc desc;
  desc.info.name = std::move(name);
  desc.info.data_type = data_type;
  desc.info.shape = std::move(shape);
  desc.device = device;
  return desc;
}

BatchNmsResult AllocateBatchResult(BatchNmsShape shape, int64_t max_detections,
                                   Device device, TensorAllocator& allocator) {
  Tensor counts = Tensor::Allocate(
      MakeOutputDesc("batch_nms_counts", DataType::kInt64,
                     {shape.batch_count}, device),
      allocator);
  Tensor boxes = Tensor::Allocate(
      MakeOutputDesc("batch_nms_boxes", DataType::kFloat32,
                     {shape.batch_count, max_detections, 4}, device),
      allocator);
  Tensor scores = Tensor::Allocate(
      MakeOutputDesc("batch_nms_scores", DataType::kFloat32,
                     {shape.batch_count, max_detections}, device),
      allocator);
  Tensor class_ids = Tensor::Allocate(
      MakeOutputDesc("batch_nms_class_ids", DataType::kInt64,
                     {shape.batch_count, max_detections}, device),
      allocator);
  Tensor indices = Tensor::Allocate(
      MakeOutputDesc("batch_nms_indices", DataType::kInt64,
                     {shape.batch_count, max_detections}, device),
      allocator);
  return BatchNmsResult{std::move(counts), std::move(boxes), std::move(scores),
                        std::move(class_ids), std::move(indices)};
}

postprocess_internal::BatchNmsHostResult ComputeBatchNmsOnHostBuffers(
    const float* input_boxes, const float* input_scores, BatchNmsShape shape,
    const BatchNmsOptions& options) {
  const std::size_t batches = static_cast<std::size_t>(shape.batch_count);
  const std::size_t max_detections =
      static_cast<std::size_t>(options.max_detections);
  if (max_detections != 0 &&
      batches > std::numeric_limits<std::size_t>::max() / max_detections) {
    throw std::invalid_argument("BatchNms output size overflows size_t");
  }
  const std::size_t output_slots =
      batches * max_detections;
  if (output_slots > std::numeric_limits<std::size_t>::max() / 4U) {
    throw std::invalid_argument("BatchNms boxes output size overflows size_t");
  }
  postprocess_internal::BatchNmsHostResult output;
  output.counts.resize(static_cast<std::size_t>(shape.batch_count), 0);
  output.boxes.resize(output_slots * 4U, 0.0F);
  output.scores.resize(output_slots, 0.0F);
  output.class_ids.resize(output_slots, -1);
  output.indices.resize(output_slots, -1);
  auto* output_counts = output.counts.data();
  auto* output_boxes = output.boxes.data();
  auto* output_scores = output.scores.data();
  auto* output_class_ids = output.class_ids.data();
  auto* output_indices = output.indices.data();
  const std::size_t box_count = static_cast<std::size_t>(shape.box_count);
  const std::size_t class_count = static_cast<std::size_t>(shape.class_count);
  for (int64_t batch = 0; batch < shape.batch_count; ++batch) {
    const std::size_t batch_index = static_cast<std::size_t>(batch);
    const float* batch_boxes = input_boxes + batch_index * box_count * 4U;
    const float* batch_scores =
        input_scores + batch_index * box_count * class_count;
    std::vector<float> areas(box_count);
    std::vector<uint8_t> valid_boxes(box_count, 0);
    for (int64_t box = 0; box < shape.box_count; ++box) {
      const std::size_t box_index = static_cast<std::size_t>(box);
      if (IsValidBatchBox(batch_boxes, box)) {
        valid_boxes[box_index] = 1;
        areas[box_index] = BoxArea(batch_boxes, box, 0.0F);
      }
    }

    std::vector<BatchCandidate> batch_selected;
    if (options.class_agnostic) {
      std::vector<BatchCandidate> candidates;
      candidates.reserve(box_count);
      for (int64_t box = 0; box < shape.box_count; ++box) {
        if (valid_boxes[static_cast<std::size_t>(box)] == 0) {
          continue;
        }
        const std::size_t score_offset =
            static_cast<std::size_t>(box) * class_count;
        float best_score = -std::numeric_limits<float>::infinity();
        int64_t best_class = 0;
        bool has_finite_score = false;
        for (int64_t class_id = 0; class_id < shape.class_count; ++class_id) {
          const float score =
              batch_scores[score_offset + static_cast<std::size_t>(class_id)];
          if (!std::isfinite(score)) {
            continue;
          }
          if (!has_finite_score || score > best_score) {
            best_score = score;
            best_class = class_id;
            has_finite_score = true;
          }
        }
        if (has_finite_score && best_score >= options.score_threshold) {
          candidates.push_back(BatchCandidate{
              box, best_class, best_score, static_cast<std::size_t>(box)});
        }
      }
      batch_selected = SelectCandidateGroup(
          batch_boxes, areas, std::move(candidates), options.iou_threshold,
          options.max_detections);
    } else {
      for (int64_t class_id = 0; class_id < shape.class_count; ++class_id) {
        std::vector<BatchCandidate> candidates;
        candidates.reserve(box_count);
        for (int64_t box = 0; box < shape.box_count; ++box) {
          if (valid_boxes[static_cast<std::size_t>(box)] == 0) {
            continue;
          }
          const std::size_t stable_index =
              static_cast<std::size_t>(box) * class_count +
              static_cast<std::size_t>(class_id);
          const float score = batch_scores[stable_index];
          if (std::isfinite(score) && score >= options.score_threshold) {
            candidates.push_back(
                BatchCandidate{box, class_id, score, stable_index});
          }
        }
        std::vector<BatchCandidate> class_selected = SelectCandidateGroup(
            batch_boxes, areas, std::move(candidates), options.iou_threshold,
            options.max_detections);
        batch_selected.insert(batch_selected.end(), class_selected.begin(),
                              class_selected.end());
      }
      std::stable_sort(batch_selected.begin(), batch_selected.end(),
                       CandidateScoreDescending);
      if (batch_selected.size() > max_detections) {
        batch_selected.resize(max_detections);
      }
    }

    output_counts[batch_index] =
        static_cast<int64_t>(batch_selected.size());
    const std::size_t output_base = batch_index * max_detections;
    for (std::size_t slot = 0; slot < batch_selected.size(); ++slot) {
      const BatchCandidate& candidate = batch_selected[slot];
      const std::size_t output_index = output_base + slot;
      const std::size_t source_box =
          static_cast<std::size_t>(candidate.box_index) * 4U;
      std::copy_n(batch_boxes + source_box, 4, output_boxes + output_index * 4U);
      output_scores[output_index] = candidate.score;
      output_class_ids[output_index] = candidate.class_id;
      output_indices[output_index] = candidate.box_index;
    }
  }
  return output;
}

BatchNmsResult RunBatchNmsOnHost(const Tensor& boxes, const Tensor& scores,
                                 BatchNmsShape shape,
                                 const BatchNmsOptions& options,
                                 TensorAllocator& allocator) {
  postprocess_internal::BatchNmsHostResult host =
      ComputeBatchNmsOnHostBuffers(
          static_cast<const float*>(boxes.data()),
          static_cast<const float*>(scores.data()), shape, options);
  BatchNmsResult output = AllocateBatchResult(
      shape, options.max_detections, boxes.device(), allocator);
  const auto copy = [](Tensor& destination, const auto& source) {
    if (destination.bytes() > 0) {
      std::memcpy(destination.data(), source.data(), destination.bytes());
    }
  };
  copy(output.counts, host.counts);
  copy(output.boxes, host.boxes);
  copy(output.scores, host.scores);
  copy(output.class_ids, host.class_ids);
  copy(output.indices, host.indices);
  return output;
}

std::vector<int64_t> MakeScoreOrder(const float* scores, std::size_t count) {
  std::vector<int64_t> order(count);
  std::iota(order.begin(), order.end(), int64_t{0});
  std::stable_sort(
      order.begin(), order.end(),
      [scores](int64_t lhs, int64_t rhs) { return scores[lhs] > scores[rhs]; });
  return order;
}

Tensor MakeIndexTensor(std::vector<int64_t> indices, Device device,
                       TensorAllocator& allocator) {
  Tensor output = Tensor::Allocate(
      MakeOutputDesc("nms_indices", DataType::kInt64,
                     {static_cast<int64_t>(indices.size())}, device),
      allocator);
  if (output.bytes() > 0) {
    std::memcpy(output.data(), indices.data(), output.bytes());
  }
  return output;
}

Tensor RunNmsOnHost(const Tensor& boxes, const Tensor& scores,
                    const Tensor* class_ids, const Tensor* batch_ids,
                    NmsParameters parameters, TensorAllocator& allocator) {
  const std::size_t count = BoxCount(boxes);
  if (count == 0) {
    return MakeIndexTensor({}, boxes.device(), allocator);
  }

  const auto* boxes_data = static_cast<const float*>(boxes.data());
  const auto* scores_data = static_cast<const float*>(scores.data());
  const auto* class_data = class_ids == nullptr
                               ? nullptr
                               : static_cast<const int64_t*>(class_ids->data());
  const auto* batch_data = batch_ids == nullptr
                               ? nullptr
                               : static_cast<const int64_t*>(batch_ids->data());
  std::vector<int64_t> order = MakeScoreOrder(scores_data, count);

  std::vector<float> areas(count);
  for (std::size_t index = 0; index < count; ++index) {
    areas[index] = BoxArea(boxes_data, static_cast<int64_t>(index),
                           parameters.coordinate_offset);
  }

  std::vector<uint8_t> suppressed(count, 0);
  std::vector<int64_t> keep;
  keep.reserve(count);
  std::unordered_map<int64_t, int64_t> batch_output_counts;
  for (std::size_t sorted_index = 0; sorted_index < count; ++sorted_index) {
    if (suppressed[sorted_index] != 0) {
      continue;
    }

    const int64_t selected = order[sorted_index];
    if (batch_data != nullptr && parameters.max_output_boxes > 0 &&
        batch_output_counts[batch_data[selected]] >=
            parameters.max_output_boxes) {
      continue;
    }
    keep.push_back(selected);
    if (batch_data != nullptr) {
      ++batch_output_counts[batch_data[selected]];
    } else if (parameters.max_output_boxes > 0 &&
               keep.size() ==
                   static_cast<std::size_t>(parameters.max_output_boxes)) {
      break;
    }

    for (std::size_t next_index = sorted_index + 1; next_index < count;
         ++next_index) {
      if (suppressed[next_index] != 0) {
        continue;
      }

      const int64_t candidate = order[next_index];
      if ((batch_data != nullptr &&
           batch_data[selected] != batch_data[candidate]) ||
          (class_data != nullptr &&
           class_data[selected] != class_data[candidate])) {
        continue;
      }
      if (IoU(boxes_data, areas, selected, candidate,
              parameters.coordinate_offset) > parameters.iou_threshold) {
        suppressed[next_index] = 1;
      }
    }
  }
  return MakeIndexTensor(std::move(keep), boxes.device(), allocator);
}

Tensor RunNms(const Tensor& boxes, const Tensor& scores,
              const Tensor* class_ids, const Tensor* batch_ids,
              NmsParameters parameters, TensorAllocator& allocator,
              ExecutionStream* execution_stream) {
  ValidateParameters(parameters);
  ValidateInputs(boxes, scores);
  if (class_ids != nullptr && batch_ids == nullptr) {
    throw std::logic_error("Class-aware NMS requires batch IDs");
  }
  if (batch_ids != nullptr) {
    ValidateGroupIds(boxes, *batch_ids);
  }
  if (class_ids != nullptr) {
    ValidateGroupIds(boxes, *class_ids);
  }
  ValidateExecutionStream(boxes, execution_stream);

  if (boxes.device().type == DeviceType::kCpu) {
    return RunNmsOnHost(boxes, scores, class_ids, batch_ids, parameters,
                        allocator);
  }
  if (boxes.device().type == DeviceType::kCuda) {
#if defined(MW_INFER_HAS_CUDA_POSTPROCESS)
    return postprocess_internal::RunNmsOnDevice(
        boxes, scores, class_ids, batch_ids, parameters, allocator,
        execution_stream);
#else
    throw std::runtime_error("CUDA NMS is unavailable in this build");
#endif
  }
  throw std::invalid_argument("NMS tensor device is unsupported");
}

}  // namespace

Tensor Nms(const Tensor& boxes, const Tensor& scores, float iou_threshold,
           float coordinate_offset, int64_t max_output_boxes,
           TensorAllocator& allocator, ExecutionStream* execution_stream) {
  NmsParameters parameters;
  parameters.iou_threshold = iou_threshold;
  parameters.coordinate_offset = coordinate_offset;
  parameters.max_output_boxes = max_output_boxes;
  return RunNms(boxes, scores, nullptr, nullptr, parameters, allocator,
                execution_stream);
}

BatchNmsResult BatchNms(const Tensor& boxes, const Tensor& scores,
                        const BatchNmsOptions& options,
                        TensorAllocator& allocator,
                        ExecutionStream* execution_stream) {
  const BatchNmsShape shape = ValidateBatchInputs(boxes, scores);
  ValidateBatchOptions(options);
  ValidateExecutionStream(boxes, execution_stream);
  if (boxes.device().type == DeviceType::kCpu) {
    return RunBatchNmsOnHost(boxes, scores, shape, options, allocator);
  }
  if (boxes.device().type == DeviceType::kCuda) {
#if defined(MW_INFER_HAS_CUDA_POSTPROCESS)
    return postprocess_internal::RunBatchNmsOnDevice(
        boxes, scores, options, allocator, execution_stream);
#else
    throw std::runtime_error("CUDA BatchNms is unavailable in this build");
#endif
  }
  throw std::invalid_argument("BatchNms tensor device is unsupported");
}

namespace postprocess_internal {

BatchNmsHostResult RunBatchNmsOnHostBuffers(
    const float* boxes, const float* scores, int64_t batch_count,
    int64_t box_count, int64_t class_count,
    const BatchNmsOptions& options) {
  return ComputeBatchNmsOnHostBuffers(
      boxes, scores, BatchNmsShape{batch_count, box_count, class_count},
      options);
}

Tensor RunClassAwareBatchNms(
    const Tensor& boxes, const Tensor& scores, const Tensor& class_ids,
    const Tensor& batch_ids, float iou_threshold, float coordinate_offset,
    int64_t max_output_boxes, TensorAllocator& allocator,
    ExecutionStream* execution_stream) {
  NmsParameters parameters;
  parameters.iou_threshold = iou_threshold;
  parameters.coordinate_offset = coordinate_offset;
  parameters.max_output_boxes = max_output_boxes;
  return RunNms(boxes, scores, &class_ids, &batch_ids, parameters, allocator,
                execution_stream);
}

}  // namespace postprocess_internal

}  // namespace mw::infer
