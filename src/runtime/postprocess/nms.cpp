#include "mw/infer/runtime/postprocess/nms.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "nms_internal.h"

namespace mw::infer {

namespace {

using postprocess_internal::NmsParameters;

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
  TensorDesc desc;
  desc.info.name = "nms_indices";
  desc.info.data_type = DataType::kInt64;
  desc.info.shape = {static_cast<int64_t>(indices.size())};
  desc.device = device;
  Tensor output = Tensor::Allocate(std::move(desc), allocator);
  if (output.bytes() > 0) {
    std::memcpy(output.data(), indices.data(), output.bytes());
  }
  return output;
}

Tensor RunNmsOnHost(const Tensor& boxes, const Tensor& scores,
                    NmsParameters parameters, TensorAllocator& allocator) {
  const std::size_t count = BoxCount(boxes);
  if (count == 0) {
    return MakeIndexTensor({}, boxes.device(), allocator);
  }

  const auto* boxes_data = static_cast<const float*>(boxes.data());
  const auto* scores_data = static_cast<const float*>(scores.data());
  std::vector<int64_t> order = MakeScoreOrder(scores_data, count);

  std::vector<float> areas(count);
  for (std::size_t index = 0; index < count; ++index) {
    areas[index] = BoxArea(boxes_data, static_cast<int64_t>(index),
                           parameters.coordinate_offset);
  }

  std::vector<uint8_t> suppressed(count, 0);
  std::vector<int64_t> keep;
  keep.reserve(count);
  for (std::size_t sorted_index = 0; sorted_index < count; ++sorted_index) {
    if (suppressed[sorted_index] != 0) {
      continue;
    }

    const int64_t selected = order[sorted_index];
    keep.push_back(selected);
    if (parameters.max_output_boxes > 0 &&
        keep.size() == static_cast<std::size_t>(parameters.max_output_boxes)) {
      break;
    }

    for (std::size_t next_index = sorted_index + 1; next_index < count;
         ++next_index) {
      if (suppressed[next_index] != 0) {
        continue;
      }

      const int64_t candidate = order[next_index];
      if (IoU(boxes_data, areas, selected, candidate,
              parameters.coordinate_offset) > parameters.iou_threshold) {
        suppressed[next_index] = 1;
      }
    }
  }
  return MakeIndexTensor(std::move(keep), boxes.device(), allocator);
}

}  // namespace

Tensor Nms(const Tensor& boxes, const Tensor& scores, float iou_threshold,
           float coordinate_offset, int64_t max_output_boxes,
           TensorAllocator& allocator) {
  NmsParameters parameters;
  parameters.iou_threshold = iou_threshold;
  parameters.coordinate_offset = coordinate_offset;
  parameters.max_output_boxes = max_output_boxes;
  ValidateParameters(parameters);
  ValidateInputs(boxes, scores);

  if (boxes.device().type == DeviceType::kCpu) {
    return RunNmsOnHost(boxes, scores, parameters, allocator);
  }
  if (boxes.device().type == DeviceType::kCuda) {
#if defined(MW_INFER_HAS_CUDA_POSTPROCESS)
    return postprocess_internal::RunNmsOnDevice(boxes, scores, parameters,
                                                allocator);
#else
    throw std::runtime_error("CUDA NMS is unavailable in this build");
#endif
  }
  throw std::invalid_argument("NMS tensor device is unsupported");
}

}  // namespace mw::infer
