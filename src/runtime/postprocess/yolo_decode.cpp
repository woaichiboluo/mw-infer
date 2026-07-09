#include "mw/infer/runtime/postprocess/yolo_decode.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mw::infer {

#if defined(MW_INFER_HAS_CUDA_POSTPROCESS)
namespace postprocess_internal {
YoloDecodeResult RunYoloDecodeOnDevice(
    const Tensor& predictions, int64_t batch_count, int64_t channel_count,
    int64_t candidate_count, bool channel_first, YoloDecodeOptions options,
    TensorAllocator& allocator);
}  // namespace postprocess_internal
#endif

namespace {

struct YoloShape {
  int64_t batch_count = 0;
  int64_t channel_count = 0;
  int64_t candidate_count = 0;
  bool channel_first = true;
};

bool HasObjectness(YoloVersion version) {
  switch (version) {
    case YoloVersion::kYoloV5:
      return true;
    case YoloVersion::kYoloV8:
    case YoloVersion::kYoloV11:
      return false;
  }
  throw std::invalid_argument("YOLO version is unsupported");
}

int64_t ClassStart(YoloVersion version) {
  return HasObjectness(version) ? 5 : 4;
}

const char* YoloName(YoloVersion version) {
  switch (version) {
    case YoloVersion::kYoloV5:
      return "YOLOv5";
    case YoloVersion::kYoloV8:
      return "YOLOv8";
    case YoloVersion::kYoloV11:
      return "YOLOv11";
  }
  return "YOLO";
}

void ValidateOptions(YoloDecodeOptions options) {
  if (!std::isfinite(options.score_threshold)) {
    throw std::invalid_argument("YOLO score threshold must be finite");
  }
  if (!std::isfinite(options.class_offset) || options.class_offset < 0.0F) {
    throw std::invalid_argument(
        "YOLO class offset must be finite and non-negative");
  }
  static_cast<void>(ClassStart(options.version));
}

bool ResolveChannelFirst(const std::vector<int64_t>& shape,
                         YoloDecodeOptions options) {
  const int64_t min_channel_count = ClassStart(options.version) + 1;
  const bool dim1_can_be_channels = shape[1] >= min_channel_count;
  const bool dim2_can_be_channels = shape[2] >= min_channel_count;
  if (dim1_can_be_channels && !dim2_can_be_channels) {
    return true;
  }
  if (!dim1_can_be_channels && dim2_can_be_channels) {
    return false;
  }
  return shape[1] <= shape[2];
}

YoloShape ValidatePredictions(const Tensor& predictions,
                              YoloDecodeOptions options) {
  if (predictions.empty()) {
    throw std::invalid_argument("YOLO predictions tensor is empty");
  }
  if (predictions.data_type() != DataType::kFloat32) {
    throw std::invalid_argument("YOLO predictions tensor must be float32");
  }
  if (predictions.shape().size() != 3) {
    throw std::invalid_argument(
        "YOLO predictions tensor shape must be [B, C, N] or [B, N, C]");
  }

  YoloShape shape;
  shape.batch_count = predictions.shape()[0];
  if (shape.batch_count < 0) {
    throw std::invalid_argument("YOLO predictions batch must be non-negative");
  }
  shape.channel_first = ResolveChannelFirst(predictions.shape(), options);
  if (shape.channel_first) {
    shape.channel_count = predictions.shape()[1];
    shape.candidate_count = predictions.shape()[2];
  } else {
    shape.candidate_count = predictions.shape()[1];
    shape.channel_count = predictions.shape()[2];
  }

  const int64_t class_start = ClassStart(options.version);
  if (shape.channel_count <= class_start) {
    throw std::invalid_argument(std::string(YoloName(options.version)) +
                                " predictions channel count is too small");
  }
  if (shape.candidate_count < 0) {
    throw std::invalid_argument(
        "YOLO predictions candidate count must be non-negative");
  }
  if (shape.candidate_count != 0 &&
      shape.batch_count >
          std::numeric_limits<int64_t>::max() / shape.candidate_count) {
    throw std::invalid_argument("YOLO predictions count overflows int64");
  }
  return shape;
}

TensorDesc MakeFloatDesc(std::string name, std::vector<int64_t> shape,
                         Device device) {
  TensorDesc desc;
  desc.info.name = std::move(name);
  desc.info.data_type = DataType::kFloat32;
  desc.info.shape = std::move(shape);
  desc.device = device;
  return desc;
}

TensorDesc MakeClassIdDesc(std::vector<int64_t> shape, Device device) {
  TensorDesc desc;
  desc.info.name = "yolo_class_ids";
  desc.info.data_type = DataType::kInt64;
  desc.info.shape = std::move(shape);
  desc.device = device;
  return desc;
}

TensorDesc MakeBatchIdDesc(std::vector<int64_t> shape, Device device) {
  TensorDesc desc;
  desc.info.name = "yolo_batch_ids";
  desc.info.data_type = DataType::kInt64;
  desc.info.shape = std::move(shape);
  desc.device = device;
  return desc;
}

float ValueAt(const float* predictions, YoloShape shape, int64_t batch,
              int64_t channel, int64_t candidate) {
  if (shape.channel_first) {
    return predictions[static_cast<std::size_t>(
        (batch * shape.channel_count + channel) * shape.candidate_count +
        candidate)];
  }
  return predictions[static_cast<std::size_t>(
      (batch * shape.candidate_count + candidate) * shape.channel_count +
      channel)];
}

YoloDecodeResult MakeEmptyResult(Device device, TensorAllocator& allocator) {
  YoloDecodeResult result;
  result.boxes =
      Tensor::Allocate(MakeFloatDesc("yolo_boxes", {0, 4}, device), allocator);
  result.nms_boxes = Tensor::Allocate(
      MakeFloatDesc("yolo_nms_boxes", {0, 4}, device), allocator);
  result.scores =
      Tensor::Allocate(MakeFloatDesc("yolo_scores", {0}, device), allocator);
  result.class_ids = Tensor::Allocate(MakeClassIdDesc({0}, device), allocator);
  result.batch_ids = Tensor::Allocate(MakeBatchIdDesc({0}, device), allocator);
  return result;
}

YoloDecodeResult RunYoloDecodeOnHost(const Tensor& predictions, YoloShape shape,
                                     YoloDecodeOptions options,
                                     TensorAllocator& allocator) {
  const auto* input = static_cast<const float*>(predictions.data());
  const int64_t class_start = ClassStart(options.version);
  const int64_t class_count = shape.channel_count - class_start;
  const bool has_objectness = HasObjectness(options.version);

  std::vector<float> boxes;
  std::vector<float> nms_boxes;
  std::vector<float> scores;
  std::vector<int64_t> class_ids;
  std::vector<int64_t> batch_ids;
  const auto max_candidates =
      static_cast<std::size_t>(shape.batch_count * shape.candidate_count);
  boxes.reserve(max_candidates * 4U);
  nms_boxes.reserve(max_candidates * 4U);
  scores.reserve(max_candidates);
  class_ids.reserve(max_candidates);
  batch_ids.reserve(max_candidates);

  for (int64_t batch = 0; batch < shape.batch_count; ++batch) {
    for (int64_t candidate = 0; candidate < shape.candidate_count;
         ++candidate) {
      float best_class_score = -std::numeric_limits<float>::infinity();
      int64_t best_class = 0;
      for (int64_t class_index = 0; class_index < class_count; ++class_index) {
        const float class_score =
            ValueAt(input, shape, batch, class_start + class_index, candidate);
        if (class_score > best_class_score) {
          best_class_score = class_score;
          best_class = class_index;
        }
      }

      const float objectness =
          has_objectness ? ValueAt(input, shape, batch, 4, candidate) : 1.0F;
      const float score = objectness * best_class_score;
      if (score < options.score_threshold) {
        continue;
      }

      const float center_x = ValueAt(input, shape, batch, 0, candidate);
      const float center_y = ValueAt(input, shape, batch, 1, candidate);
      const float width = ValueAt(input, shape, batch, 2, candidate);
      const float height = ValueAt(input, shape, batch, 3, candidate);
      if (width <= 0.0F || height <= 0.0F) {
        continue;
      }

      const float left = center_x - width * 0.5F;
      const float top = center_y - height * 0.5F;
      const float right = center_x + width * 0.5F;
      const float bottom = center_y + height * 0.5F;
      const float nms_group =
          static_cast<float>(batch * class_count + best_class);
      const float nms_shift = nms_group * options.class_offset;

      boxes.insert(boxes.end(), {left, top, right, bottom});
      nms_boxes.insert(nms_boxes.end(),
                       {left + nms_shift, top + nms_shift, right + nms_shift,
                        bottom + nms_shift});
      scores.push_back(score);
      class_ids.push_back(best_class);
      batch_ids.push_back(batch);
    }
  }

  const int64_t count = static_cast<int64_t>(scores.size());
  YoloDecodeResult result;
  result.boxes = Tensor::Allocate(
      MakeFloatDesc("yolo_boxes", {count, 4}, predictions.device()), allocator);
  result.nms_boxes = Tensor::Allocate(
      MakeFloatDesc("yolo_nms_boxes", {count, 4}, predictions.device()),
      allocator);
  result.scores = Tensor::Allocate(
      MakeFloatDesc("yolo_scores", {count}, predictions.device()), allocator);
  result.class_ids = Tensor::Allocate(
      MakeClassIdDesc({count}, predictions.device()), allocator);
  result.batch_ids = Tensor::Allocate(
      MakeBatchIdDesc({count}, predictions.device()), allocator);
  if (count > 0) {
    std::memcpy(result.boxes.data(), boxes.data(), result.boxes.bytes());
    std::memcpy(result.nms_boxes.data(), nms_boxes.data(),
                result.nms_boxes.bytes());
    std::memcpy(result.scores.data(), scores.data(), result.scores.bytes());
    std::memcpy(result.class_ids.data(), class_ids.data(),
                result.class_ids.bytes());
    std::memcpy(result.batch_ids.data(), batch_ids.data(),
                result.batch_ids.bytes());
  }
  return result;
}

}  // namespace

YoloDecodeResult YoloDecode(const Tensor& predictions,
                            YoloDecodeOptions options,
                            TensorAllocator& allocator) {
  ValidateOptions(options);
  const YoloShape shape = ValidatePredictions(predictions, options);
  if (shape.batch_count == 0 || shape.candidate_count == 0) {
    return MakeEmptyResult(predictions.device(), allocator);
  }

  if (predictions.device().type == DeviceType::kCpu) {
    return RunYoloDecodeOnHost(predictions, shape, options, allocator);
  }
  if (predictions.device().type == DeviceType::kCuda) {
#if defined(MW_INFER_HAS_CUDA_POSTPROCESS)
    return postprocess_internal::RunYoloDecodeOnDevice(
        predictions, shape.batch_count, shape.channel_count,
        shape.candidate_count, shape.channel_first, options, allocator);
#else
    throw std::runtime_error("CUDA YOLO decode is unavailable in this build");
#endif
  }
  throw std::invalid_argument("YOLO predictions tensor device is unsupported");
}

}  // namespace mw::infer
