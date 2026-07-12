#include "mw/infer/runtime/postprocess/yolo_seg_decode.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mw/infer/runtime/execution_stream.h"
#include "nms_internal.h"
#include "yolo_seg_decode_internal.h"

namespace mw::infer {
namespace {

using postprocess_internal::YoloSegCandidates;

// Logical shape after resolving the exported candidate tensor layout.
struct PredictionShape {
  int64_t batch_count = 0;
  int64_t channel_count = 0;
  int64_t candidate_count = 0;
  int64_t class_count = 0;
  int64_t mask_count = 0;
  int64_t coefficient_start = 0;
  bool channel_first = true;
};

bool HasObjectness(YoloVersion version) {
  switch (version) {
    case YoloVersion::kYoloV5:
      return true;
    case YoloVersion::kYoloV8:
    case YoloVersion::kYoloV11:
    case YoloVersion::kYoloV26:
      return false;
  }
  throw std::invalid_argument("YOLO segmentation version is unsupported");
}

int64_t ClassStart(YoloVersion version) {
  return HasObjectness(version) ? 5 : 4;
}

void ValidateOptions(const YoloSegDecodeOptions& options) {
  if (!std::isfinite(options.score_threshold) ||
      options.score_threshold < 0.0F || options.score_threshold > 1.0F) {
    throw std::invalid_argument(
        "YOLO segmentation score threshold must be in [0, 1]");
  }
  if (!std::isfinite(options.iou_threshold) || options.iou_threshold < 0.0F ||
      options.iou_threshold > 1.0F) {
    throw std::invalid_argument(
        "YOLO segmentation IoU threshold must be in [0, 1]");
  }
  if (!std::isfinite(options.mask_threshold) || options.mask_threshold < 0.0F ||
      options.mask_threshold > 1.0F) {
    throw std::invalid_argument(
        "YOLO segmentation mask threshold must be in [0, 1]");
  }
  if (options.max_detections < 0) {
    throw std::invalid_argument(
        "YOLO segmentation max detections must be non-negative");
  }
  static_cast<void>(ClassStart(options.version));
  switch (options.prediction_layout) {
    case YoloSegPredictionLayout::kRaw:
    case YoloSegPredictionLayout::kSelected:
      break;
    default:
      throw std::invalid_argument(
          "YOLO segmentation prediction layout is unsupported");
  }
  switch (options.tensor_layout) {
    case YoloSegTensorLayout::kAuto:
    case YoloSegTensorLayout::kChannelFirst:
    case YoloSegTensorLayout::kCandidateFirst:
      break;
    default:
      throw std::invalid_argument(
          "YOLO segmentation tensor layout is unsupported");
  }
}

void ValidateTensorInputs(const Tensor& predictions, const Tensor& prototypes,
                          ImageSize input_size) {
  if (predictions.empty()) {
    throw std::invalid_argument(
        "YOLO segmentation predictions tensor is empty");
  }
  if (prototypes.empty()) {
    throw std::invalid_argument("YOLO segmentation prototypes tensor is empty");
  }
  if (predictions.data_type() != DataType::kFloat32 ||
      prototypes.data_type() != DataType::kFloat32) {
    throw std::invalid_argument(
        "YOLO segmentation predictions and prototypes must be float32");
  }
  if (predictions.shape().size() != 3) {
    throw std::invalid_argument(
        "YOLO segmentation predictions shape must be rank 3");
  }
  if (prototypes.shape().size() != 4) {
    throw std::invalid_argument(
        "YOLO segmentation prototypes shape must be [B, M, H, W]");
  }
  if (predictions.device().type != prototypes.device().type ||
      predictions.device().id != prototypes.device().id) {
    throw std::invalid_argument(
        "YOLO segmentation predictions and prototypes device mismatch");
  }
  if (input_size.width <= 0 || input_size.height <= 0) {
    throw std::invalid_argument(
        "YOLO segmentation input size must be positive");
  }
  if (predictions.shape()[0] < 0 || prototypes.shape()[0] < 0 ||
      predictions.shape()[0] != prototypes.shape()[0]) {
    throw std::invalid_argument(
        "YOLO segmentation predictions and prototypes batch mismatch");
  }
  if (prototypes.shape()[1] <= 0 || prototypes.shape()[2] <= 0 ||
      prototypes.shape()[3] <= 0) {
    throw std::invalid_argument(
        "YOLO segmentation prototype dimensions must be positive");
  }
}

void ValidateExecutionStream(const Tensor& predictions,
                             const ExecutionStream* execution_stream) {
  if (execution_stream == nullptr) {
    return;
  }
  const Device stream_device = execution_stream->device();
  if (stream_device.type != predictions.device().type ||
      stream_device.id != predictions.device().id) {
    throw std::invalid_argument(
        "YOLO segmentation execution stream device does not match tensor "
        "device");
  }
}

bool ResolveRawChannelFirst(const std::vector<int64_t>& shape,
                            int64_t minimum_channels,
                            const YoloSegDecodeOptions& options) {
  const bool dim1_can_be_channels = shape[1] >= minimum_channels;
  const bool dim2_can_be_channels = shape[2] >= minimum_channels;
  if (options.tensor_layout == YoloSegTensorLayout::kChannelFirst) {
    if (!dim1_can_be_channels) {
      throw std::invalid_argument(
          "YOLO segmentation channel-first raw tensor has too few channels");
    }
    return true;
  }
  if (options.tensor_layout == YoloSegTensorLayout::kCandidateFirst) {
    if (!dim2_can_be_channels) {
      throw std::invalid_argument(
          "YOLO segmentation candidate-first raw tensor has too few "
          "channels");
    }
    return false;
  }
  if (dim1_can_be_channels && !dim2_can_be_channels) {
    return true;
  }
  if (!dim1_can_be_channels && dim2_can_be_channels) {
    return false;
  }
  if (!dim1_can_be_channels && !dim2_can_be_channels) {
    throw std::invalid_argument(
        "YOLO segmentation raw prediction channel count is too small");
  }
  return options.version != YoloVersion::kYoloV5;
}

bool ResolveSelectedChannelFirst(const std::vector<int64_t>& shape,
                                 int64_t selected_channels,
                                 YoloSegTensorLayout tensor_layout) {
  const bool dim1_matches = shape[1] == selected_channels;
  const bool dim2_matches = shape[2] == selected_channels;
  if (tensor_layout == YoloSegTensorLayout::kChannelFirst) {
    if (!dim1_matches) {
      throw std::invalid_argument(
          "YOLO selected channel-first tensor channel count mismatch");
    }
    return true;
  }
  if (tensor_layout == YoloSegTensorLayout::kCandidateFirst) {
    if (!dim2_matches) {
      throw std::invalid_argument(
          "YOLO selected candidate-first tensor channel count mismatch");
    }
    return false;
  }
  if (!dim1_matches && !dim2_matches) {
    throw std::invalid_argument(
        "YOLO selected segmentation predictions have no 6 + mask_count "
        "channel axis");
  }
  if (dim1_matches != dim2_matches) {
    return dim1_matches;
  }
  return false;
}

PredictionShape ResolvePredictionShape(const Tensor& predictions,
                                       const Tensor& prototypes,
                                       const YoloSegDecodeOptions& options) {
  PredictionShape shape;
  shape.batch_count = predictions.shape()[0];
  shape.mask_count = prototypes.shape()[1];

  if (options.prediction_layout == YoloSegPredictionLayout::kSelected) {
    const int64_t selected_channels = 6 + shape.mask_count;
    shape.channel_first = ResolveSelectedChannelFirst(
        predictions.shape(), selected_channels, options.tensor_layout);
    shape.channel_count = selected_channels;
    shape.candidate_count =
        shape.channel_first ? predictions.shape()[2] : predictions.shape()[1];
    shape.coefficient_start = 6;
  } else {
    const int64_t class_start = ClassStart(options.version);
    const int64_t minimum_channels = class_start + 1 + shape.mask_count;
    shape.channel_first =
        ResolveRawChannelFirst(predictions.shape(), minimum_channels, options);
    shape.channel_count =
        shape.channel_first ? predictions.shape()[1] : predictions.shape()[2];
    shape.candidate_count =
        shape.channel_first ? predictions.shape()[2] : predictions.shape()[1];
    shape.class_count = shape.channel_count - class_start - shape.mask_count;
    shape.coefficient_start = class_start + shape.class_count;
    if (shape.class_count <= 0) {
      throw std::invalid_argument(
          "YOLO segmentation raw predictions contain no class scores");
    }
  }

  if (shape.candidate_count < 0) {
    throw std::invalid_argument(
        "YOLO segmentation candidate count must be non-negative");
  }
  if (shape.candidate_count != 0 &&
      shape.batch_count >
          std::numeric_limits<int64_t>::max() / shape.candidate_count) {
    throw std::invalid_argument(
        "YOLO segmentation candidate count overflows int64");
  }
  return shape;
}

float PredictionValue(const float* predictions, const PredictionShape& shape,
                      int64_t batch, int64_t channel, int64_t candidate) {
  if (shape.channel_first) {
    return predictions[static_cast<std::size_t>(
        (batch * shape.channel_count + channel) * shape.candidate_count +
        candidate)];
  }
  return predictions[static_cast<std::size_t>(
      (batch * shape.candidate_count + candidate) * shape.channel_count +
      channel)];
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

TensorDesc MakeInt64Desc(std::string name, std::vector<int64_t> shape,
                         Device device) {
  TensorDesc desc;
  desc.info.name = std::move(name);
  desc.info.data_type = DataType::kInt64;
  desc.info.shape = std::move(shape);
  desc.device = device;
  return desc;
}

TensorDesc MakeMaskDesc(std::vector<int64_t> shape, Device device) {
  TensorDesc desc;
  desc.info.name = "yolo_seg_masks";
  desc.info.data_type = DataType::kUInt8;
  desc.info.shape = std::move(shape);
  desc.device = device;
  return desc;
}

YoloSegCandidates AllocateCandidates(int64_t count, Device device,
                                     TensorAllocator& allocator) {
  YoloSegCandidates result;
  result.boxes = Tensor::Allocate(
      MakeFloatDesc("yolo_seg_boxes", {count, 4}, device), allocator);
  result.scores = Tensor::Allocate(
      MakeFloatDesc("yolo_seg_scores", {count}, device), allocator);
  result.class_ids = Tensor::Allocate(
      MakeInt64Desc("yolo_seg_class_ids", {count}, device), allocator);
  result.batch_ids = Tensor::Allocate(
      MakeInt64Desc("yolo_seg_batch_ids", {count}, device), allocator);
  result.candidate_ids = Tensor::Allocate(
      MakeInt64Desc("yolo_seg_candidate_ids", {count}, device), allocator);
  return result;
}

YoloSegCandidates RunRawDecodeOnHost(const Tensor& predictions,
                                     const PredictionShape& shape,
                                     const YoloSegDecodeOptions& options,
                                     TensorAllocator& allocator) {
  const auto* input = static_cast<const float*>(predictions.data());
  const int64_t class_start = ClassStart(options.version);
  const bool has_objectness = HasObjectness(options.version);
  std::vector<float> boxes;
  std::vector<float> scores;
  std::vector<int64_t> class_ids;
  std::vector<int64_t> batch_ids;
  std::vector<int64_t> candidate_ids;

  for (int64_t batch = 0; batch < shape.batch_count; ++batch) {
    for (int64_t candidate = 0; candidate < shape.candidate_count;
         ++candidate) {
      float best_class_score = -std::numeric_limits<float>::infinity();
      int64_t best_class = 0;
      for (int64_t class_index = 0; class_index < shape.class_count;
           ++class_index) {
        const float class_score = PredictionValue(
            input, shape, batch, class_start + class_index, candidate);
        if (class_score > best_class_score) {
          best_class_score = class_score;
          best_class = class_index;
        }
      }
      const float objectness =
          has_objectness ? PredictionValue(input, shape, batch, 4, candidate)
                         : 1.0F;
      const float score = objectness * best_class_score;
      if (!std::isfinite(score) || score < options.score_threshold) {
        continue;
      }

      const float center_x = PredictionValue(input, shape, batch, 0, candidate);
      const float center_y = PredictionValue(input, shape, batch, 1, candidate);
      const float width = PredictionValue(input, shape, batch, 2, candidate);
      const float height = PredictionValue(input, shape, batch, 3, candidate);
      if (!std::isfinite(center_x) || !std::isfinite(center_y) ||
          !std::isfinite(width) || !std::isfinite(height) || width <= 0.0F ||
          height <= 0.0F) {
        continue;
      }
      const float left = center_x - width * 0.5F;
      const float top = center_y - height * 0.5F;
      const float right = center_x + width * 0.5F;
      const float bottom = center_y + height * 0.5F;
      if (!std::isfinite(left) || !std::isfinite(top) ||
          !std::isfinite(right) || !std::isfinite(bottom)) {
        continue;
      }

      boxes.insert(boxes.end(), {left, top, right, bottom});
      scores.push_back(score);
      class_ids.push_back(best_class);
      batch_ids.push_back(batch);
      candidate_ids.push_back(candidate);
    }
  }

  const int64_t count = static_cast<int64_t>(scores.size());
  YoloSegCandidates result =
      AllocateCandidates(count, predictions.device(), allocator);
  if (count > 0) {
    std::memcpy(result.boxes.data(), boxes.data(), result.boxes.bytes());
    std::memcpy(result.scores.data(), scores.data(), result.scores.bytes());
    std::memcpy(result.class_ids.data(), class_ids.data(),
                result.class_ids.bytes());
    std::memcpy(result.batch_ids.data(), batch_ids.data(),
                result.batch_ids.bytes());
    std::memcpy(result.candidate_ids.data(), candidate_ids.data(),
                result.candidate_ids.bytes());
  }
  return result;
}

YoloSegCandidates RunSelectedDecodeOnHost(const Tensor& predictions,
                                          const PredictionShape& shape,
                                          const YoloSegDecodeOptions& options,
                                          TensorAllocator& allocator) {
  const auto* input = static_cast<const float*>(predictions.data());
  std::vector<float> boxes;
  std::vector<float> scores;
  std::vector<int64_t> class_ids;
  std::vector<int64_t> batch_ids;
  std::vector<int64_t> candidate_ids;

  for (int64_t batch = 0; batch < shape.batch_count; ++batch) {
    for (int64_t candidate = 0; candidate < shape.candidate_count;
         ++candidate) {
      const float score = PredictionValue(input, shape, batch, 4, candidate);
      if (!std::isfinite(score) || score < options.score_threshold) {
        continue;
      }
      const float left = PredictionValue(input, shape, batch, 0, candidate);
      const float top = PredictionValue(input, shape, batch, 1, candidate);
      const float right = PredictionValue(input, shape, batch, 2, candidate);
      const float bottom = PredictionValue(input, shape, batch, 3, candidate);
      const float class_value =
          PredictionValue(input, shape, batch, 5, candidate);
      if (!std::isfinite(left) || !std::isfinite(top) ||
          !std::isfinite(right) || !std::isfinite(bottom) || right <= left ||
          bottom <= top || !std::isfinite(class_value) || class_value < 0.0F ||
          std::floor(class_value) != class_value ||
          class_value > static_cast<float>(std::numeric_limits<int>::max())) {
        continue;
      }

      boxes.insert(boxes.end(), {left, top, right, bottom});
      scores.push_back(score);
      class_ids.push_back(static_cast<int64_t>(class_value));
      batch_ids.push_back(batch);
      candidate_ids.push_back(candidate);
    }
  }

  const int64_t count = static_cast<int64_t>(scores.size());
  YoloSegCandidates result =
      AllocateCandidates(count, predictions.device(), allocator);
  if (count > 0) {
    std::memcpy(result.boxes.data(), boxes.data(), result.boxes.bytes());
    std::memcpy(result.scores.data(), scores.data(), result.scores.bytes());
    std::memcpy(result.class_ids.data(), class_ids.data(),
                result.class_ids.bytes());
    std::memcpy(result.batch_ids.data(), batch_ids.data(),
                result.batch_ids.bytes());
    std::memcpy(result.candidate_ids.data(), candidate_ids.data(),
                result.candidate_ids.bytes());
  }
  return result;
}

Tensor MakeIndexTensor(const std::vector<int64_t>& indices, Device device,
                       TensorAllocator& allocator) {
  TensorDesc desc =
      MakeInt64Desc("yolo_seg_indices", {static_cast<int64_t>(indices.size())},
                    Device{DeviceType::kCpu, 0});
  if (device.type == DeviceType::kCpu) {
    desc.device = device;
    Tensor output = Tensor::Allocate(std::move(desc), allocator);
    if (!indices.empty()) {
      std::memcpy(output.data(), indices.data(), output.bytes());
    }
    return output;
  }

  Tensor host = Tensor::Allocate(std::move(desc));
  if (!indices.empty()) {
    std::memcpy(host.data(), indices.data(), host.bytes());
  }
  return host.CopyTo(device, allocator);
}

Tensor MakeSelectedIndices(const YoloSegCandidates& candidates,
                           int64_t batch_count, int64_t max_detections,
                           TensorAllocator& allocator) {
  const std::vector<int64_t> batch_ids =
      candidates.batch_ids.CopyToHostVector<int64_t>();
  std::vector<int64_t> counts(static_cast<std::size_t>(batch_count), 0);
  std::vector<int64_t> indices;
  indices.reserve(batch_ids.size());
  for (std::size_t index = 0; index < batch_ids.size(); ++index) {
    const int64_t batch = batch_ids[index];
    if (batch < 0 || batch >= batch_count) {
      throw std::runtime_error("YOLO selected batch id is out of range");
    }
    int64_t& count = counts[static_cast<std::size_t>(batch)];
    if (max_detections == 0 || count < max_detections) {
      indices.push_back(static_cast<int64_t>(index));
      ++count;
    }
  }
  return MakeIndexTensor(indices, candidates.batch_ids.device(), allocator);
}

float Sigmoid(float value) {
  if (value >= 0.0F) {
    return 1.0F / (1.0F + std::exp(-value));
  }
  const float exponential = std::exp(value);
  return exponential / (1.0F + exponential);
}

Tensor RunMasksOnHost(const Tensor& predictions, const Tensor& prototypes,
                      const Tensor& boxes, const Tensor& batch_ids,
                      const Tensor& candidate_ids,
                      const PredictionShape& prediction_shape,
                      bool scale_coefficients_by_objectness,
                      ImageSize input_size, float mask_threshold,
                      TensorAllocator& allocator) {
  const int64_t selected_count = boxes.shape()[0];
  Tensor output = Tensor::Allocate(
      MakeMaskDesc({selected_count, input_size.height, input_size.width},
                   predictions.device()),
      allocator);
  if (selected_count == 0) {
    return output;
  }

  const int64_t prototype_height = prototypes.shape()[2];
  const int64_t prototype_width = prototypes.shape()[3];
  const int64_t prototype_plane = prototype_height * prototype_width;
  const bool interpolate_probabilities = mask_threshold != 0.5F;
  const float value_threshold =
      interpolate_probabilities ? mask_threshold : 0.0F;
  Tensor logit_storage = Tensor::Allocate(
      MakeFloatDesc("yolo_seg_mask_values",
                    {selected_count, prototype_height, prototype_width},
                    predictions.device()),
      allocator);
  std::memset(logit_storage.data(), 0, logit_storage.bytes());
  auto* logits = static_cast<float*>(logit_storage.data());
  const auto* prediction_data = static_cast<const float*>(predictions.data());
  const auto* prototype_data = static_cast<const float*>(prototypes.data());
  const auto* box_data = static_cast<const float*>(boxes.data());
  const auto* batch_data = static_cast<const int64_t*>(batch_ids.data());
  const auto* candidate_data =
      static_cast<const int64_t*>(candidate_ids.data());
  const float x_scale = static_cast<float>(prototype_width) / input_size.width;
  const float y_scale =
      static_cast<float>(prototype_height) / input_size.height;

  for (int64_t selected = 0; selected < selected_count; ++selected) {
    const int64_t batch = batch_data[selected];
    const int64_t candidate = candidate_data[selected];
    if (batch < 0 || batch >= prediction_shape.batch_count || candidate < 0 ||
        candidate >= prediction_shape.candidate_count) {
      throw std::runtime_error(
          "YOLO segmentation mask source index is out of range");
    }
    const float objectness =
        scale_coefficients_by_objectness
            ? PredictionValue(prediction_data, prediction_shape, batch, 4,
                              candidate)
            : 1.0F;
    float* mask_logits =
        logits + static_cast<std::size_t>(selected * prototype_plane);
    for (int64_t mask_channel = 0; mask_channel < prediction_shape.mask_count;
         ++mask_channel) {
      const float coefficient =
          PredictionValue(prediction_data, prediction_shape, batch,
                          prediction_shape.coefficient_start + mask_channel,
                          candidate) *
          objectness;
      const float* prototype =
          prototype_data +
          static_cast<std::size_t>(
              (batch * prediction_shape.mask_count + mask_channel) *
              prototype_plane);
      for (int64_t pixel = 0; pixel < prototype_plane; ++pixel) {
        mask_logits[pixel] += coefficient * prototype[pixel];
      }
    }

    const float left = box_data[selected * 4] * x_scale;
    const float top = box_data[selected * 4 + 1] * y_scale;
    const float right = box_data[selected * 4 + 2] * x_scale;
    const float bottom = box_data[selected * 4 + 3] * y_scale;
    for (int64_t y = 0; y < prototype_height; ++y) {
      for (int64_t x = 0; x < prototype_width; ++x) {
        float& value = mask_logits[y * prototype_width + x];
        if (static_cast<float>(x) < left || static_cast<float>(x) >= right ||
            static_cast<float>(y) < top || static_cast<float>(y) >= bottom) {
          value = 0.0F;
        } else if (interpolate_probabilities) {
          value = Sigmoid(value);
        }
      }
    }
  }

  auto* masks = static_cast<std::uint8_t*>(output.data());
  const int64_t output_plane =
      static_cast<int64_t>(input_size.height) * input_size.width;
  for (int64_t selected = 0; selected < selected_count; ++selected) {
    const float* mask_logits =
        logits + static_cast<std::size_t>(selected * prototype_plane);
    const float box_left = box_data[selected * 4];
    const float box_top = box_data[selected * 4 + 1];
    const float box_right = box_data[selected * 4 + 2];
    const float box_bottom = box_data[selected * 4 + 3];
    for (int64_t y = 0; y < input_size.height; ++y) {
      const float source_y = (static_cast<float>(y) + 0.5F) * prototype_height /
                                 input_size.height -
                             0.5F;
      const float clamped_y =
          std::clamp(source_y, 0.0F, static_cast<float>(prototype_height - 1));
      const int64_t y0 = static_cast<int64_t>(std::floor(clamped_y));
      const int64_t y1 = std::min(y0 + 1, prototype_height - 1);
      const float y_weight = clamped_y - static_cast<float>(y0);
      for (int64_t x = 0; x < input_size.width; ++x) {
        const std::size_t output_index = static_cast<std::size_t>(
            selected * output_plane + y * input_size.width + x);
        if (static_cast<float>(x) < box_left ||
            static_cast<float>(x) >= box_right ||
            static_cast<float>(y) < box_top ||
            static_cast<float>(y) >= box_bottom) {
          masks[output_index] = 0;
          continue;
        }
        const float source_x = (static_cast<float>(x) + 0.5F) *
                                   prototype_width / input_size.width -
                               0.5F;
        const float clamped_x =
            std::clamp(source_x, 0.0F, static_cast<float>(prototype_width - 1));
        const int64_t x0 = static_cast<int64_t>(std::floor(clamped_x));
        const int64_t x1 = std::min(x0 + 1, prototype_width - 1);
        const float x_weight = clamped_x - static_cast<float>(x0);
        const float top_value =
            mask_logits[y0 * prototype_width + x0] * (1.0F - x_weight) +
            mask_logits[y0 * prototype_width + x1] * x_weight;
        const float bottom_value =
            mask_logits[y1 * prototype_width + x0] * (1.0F - x_weight) +
            mask_logits[y1 * prototype_width + x1] * x_weight;
        const float value =
            top_value * (1.0F - y_weight) + bottom_value * y_weight;
        masks[output_index] = value > value_threshold ? 1 : 0;
      }
    }
  }
  return output;
}

}  // namespace

YoloSegDecodeResult YoloSegDecode(const Tensor& predictions,
                                  const Tensor& prototypes,
                                  ImageSize input_size,
                                  YoloSegDecodeOptions options,
                                  TensorAllocator& allocator,
                                  ExecutionStream* execution_stream) {
  ValidateOptions(options);
  ValidateTensorInputs(predictions, prototypes, input_size);
  ValidateExecutionStream(predictions, execution_stream);
  const PredictionShape shape =
      ResolvePredictionShape(predictions, prototypes, options);

  YoloSegCandidates candidates;
  if (predictions.device().type == DeviceType::kCpu) {
    candidates =
        options.prediction_layout == YoloSegPredictionLayout::kRaw
            ? RunRawDecodeOnHost(predictions, shape, options, allocator)
            : RunSelectedDecodeOnHost(predictions, shape, options, allocator);
  } else if (predictions.device().type == DeviceType::kCuda) {
#if defined(MW_INFER_HAS_CUDA_POSTPROCESS)
    candidates =
        options.prediction_layout == YoloSegPredictionLayout::kRaw
            ? postprocess_internal::RunRawYoloSegDecodeOnDevice(
                  predictions, shape.batch_count, shape.channel_count,
                  shape.candidate_count, shape.channel_first, shape.class_count,
                  options, allocator, execution_stream)
            : postprocess_internal::RunSelectedYoloSegDecodeOnDevice(
                  predictions, shape.batch_count, shape.channel_count,
                  shape.candidate_count, shape.channel_first, options,
                  allocator, execution_stream);
#else
    throw std::runtime_error(
        "CUDA YOLO segmentation decode is unavailable in this build");
#endif
  } else {
    throw std::invalid_argument(
        "YOLO segmentation prediction device is unsupported");
  }

  Tensor keep;
  if (options.prediction_layout == YoloSegPredictionLayout::kRaw) {
    keep = postprocess_internal::RunClassAwareBatchNms(
        candidates.boxes, candidates.scores, candidates.class_ids,
        candidates.batch_ids, options.iou_threshold, 0.0F,
        options.max_detections, allocator, execution_stream);
  } else {
    keep = MakeSelectedIndices(candidates, shape.batch_count,
                               options.max_detections, allocator);
  }

  Tensor boxes = candidates.boxes.GatherRows(keep, allocator, execution_stream);
  Tensor scores =
      candidates.scores.GatherRows(keep, allocator, execution_stream);
  Tensor class_ids =
      candidates.class_ids.GatherRows(keep, allocator, execution_stream);
  Tensor batch_ids =
      candidates.batch_ids.GatherRows(keep, allocator, execution_stream);
  Tensor candidate_ids =
      candidates.candidate_ids.GatherRows(keep, allocator, execution_stream);
  Tensor masks;
  const bool scale_coefficients_by_objectness =
      options.prediction_layout == YoloSegPredictionLayout::kRaw &&
      HasObjectness(options.version);
  if (predictions.device().type == DeviceType::kCpu) {
    masks =
        RunMasksOnHost(predictions, prototypes, boxes, batch_ids, candidate_ids,
                       shape, scale_coefficients_by_objectness, input_size,
                       options.mask_threshold, allocator);
  } else {
#if defined(MW_INFER_HAS_CUDA_POSTPROCESS)
    masks = postprocess_internal::RunYoloSegMasksOnDevice(
        predictions, prototypes, boxes, batch_ids, candidate_ids,
        shape.channel_count, shape.candidate_count, shape.channel_first,
        shape.coefficient_start, scale_coefficients_by_objectness, input_size,
        options.mask_threshold, allocator, execution_stream);
#else
    throw std::runtime_error(
        "CUDA YOLO segmentation masks are unavailable in this build");
#endif
  }

  return YoloSegDecodeResult{std::move(boxes), std::move(scores),
                             std::move(class_ids), std::move(batch_ids),
                             std::move(masks)};
}

}  // namespace mw::infer
