#include "mw/infer/runtime/postprocess/yolo_decode.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mw/infer/runtime/execution_stream.h"

namespace mw::infer {

#if defined(MW_INFER_HAS_CUDA_POSTPROCESS)
namespace postprocess_internal {
YoloDecodeResult RunYoloDecodeOnDevice(
    const Tensor& predictions, int64_t batch_count, int64_t channel_count,
    int64_t candidate_count, bool channel_first, YoloDecodeOptions options,
    TensorAllocator& allocator, ExecutionStream* execution_stream);
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
    case YoloVersion::kYoloV26:
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
    case YoloVersion::kYoloV26:
      return "YOLOv26";
  }
  return "YOLO";
}

void ValidateOptions(YoloDecodeOptions options) {
  static_cast<void>(ClassStart(options.version));
}

bool ResolveChannelFirst(const std::vector<int64_t>& shape,
                         YoloDecodeOptions options) {
  const int64_t min_channel_count = ClassStart(options.version) + 1;
  const bool dim1_can_be_channels = shape[1] >= min_channel_count;
  const bool dim2_can_be_channels = shape[2] >= min_channel_count;
  switch (options.tensor_layout) {
    case YoloTensorLayout::kChannelFirst:
      if (!dim1_can_be_channels) {
        throw std::invalid_argument(
            "YOLO channel-first tensor has too few channels");
      }
      return true;
    case YoloTensorLayout::kCandidateFirst:
      if (!dim2_can_be_channels) {
        throw std::invalid_argument(
            "YOLO candidate-first tensor has too few channels");
      }
      return false;
    case YoloTensorLayout::kAuto:
      break;
    default:
      throw std::invalid_argument("YOLO tensor layout is unsupported");
  }

  if (dim1_can_be_channels && !dim2_can_be_channels) {
    return true;
  }
  if (!dim1_can_be_channels && dim2_can_be_channels) {
    return false;
  }
  if (!dim1_can_be_channels && !dim2_can_be_channels) {
    throw std::invalid_argument(
        "YOLO tensor layout cannot be inferred from the prediction shape");
  }
  return options.version != YoloVersion::kYoloV5;
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

void ValidateExecutionStream(const Tensor& predictions,
                             const ExecutionStream* execution_stream) {
  if (execution_stream == nullptr) {
    return;
  }
  const Device stream_device = execution_stream->device();
  if (stream_device.type != predictions.device().type ||
      stream_device.id != predictions.device().id) {
    throw std::invalid_argument(
        "YOLO execution stream device does not match prediction device");
  }
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

YoloDecodeResult AllocateResult(const Tensor& predictions, YoloShape shape,
                                int64_t class_count,
                                TensorAllocator& allocator) {
  YoloDecodeResult result;
  result.boxes = Tensor::Allocate(
      MakeFloatDesc("yolo_boxes",
                    {shape.batch_count, shape.candidate_count, 4},
                    predictions.device()),
      allocator);
  result.scores = Tensor::Allocate(
      MakeFloatDesc("yolo_scores",
                    {shape.batch_count, shape.candidate_count, class_count},
                    predictions.device()),
      allocator);
  return result;
}

YoloDecodeResult RunYoloDecodeOnHost(const Tensor& predictions, YoloShape shape,
                                     YoloDecodeOptions options,
                                     TensorAllocator& allocator) {
  const auto* input = static_cast<const float*>(predictions.data());
  const int64_t class_start = ClassStart(options.version);
  const int64_t class_count = shape.channel_count - class_start;
  const bool has_objectness = HasObjectness(options.version);

  YoloDecodeResult result =
      AllocateResult(predictions, shape, class_count, allocator);
  auto* boxes = static_cast<float*>(result.boxes.data());
  auto* scores = static_cast<float*>(result.scores.data());

  for (int64_t batch = 0; batch < shape.batch_count; ++batch) {
    for (int64_t candidate = 0; candidate < shape.candidate_count;
         ++candidate) {
      const float center_x = ValueAt(input, shape, batch, 0, candidate);
      const float center_y = ValueAt(input, shape, batch, 1, candidate);
      const float width = ValueAt(input, shape, batch, 2, candidate);
      const float height = ValueAt(input, shape, batch, 3, candidate);
      const float left = center_x - width * 0.5F;
      const float top = center_y - height * 0.5F;
      const float right = center_x + width * 0.5F;
      const float bottom = center_y + height * 0.5F;
      const std::size_t candidate_index = static_cast<std::size_t>(
          batch * shape.candidate_count + candidate);
      float* output_box = boxes + candidate_index * 4U;
      output_box[0] = left;
      output_box[1] = top;
      output_box[2] = right;
      output_box[3] = bottom;

      const float objectness =
          has_objectness ? ValueAt(input, shape, batch, 4, candidate) : 1.0F;
      float* output_scores =
          scores + candidate_index * static_cast<std::size_t>(class_count);
      for (int64_t class_index = 0; class_index < class_count; ++class_index) {
        const float class_score =
            ValueAt(input, shape, batch, class_start + class_index, candidate);
        output_scores[class_index] = objectness * class_score;
      }
    }
  }
  return result;
}

}  // namespace

YoloDecodeResult YoloDecode(const Tensor& predictions,
                            YoloDecodeOptions options,
                            TensorAllocator& allocator,
                            ExecutionStream* execution_stream) {
  ValidateOptions(options);
  const YoloShape shape = ValidatePredictions(predictions, options);
  ValidateExecutionStream(predictions, execution_stream);
  if (predictions.device().type == DeviceType::kCpu) {
    return RunYoloDecodeOnHost(predictions, shape, options, allocator);
  }
  if (predictions.device().type == DeviceType::kCuda) {
#if defined(MW_INFER_HAS_CUDA_POSTPROCESS)
    return postprocess_internal::RunYoloDecodeOnDevice(
        predictions, shape.batch_count, shape.channel_count,
        shape.candidate_count, shape.channel_first, options, allocator,
        execution_stream);
#else
    throw std::runtime_error("CUDA YOLO decode is unavailable in this build");
#endif
  }
  throw std::invalid_argument("YOLO predictions tensor device is unsupported");
}

}  // namespace mw::infer
