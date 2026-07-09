#include "mw/infer/runtime/postprocess/segmentation.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mw::infer {

#if defined(MW_INFER_HAS_CUDA_POSTPROCESS)
namespace postprocess_internal {
Tensor RunRestoreSegmentationLogitsOnDevice(
    const Tensor& logits, int64_t batch, int64_t classes, int64_t height,
    int64_t width, const std::vector<GeometryTrace>& traces,
    TensorAllocator& allocator);

SemanticSegmentationResult RunSemanticSegmentationOnDevice(
    const Tensor& logits, int64_t batch, int64_t classes, int64_t height,
    int64_t width, float binary_threshold, TensorAllocator& allocator);
}  // namespace postprocess_internal
#endif

namespace {

struct SegmentationShape {
  int64_t batch = 0;
  int64_t classes = 0;
  int64_t height = 0;
  int64_t width = 0;
};

struct RestorePlan {
  ImageSize output_size;
  std::vector<ImageSize> transformed_sizes;
};

struct SampleCoordinate {
  float x = 0.0F;
  float y = 0.0F;
  bool valid = true;
};

int CheckedInt64ToInt(int64_t value, const char* name) {
  if (value <= 0 || value > std::numeric_limits<int>::max()) {
    throw std::invalid_argument(std::string(name) + " exceeds int range");
  }
  return static_cast<int>(value);
}

ImageSize CheckedImageSize(int64_t width, int64_t height, const char* name) {
  return ImageSize{CheckedInt64ToInt(width, name),
                   CheckedInt64ToInt(height, name)};
}

void ValidateImageSize(ImageSize size, const char* name) {
  if (size.width <= 0 || size.height <= 0) {
    throw std::invalid_argument(std::string(name) + " must be positive");
  }
}

SegmentationShape ValidateLogits(const Tensor& logits) {
  if (logits.empty()) {
    throw std::invalid_argument("Segmentation logits tensor is empty");
  }
  if (logits.data_type() != DataType::kFloat32) {
    throw std::invalid_argument("Segmentation logits tensor must be float32");
  }
  if (logits.shape().size() != 4) {
    throw std::invalid_argument(
        "Segmentation logits tensor shape must be [N, C, H, W]");
  }

  SegmentationShape shape;
  shape.batch = logits.shape()[0];
  shape.classes = logits.shape()[1];
  shape.height = logits.shape()[2];
  shape.width = logits.shape()[3];
  if (shape.batch <= 0 || shape.classes <= 0 || shape.height <= 0 ||
      shape.width <= 0) {
    throw std::invalid_argument(
        "Segmentation logits tensor dimensions must be positive");
  }
  return shape;
}

void ValidateOptions(const SemanticSegmentationOptions& options) {
  if (!std::isfinite(options.binary_threshold) ||
      options.binary_threshold < 0.0F || options.binary_threshold > 1.0F) {
    throw std::invalid_argument(
        "Semantic segmentation binary threshold must be in [0, 1]");
  }
}

ImageSize TraceOriginalSize(const GeometryTrace& trace,
                            ImageSize fallback_size) {
  if (trace.empty()) {
    return fallback_size;
  }
  return trace.step(0).before_size;
}

ImageSize TraceTransformedSize(const GeometryTrace& trace,
                               ImageSize fallback_size) {
  if (trace.empty()) {
    return fallback_size;
  }
  return trace.step(trace.size() - 1).after_size;
}

RestorePlan BuildRestorePlan(SegmentationShape shape,
                             const std::vector<GeometryTrace>& traces) {
  if (shape.batch != static_cast<int64_t>(traces.size())) {
    throw std::invalid_argument(
        "Segmentation logits batch size and geometry trace count mismatch");
  }

  const ImageSize logits_size =
      CheckedImageSize(shape.width, shape.height, "Segmentation logits size");
  RestorePlan plan;
  plan.transformed_sizes.reserve(traces.size());

  for (std::size_t index = 0; index < traces.size(); ++index) {
    const ImageSize original_size =
        TraceOriginalSize(traces[index], logits_size);
    const ImageSize transformed_size =
        TraceTransformedSize(traces[index], logits_size);
    ValidateImageSize(original_size, "Segmentation original size");
    ValidateImageSize(transformed_size, "Segmentation transformed size");

    if (index == 0) {
      plan.output_size = original_size;
    } else if (plan.output_size.width != original_size.width ||
               plan.output_size.height != original_size.height) {
      throw std::invalid_argument(
          "Restored segmentation batch requires identical original sizes");
    }
    plan.transformed_sizes.push_back(transformed_size);
  }

  return plan;
}

TensorDesc MakeClassIdDesc(std::vector<int64_t> shape, Device device) {
  TensorDesc desc;
  desc.info.name = "segmentation_class_ids";
  desc.info.data_type = DataType::kInt64;
  desc.info.shape = std::move(shape);
  desc.device = device;
  return desc;
}

TensorDesc MakeScoreDesc(std::vector<int64_t> shape, Device device) {
  TensorDesc desc;
  desc.info.name = "segmentation_scores";
  desc.info.data_type = DataType::kFloat32;
  desc.info.shape = std::move(shape);
  desc.device = device;
  return desc;
}

std::string RestoredLogitsName(const Tensor& logits) {
  if (logits.name().empty()) {
    return "segmentation_logits_restored";
  }
  return logits.name() + "_restored";
}

TensorDesc MakeRestoredLogitsDesc(const Tensor& logits,
                                  std::vector<int64_t> shape) {
  TensorDesc desc;
  desc.info.name = RestoredLogitsName(logits);
  desc.info.data_type = DataType::kFloat32;
  desc.info.shape = std::move(shape);
  desc.device = logits.device();
  return desc;
}

std::vector<int64_t> OutputShape(SegmentationShape shape) {
  return {shape.batch, shape.height, shape.width};
}

void ForwardResize(ImageSize before_size, ImageSize after_size, float* x,
                   float* y) {
  *x = (*x + 0.5F) * static_cast<float>(after_size.width) /
           static_cast<float>(before_size.width) -
       0.5F;
  *y = (*y + 0.5F) * static_cast<float>(after_size.height) /
           static_cast<float>(before_size.height) -
       0.5F;
}

void ApplyForwardStep(const GeometryStep& step, float* x, float* y) {
  switch (step.kind) {
    case GeometryStepKind::kResize:
      ForwardResize(step.before_size, step.after_size, x, y);
      return;
    case GeometryStepKind::kPad:
      *x += static_cast<float>(step.pad.padding.left);
      *y += static_cast<float>(step.pad.padding.top);
      return;
    case GeometryStepKind::kCrop:
      *x -= static_cast<float>(step.crop.crop_rect.x);
      *y -= static_cast<float>(step.crop.crop_rect.y);
      return;
    case GeometryStepKind::kLetterBox:
      ForwardResize(step.before_size, step.letterbox.resized_size, x, y);
      *x += static_cast<float>(step.letterbox.padding.left);
      *y += static_cast<float>(step.letterbox.padding.top);
      return;
  }
}

bool IsInsideSamplingBounds(float x, float y, ImageSize size) {
  return x >= -0.5F && y >= -0.5F &&
         x <= static_cast<float>(size.width) - 0.5F &&
         y <= static_cast<float>(size.height) - 0.5F;
}

SampleCoordinate MapOriginalPixelToLogits(const GeometryTrace& trace,
                                          ImageSize transformed_size,
                                          ImageSize logits_size, int x, int y) {
  float source_x = static_cast<float>(x);
  float source_y = static_cast<float>(y);
  for (const GeometryStep& step : trace.steps()) {
    ApplyForwardStep(step, &source_x, &source_y);
  }

  if (!IsInsideSamplingBounds(source_x, source_y, transformed_size)) {
    return SampleCoordinate{0.0F, 0.0F, false};
  }

  ForwardResize(transformed_size, logits_size, &source_x, &source_y);
  if (!IsInsideSamplingBounds(source_x, source_y, logits_size)) {
    return SampleCoordinate{0.0F, 0.0F, false};
  }

  return SampleCoordinate{source_x, source_y, true};
}

float SampleBilinear(const float* input, SegmentationShape shape, int64_t batch,
                     int64_t class_id, SampleCoordinate coordinate) {
  const float x =
      std::clamp(coordinate.x, 0.0F, static_cast<float>(shape.width - 1));
  const float y =
      std::clamp(coordinate.y, 0.0F, static_cast<float>(shape.height - 1));
  const int64_t x0 = static_cast<int64_t>(std::floor(x));
  const int64_t y0 = static_cast<int64_t>(std::floor(y));
  const int64_t x1 = std::min<int64_t>(x0 + 1, shape.width - 1);
  const int64_t y1 = std::min<int64_t>(y0 + 1, shape.height - 1);
  const float wx = x - static_cast<float>(x0);
  const float wy = y - static_cast<float>(y0);
  const int64_t plane_size = shape.height * shape.width;
  const int64_t plane_base = (batch * shape.classes + class_id) * plane_size;

  const float top_left = input[plane_base + y0 * shape.width + x0];
  const float top_right = input[plane_base + y0 * shape.width + x1];
  const float bottom_left = input[plane_base + y1 * shape.width + x0];
  const float bottom_right = input[plane_base + y1 * shape.width + x1];
  const float top = top_left + (top_right - top_left) * wx;
  const float bottom = bottom_left + (bottom_right - bottom_left) * wx;
  return top + (bottom - top) * wy;
}

float Sigmoid(float value) {
  if (value >= 0.0F) {
    return 1.0F / (1.0F + std::exp(-value));
  }
  const float exp_value = std::exp(value);
  return exp_value / (1.0F + exp_value);
}

bool IsRestoreInvalidScore(float value) {
  return value == std::numeric_limits<float>::lowest();
}

float SelectedSoftmaxProbability(const float* input, int64_t base,
                                 int64_t plane_size, int64_t classes,
                                 float selected_logit) {
  float sum = 0.0F;
  for (int64_t class_id = 0; class_id < classes; ++class_id) {
    sum += std::exp(input[base + class_id * plane_size] - selected_logit);
  }
  return 1.0F / sum;
}

Tensor RunRestoreSegmentationLogitsOnHost(
    const Tensor& logits, SegmentationShape shape,
    const std::vector<GeometryTrace>& traces, const RestorePlan& plan,
    TensorAllocator& allocator) {
  const std::vector<int64_t> output_shape = {shape.batch, shape.classes,
                                             plan.output_size.height,
                                             plan.output_size.width};
  Tensor output =
      Tensor::Allocate(MakeRestoredLogitsDesc(logits, output_shape), allocator);

  const auto* input = logits.data<float>();
  auto* restored = output.data<float>();
  const ImageSize logits_size =
      CheckedImageSize(shape.width, shape.height, "Segmentation logits size");
  const int64_t output_plane_size =
      static_cast<int64_t>(plan.output_size.height) * plan.output_size.width;
  const float invalid_score = std::numeric_limits<float>::lowest();

  for (int64_t batch = 0; batch < shape.batch; ++batch) {
    const GeometryTrace& trace = traces[static_cast<std::size_t>(batch)];
    const ImageSize transformed_size =
        plan.transformed_sizes[static_cast<std::size_t>(batch)];
    for (int y = 0; y < plan.output_size.height; ++y) {
      for (int x = 0; x < plan.output_size.width; ++x) {
        const SampleCoordinate coordinate = MapOriginalPixelToLogits(
            trace, transformed_size, logits_size, x, y);
        const int64_t pixel =
            static_cast<int64_t>(y) * plan.output_size.width + x;
        for (int64_t class_id = 0; class_id < shape.classes; ++class_id) {
          const int64_t output_index =
              (batch * shape.classes + class_id) * output_plane_size + pixel;
          restored[output_index] =
              coordinate.valid
                  ? SampleBilinear(input, shape, batch, class_id, coordinate)
                  : invalid_score;
        }
      }
    }
  }

  return output;
}

SemanticSegmentationResult RunSemanticSegmentationOnHost(
    const Tensor& logits, SegmentationShape shape,
    const SemanticSegmentationOptions& options, TensorAllocator& allocator) {
  const std::vector<int64_t> output_shape = OutputShape(shape);
  Tensor class_ids = Tensor::Allocate(
      MakeClassIdDesc(output_shape, logits.device()), allocator);
  Tensor scores =
      Tensor::Allocate(MakeScoreDesc(output_shape, logits.device()), allocator);

  const auto* input = static_cast<const float*>(logits.data());
  auto* output_ids = static_cast<int64_t*>(class_ids.data());
  auto* output_scores = static_cast<float*>(scores.data());
  const int64_t plane_size = shape.height * shape.width;

  for (int64_t batch = 0; batch < shape.batch; ++batch) {
    for (int64_t pixel = 0; pixel < plane_size; ++pixel) {
      const int64_t output_index = batch * plane_size + pixel;
      if (shape.classes == 1) {
        const float value = input[output_index];
        if (IsRestoreInvalidScore(value)) {
          output_ids[output_index] = 0;
          output_scores[output_index] = 0.0F;
          continue;
        }

        const float probability = Sigmoid(value);
        const bool foreground = probability >= options.binary_threshold;
        output_ids[output_index] = foreground ? 1 : 0;
        output_scores[output_index] =
            foreground ? probability : 1.0F - probability;
        continue;
      }

      const int64_t base = batch * shape.classes * plane_size + pixel;
      int64_t best_class = 0;
      float best_score = input[base];
      for (int64_t class_id = 1; class_id < shape.classes; ++class_id) {
        const float score = input[base + class_id * plane_size];
        if (score > best_score) {
          best_score = score;
          best_class = class_id;
        }
      }

      output_ids[output_index] = best_class;
      output_scores[output_index] =
          IsRestoreInvalidScore(best_score)
              ? 0.0F
              : SelectedSoftmaxProbability(input, base, plane_size,
                                           shape.classes, best_score);
    }
  }

  return SemanticSegmentationResult{std::move(class_ids), std::move(scores)};
}

}  // namespace

SemanticSegmentationResult SemanticSegmentation(
    const Tensor& logits, const SemanticSegmentationOptions& options,
    TensorAllocator& allocator) {
  const SegmentationShape shape = ValidateLogits(logits);
  ValidateOptions(options);
  if (logits.device().type == DeviceType::kCpu) {
    return RunSemanticSegmentationOnHost(logits, shape, options, allocator);
  }
  if (logits.device().type == DeviceType::kCuda) {
#if defined(MW_INFER_HAS_CUDA_POSTPROCESS)
    return postprocess_internal::RunSemanticSegmentationOnDevice(
        logits, shape.batch, shape.classes, shape.height, shape.width,
        options.binary_threshold, allocator);
#else
    throw std::runtime_error(
        "CUDA semantic segmentation is unavailable in this build");
#endif
  }
  throw std::invalid_argument(
      "Segmentation logits tensor device is unsupported");
}

Tensor RestoreSegmentationLogits(const Tensor& logits,
                                 const std::vector<GeometryTrace>& traces,
                                 TensorAllocator& allocator) {
  const SegmentationShape shape = ValidateLogits(logits);
  const RestorePlan plan = BuildRestorePlan(shape, traces);
  if (logits.device().type == DeviceType::kCpu) {
    return RunRestoreSegmentationLogitsOnHost(logits, shape, traces, plan,
                                              allocator);
  }
  if (logits.device().type == DeviceType::kCuda) {
#if defined(MW_INFER_HAS_CUDA_POSTPROCESS)
    return postprocess_internal::RunRestoreSegmentationLogitsOnDevice(
        logits, shape.batch, shape.classes, shape.height, shape.width, traces,
        allocator);
#else
    throw std::runtime_error(
        "CUDA segmentation logits restore is unavailable in this build");
#endif
  }
  throw std::invalid_argument(
      "Segmentation logits tensor device is unsupported");
}

SemanticSegmentationResult SemanticSegmentation(
    const Tensor& logits, const std::vector<GeometryTrace>& traces,
    const SemanticSegmentationOptions& options, TensorAllocator& allocator) {
  ValidateOptions(options);
  Tensor restored_logits = RestoreSegmentationLogits(logits, traces, allocator);
  return SemanticSegmentation(restored_logits, options, allocator);
}

}  // namespace mw::infer
