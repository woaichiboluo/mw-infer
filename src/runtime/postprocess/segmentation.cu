#include <cuda_runtime_api.h>

#include <cfloat>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mw/infer/runtime/postprocess/segmentation.h"

namespace mw::infer::postprocess_internal {
namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kMaxGeometrySteps = 8;

struct DeviceGeometryStep {
  int kind = 0;
  int before_width = 0;
  int before_height = 0;
  int after_width = 0;
  int after_height = 0;
  int resized_width = 0;
  int resized_height = 0;
  int pad_left = 0;
  int pad_top = 0;
  int crop_x = 0;
  int crop_y = 0;
};

struct DeviceRestorePlan {
  int step_count = 0;
  int transformed_width = 0;
  int transformed_height = 0;
  DeviceGeometryStep steps[kMaxGeometrySteps];
};

std::string CudaErrorMessage(cudaError_t status, const char* operation) {
  return std::string(operation) + " failed: " + cudaGetErrorString(status);
}

void CheckCuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(CudaErrorMessage(status, operation));
  }
}

int CheckedInt64ToInt(int64_t value, const char* name) {
  if (value <= 0 || value > std::numeric_limits<int>::max()) {
    throw std::invalid_argument(std::string(name) + " exceeds int range");
  }
  return static_cast<int>(value);
}

int CheckedImageSizeValue(int value, const char* name) {
  if (value <= 0) {
    throw std::invalid_argument(std::string(name) + " must be positive");
  }
  return value;
}

int CheckedProductToInt(int64_t a, int64_t b, int64_t c, int64_t d,
                        const char* name) {
  if (a <= 0 || b <= 0 || c <= 0 || d <= 0) {
    throw std::invalid_argument(std::string(name) + " must be positive");
  }
  if (a > std::numeric_limits<int64_t>::max() / b ||
      a * b > std::numeric_limits<int64_t>::max() / c ||
      a * b * c > std::numeric_limits<int64_t>::max() / d) {
    throw std::invalid_argument(std::string(name) + " overflows int64");
  }
  const int64_t product = a * b * c * d;
  if (product > std::numeric_limits<int>::max()) {
    throw std::invalid_argument(std::string(name) + " exceeds int range");
  }
  return static_cast<int>(product);
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

int DeviceStepKind(GeometryStepKind kind) {
  switch (kind) {
    case GeometryStepKind::kResize:
      return 0;
    case GeometryStepKind::kPad:
      return 1;
    case GeometryStepKind::kCrop:
      return 2;
    case GeometryStepKind::kLetterBox:
      return 3;
  }
  return -1;
}

DeviceGeometryStep MakeDeviceStep(const GeometryStep& step) {
  DeviceGeometryStep device_step;
  device_step.kind = DeviceStepKind(step.kind);
  device_step.before_width =
      CheckedImageSizeValue(step.before_size.width, "Geometry step width");
  device_step.before_height =
      CheckedImageSizeValue(step.before_size.height, "Geometry step height");
  device_step.after_width =
      CheckedImageSizeValue(step.after_size.width, "Geometry step width");
  device_step.after_height =
      CheckedImageSizeValue(step.after_size.height, "Geometry step height");
  device_step.resized_width = step.letterbox.resized_size.width;
  device_step.resized_height = step.letterbox.resized_size.height;
  device_step.pad_left = step.pad.padding.left;
  device_step.pad_top = step.pad.padding.top;
  device_step.crop_x = step.crop.crop_rect.x;
  device_step.crop_y = step.crop.crop_rect.y;
  if (step.kind == GeometryStepKind::kLetterBox) {
    device_step.resized_width = CheckedImageSizeValue(
        step.letterbox.resized_size.width, "LetterBox resized width");
    device_step.resized_height = CheckedImageSizeValue(
        step.letterbox.resized_size.height, "LetterBox resized height");
    device_step.pad_left = step.letterbox.padding.left;
    device_step.pad_top = step.letterbox.padding.top;
  }
  return device_step;
}

std::vector<DeviceRestorePlan> BuildRestorePlans(
    int64_t batch, int64_t height, int64_t width,
    const std::vector<GeometryTrace>& traces, ImageSize* output_size) {
  if (batch != static_cast<int64_t>(traces.size())) {
    throw std::invalid_argument(
        "Segmentation logits batch size and geometry trace count mismatch");
  }

  const ImageSize fallback_size{
      CheckedInt64ToInt(width, "Segmentation width"),
      CheckedInt64ToInt(height, "Segmentation height")};
  std::vector<DeviceRestorePlan> plans;
  plans.reserve(traces.size());
  for (std::size_t index = 0; index < traces.size(); ++index) {
    if (traces[index].size() > kMaxGeometrySteps) {
      throw std::invalid_argument(
          "Segmentation geometry trace has too many steps for CUDA restore");
    }

    const ImageSize original_size =
        TraceOriginalSize(traces[index], fallback_size);
    const ImageSize transformed_size =
        TraceTransformedSize(traces[index], fallback_size);
    CheckedImageSizeValue(original_size.width, "Segmentation original width");
    CheckedImageSizeValue(original_size.height, "Segmentation original height");
    CheckedImageSizeValue(transformed_size.width,
                          "Segmentation transformed width");
    CheckedImageSizeValue(transformed_size.height,
                          "Segmentation transformed height");
    if (index == 0) {
      *output_size = original_size;
    } else if (output_size->width != original_size.width ||
               output_size->height != original_size.height) {
      throw std::invalid_argument(
          "Restored segmentation batch requires identical original sizes");
    }

    DeviceRestorePlan plan;
    plan.step_count = static_cast<int>(traces[index].size());
    plan.transformed_width = transformed_size.width;
    plan.transformed_height = transformed_size.height;
    for (std::size_t step_index = 0; step_index < traces[index].size();
         ++step_index) {
      plan.steps[step_index] = MakeDeviceStep(traces[index].step(step_index));
    }
    plans.push_back(plan);
  }

  return plans;
}

__global__ void SemanticSegmentationKernel(const float* logits,
                                           int64_t* class_ids, float* scores,
                                           int class_count, int height,
                                           int width, float binary_threshold,
                                           int pixel_count) {
  const int output_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (output_index >= pixel_count) {
    return;
  }

  const int64_t image_pixels = static_cast<int64_t>(height) * width;
  const int batch = output_index / image_pixels;
  const int64_t pixel = output_index - batch * image_pixels;
  const int64_t base = batch * class_count * image_pixels + pixel;

  if (class_count == 1) {
    const float value = logits[output_index];
    if (value == -FLT_MAX) {
      class_ids[output_index] = 0;
      scores[output_index] = 0.0F;
      return;
    }

    const float probability = 1.0F / (1.0F + expf(-value));
    const bool foreground = probability >= binary_threshold;
    class_ids[output_index] = foreground ? 1 : 0;
    scores[output_index] = foreground ? probability : 1.0F - probability;
    return;
  }

  int best_class = 0;
  float best_score = logits[base];
  for (int class_id = 1; class_id < class_count; ++class_id) {
    const float score = logits[base + class_id * image_pixels];
    if (score > best_score) {
      best_score = score;
      best_class = class_id;
    }
  }

  class_ids[output_index] = best_class;
  if (best_score == -FLT_MAX) {
    scores[output_index] = 0.0F;
    return;
  }

  float sum = 0.0F;
  for (int class_id = 0; class_id < class_count; ++class_id) {
    sum += expf(logits[base + class_id * image_pixels] - best_score);
  }
  scores[output_index] = 1.0F / sum;
}

__device__ void ForwardResize(int before_width, int before_height,
                              int after_width, int after_height, float* x,
                              float* y) {
  *x = (*x + 0.5F) * static_cast<float>(after_width) /
           static_cast<float>(before_width) -
       0.5F;
  *y = (*y + 0.5F) * static_cast<float>(after_height) /
           static_cast<float>(before_height) -
       0.5F;
}

__device__ void ApplyForwardStep(const DeviceGeometryStep& step, float* x,
                                 float* y) {
  if (step.kind == 0) {
    ForwardResize(step.before_width, step.before_height, step.after_width,
                  step.after_height, x, y);
    return;
  }
  if (step.kind == 1) {
    *x += static_cast<float>(step.pad_left);
    *y += static_cast<float>(step.pad_top);
    return;
  }
  if (step.kind == 2) {
    *x -= static_cast<float>(step.crop_x);
    *y -= static_cast<float>(step.crop_y);
    return;
  }
  if (step.kind == 3) {
    ForwardResize(step.before_width, step.before_height, step.resized_width,
                  step.resized_height, x, y);
    *x += static_cast<float>(step.pad_left);
    *y += static_cast<float>(step.pad_top);
  }
}

__device__ bool IsInsideSamplingBounds(float x, float y, int width,
                                       int height) {
  return x >= -0.5F && y >= -0.5F && x <= static_cast<float>(width) - 0.5F &&
         y <= static_cast<float>(height) - 0.5F;
}

__device__ bool MapOriginalPixelToLogits(const DeviceRestorePlan& plan,
                                         int logits_width, int logits_height,
                                         int x, int y, float* source_x,
                                         float* source_y) {
  *source_x = static_cast<float>(x);
  *source_y = static_cast<float>(y);
  for (int index = 0; index < plan.step_count; ++index) {
    ApplyForwardStep(plan.steps[index], source_x, source_y);
  }

  if (!IsInsideSamplingBounds(*source_x, *source_y, plan.transformed_width,
                              plan.transformed_height)) {
    return false;
  }

  ForwardResize(plan.transformed_width, plan.transformed_height, logits_width,
                logits_height, source_x, source_y);
  return IsInsideSamplingBounds(*source_x, *source_y, logits_width,
                                logits_height);
}

__device__ float SampleBilinear(const float* input, int batch, int class_id,
                                int class_count, int height, int width, float x,
                                float y) {
  const float clamped_x = fminf(fmaxf(x, 0.0F), static_cast<float>(width - 1));
  const float clamped_y = fminf(fmaxf(y, 0.0F), static_cast<float>(height - 1));
  const int x0 = static_cast<int>(floorf(clamped_x));
  const int y0 = static_cast<int>(floorf(clamped_y));
  const int x1 = (x0 + 1 < width) ? x0 + 1 : width - 1;
  const int y1 = (y0 + 1 < height) ? y0 + 1 : height - 1;
  const float wx = clamped_x - static_cast<float>(x0);
  const float wy = clamped_y - static_cast<float>(y0);
  const int64_t plane_size = static_cast<int64_t>(height) * width;
  const int64_t plane_base =
      (static_cast<int64_t>(batch) * class_count + class_id) * plane_size;

  const float top_left = input[plane_base + y0 * width + x0];
  const float top_right = input[plane_base + y0 * width + x1];
  const float bottom_left = input[plane_base + y1 * width + x0];
  const float bottom_right = input[plane_base + y1 * width + x1];
  const float top = top_left + (top_right - top_left) * wx;
  const float bottom = bottom_left + (bottom_right - bottom_left) * wx;
  return top + (bottom - top) * wy;
}

__global__ void RestoreSegmentationLogitsKernel(
    const float* logits, float* restored, const DeviceRestorePlan* plans,
    int class_count, int input_height, int input_width, int output_height,
    int output_width, int element_count) {
  const int output_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (output_index >= element_count) {
    return;
  }

  const int output_plane_size = output_height * output_width;
  const int pixel = output_index % output_plane_size;
  const int channel_index = output_index / output_plane_size;
  const int class_id = channel_index % class_count;
  const int batch = channel_index / class_count;
  const int y = pixel / output_width;
  const int x = pixel - y * output_width;

  float source_x = 0.0F;
  float source_y = 0.0F;
  if (!MapOriginalPixelToLogits(plans[batch], input_width, input_height, x, y,
                                &source_x, &source_y)) {
    restored[output_index] = -FLT_MAX;
    return;
  }

  restored[output_index] =
      SampleBilinear(logits, batch, class_id, class_count, input_height,
                     input_width, source_x, source_y);
}

}  // namespace

Tensor RunRestoreSegmentationLogitsOnDevice(
    const Tensor& logits, int64_t batch, int64_t classes, int64_t height,
    int64_t width, const std::vector<GeometryTrace>& traces,
    TensorAllocator& allocator) {
  CheckedInt64ToInt(batch, "Segmentation batch size");
  const int class_count =
      CheckedInt64ToInt(classes, "Segmentation class count");
  const int input_height = CheckedInt64ToInt(height, "Segmentation height");
  const int input_width = CheckedInt64ToInt(width, "Segmentation width");

  ImageSize output_size;
  const std::vector<DeviceRestorePlan> plans =
      BuildRestorePlans(batch, height, width, traces, &output_size);
  const int output_height = output_size.height;
  const int output_width = output_size.width;
  const int element_count =
      CheckedProductToInt(batch, classes, output_height, output_width,
                          "Restored segmentation logits element count");
  const std::vector<int64_t> output_shape = {batch, classes, output_height,
                                             output_width};

  CheckCuda(cudaSetDevice(logits.device().id), "cudaSetDevice");
  Tensor restored =
      Tensor::Allocate(MakeRestoredLogitsDesc(logits, output_shape), allocator);
  DeviceRestorePlan* device_plans = nullptr;
  CheckCuda(cudaMalloc(&device_plans, plans.size() * sizeof(DeviceRestorePlan)),
            "cudaMalloc");
  try {
    CheckCuda(cudaMemcpy(device_plans, plans.data(),
                         plans.size() * sizeof(DeviceRestorePlan),
                         cudaMemcpyHostToDevice),
              "cudaMemcpy");
    const int blocks =
        (element_count + kThreadsPerBlock - 1) / kThreadsPerBlock;
    RestoreSegmentationLogitsKernel<<<blocks, kThreadsPerBlock>>>(
        static_cast<const float*>(logits.data()),
        static_cast<float*>(restored.data()), device_plans, class_count,
        input_height, input_width, output_height, output_width, element_count);
    CheckCuda(cudaGetLastError(), "RestoreSegmentationLogitsKernel");
    CheckCuda(cudaFree(device_plans), "cudaFree");
    device_plans = nullptr;
  } catch (...) {
    if (device_plans != nullptr) {
      static_cast<void>(cudaFree(device_plans));
    }
    throw;
  }

  return restored;
}

SemanticSegmentationResult RunSemanticSegmentationOnDevice(
    const Tensor& logits, int64_t batch, int64_t classes, int64_t height,
    int64_t width, float binary_threshold, TensorAllocator& allocator) {
  CheckedInt64ToInt(batch, "Segmentation batch size");
  const int class_count =
      CheckedInt64ToInt(classes, "Segmentation class count");
  const int output_height = CheckedInt64ToInt(height, "Segmentation height");
  const int output_width = CheckedInt64ToInt(width, "Segmentation width");
  if (batch > std::numeric_limits<int>::max() / height ||
      batch * height > std::numeric_limits<int>::max() / width) {
    throw std::invalid_argument(
        "Segmentation output pixel count exceeds int range");
  }
  const int pixel_count = static_cast<int>(batch * height * width);
  const std::vector<int64_t> output_shape = {batch, height, width};

  CheckCuda(cudaSetDevice(logits.device().id), "cudaSetDevice");
  Tensor class_ids = Tensor::Allocate(
      MakeClassIdDesc(output_shape, logits.device()), allocator);
  Tensor scores =
      Tensor::Allocate(MakeScoreDesc(output_shape, logits.device()), allocator);

  const int blocks = (pixel_count + kThreadsPerBlock - 1) / kThreadsPerBlock;
  SemanticSegmentationKernel<<<blocks, kThreadsPerBlock>>>(
      static_cast<const float*>(logits.data()),
      static_cast<int64_t*>(class_ids.data()),
      static_cast<float*>(scores.data()), class_count, output_height,
      output_width, binary_threshold, pixel_count);
  CheckCuda(cudaGetLastError(), "SemanticSegmentationKernel");

  return SemanticSegmentationResult{std::move(class_ids), std::move(scores)};
}

}  // namespace mw::infer::postprocess_internal
