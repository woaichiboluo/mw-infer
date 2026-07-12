#include "mw/infer/runtime/process/geometry.h"

#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>

namespace mw::infer {

#if defined(MW_INFER_HAS_OPENCV_GEOMETRY_ADAPTER)
std::unique_ptr<GeometryAdapter> CreateOpenCvMatGeometryAdapter();
#endif

#if defined(MW_INFER_HAS_OPENCV_CUDA_ADAPTER)
std::unique_ptr<GeometryAdapter> CreateOpenCvCudaGeometryAdapter();
#endif

namespace {

void ValidateSize(ImageSize size, const char* name) {
  if (size.width <= 0 || size.height <= 0) {
    throw std::invalid_argument(std::string(name) + " must be positive");
  }
}

void ValidatePadding(Padding padding) {
  if (padding.left < 0 || padding.top < 0 || padding.right < 0 ||
      padding.bottom < 0) {
    throw std::invalid_argument("Padding values must be non-negative");
  }
}

void ValidateCropRect(ImageSize image_size, Rect rect) {
  if (rect.width <= 0 || rect.height <= 0) {
    throw std::invalid_argument("Crop rect size must be positive");
  }
  if (rect.x < 0 || rect.y < 0 || rect.x + rect.width > image_size.width ||
      rect.y + rect.height > image_size.height) {
    throw std::invalid_argument("Crop rect is outside image bounds");
  }
}

ImageSize ResizeKeepRatioSize(ImageSize input_size, ImageSize target_size) {
  const double scale = std::min(static_cast<double>(target_size.width) /
                                    static_cast<double>(input_size.width),
                                static_cast<double>(target_size.height) /
                                    static_cast<double>(input_size.height));
  const int resized_width =
      std::max(1, static_cast<int>(std::round(input_size.width * scale)));
  const int resized_height =
      std::max(1, static_cast<int>(std::round(input_size.height * scale)));
  return ImageSize{resized_width, resized_height};
}

Padding CenterPadding(ImageSize resized_size, ImageSize target_size) {
  const int pad_width = target_size.width - resized_size.width;
  const int pad_height = target_size.height - resized_size.height;
  if (pad_width < 0 || pad_height < 0) {
    throw std::invalid_argument("Resized image is larger than target size");
  }
  return Padding{pad_width / 2, pad_height / 2, pad_width - pad_width / 2,
                 pad_height - pad_height / 2};
}

Rect CenterCropRect(ImageSize image_size, ImageSize crop_size) {
  ValidateSize(crop_size, "CenterCrop size");
  if (crop_size.width > image_size.width ||
      crop_size.height > image_size.height) {
    throw std::invalid_argument("CenterCrop size is larger than image size");
  }
  return Rect{(image_size.width - crop_size.width) / 2,
              (image_size.height - crop_size.height) / 2, crop_size.width,
              crop_size.height};
}

bool SameDevice(Device lhs, Device rhs) {
  return lhs.type == rhs.type && lhs.id == rhs.id;
}

void ValidateStreamDevice(const RawImage& image, ExecutionStream& stream,
                          const char* operation) {
  if (!SameDevice(image.device(), stream.device())) {
    throw std::invalid_argument(std::string(operation) +
                                " stream device does not match image device");
  }
}

const char* GeometryStepName(GeometryStepKind kind) {
  switch (kind) {
    case GeometryStepKind::kResize:
      return "Resize";
    case GeometryStepKind::kPad:
      return "Pad";
    case GeometryStepKind::kCrop:
      return "Crop";
    case GeometryStepKind::kLetterBox:
      return "LetterBox";
  }
  return "Unknown";
}

void AppendPadding(fmt::memory_buffer& buffer, Padding padding) {
  fmt::format_to(std::back_inserter(buffer),
                 "{{left={},top={},right={},bottom={}}}", padding.left,
                 padding.top, padding.right, padding.bottom);
}

void AppendRect(fmt::memory_buffer& buffer, Rect rect) {
  fmt::format_to(std::back_inserter(buffer), "{{x={},y={},width={},height={}}}",
                 rect.x, rect.y, rect.width, rect.height);
}

void AppendStep(fmt::memory_buffer& buffer, std::size_t index,
                const GeometryStep& step) {
  fmt::format_to(std::back_inserter(buffer), "#{} {} before={}x{} after={}x{}",
                 index, GeometryStepName(step.kind), step.before_size.width,
                 step.before_size.height, step.after_size.width,
                 step.after_size.height);

  switch (step.kind) {
    case GeometryStepKind::kResize:
      fmt::format_to(std::back_inserter(buffer),
                     " scale_x={:.6g} scale_y={:.6g}", step.resize.scale_x,
                     step.resize.scale_y);
      break;
    case GeometryStepKind::kPad:
      fmt::format_to(std::back_inserter(buffer), " padding=");
      AppendPadding(buffer, step.pad.padding);
      break;
    case GeometryStepKind::kCrop:
      fmt::format_to(std::back_inserter(buffer), " rect=");
      AppendRect(buffer, step.crop.crop_rect);
      break;
    case GeometryStepKind::kLetterBox:
      fmt::format_to(std::back_inserter(buffer),
                     " resized={}x{} scale_x={:.6g} scale_y={:.6g} padding=",
                     step.letterbox.resized_size.width,
                     step.letterbox.resized_size.height, step.letterbox.scale_x,
                     step.letterbox.scale_y);
      AppendPadding(buffer, step.letterbox.padding);
      break;
  }
}

Point2f RestorePointByStep(Point2f point, const GeometryStep& step) {
  switch (step.kind) {
    case GeometryStepKind::kResize:
      return Point2f{point.x / step.resize.scale_x,
                     point.y / step.resize.scale_y};
    case GeometryStepKind::kPad:
      return Point2f{point.x - static_cast<float>(step.pad.padding.left),
                     point.y - static_cast<float>(step.pad.padding.top)};
    case GeometryStepKind::kCrop:
      return Point2f{point.x + static_cast<float>(step.crop.crop_rect.x),
                     point.y + static_cast<float>(step.crop.crop_rect.y)};
    case GeometryStepKind::kLetterBox:
      return Point2f{
          (point.x - static_cast<float>(step.letterbox.padding.left)) /
              step.letterbox.scale_x,
          (point.y - static_cast<float>(step.letterbox.padding.top)) /
              step.letterbox.scale_y};
  }
  return point;
}

RawImageBatch MakeRawImageBatch(std::vector<RawImage> images) {
  return RawImageBatch(std::move(images));
}

}  // namespace

ImageSize ResizeShortSideSize(ImageSize input_size, int short_side) {
  ValidateSize(input_size, "ResizeShortSide input size");
  if (short_side <= 0) {
    throw std::invalid_argument("ResizeShortSide short side must be positive");
  }

  if (input_size.width <= input_size.height) {
    const int height = std::max(
        1, static_cast<int>(std::round(static_cast<double>(input_size.height) *
                                       short_side / input_size.width)));
    return ImageSize{short_side, height};
  }

  const int width = std::max(
      1, static_cast<int>(std::round(static_cast<double>(input_size.width) *
                                     short_side / input_size.height)));
  return ImageSize{width, short_side};
}

bool GeometryTrace::empty() const { return steps_.empty(); }

std::size_t GeometryTrace::size() const { return steps_.size(); }

const std::vector<GeometryStep>& GeometryTrace::steps() const { return steps_; }

const GeometryStep& GeometryTrace::step(std::size_t index) const {
  return steps_.at(index);
}

void GeometryTrace::AddResize(ImageSize before_size, ImageSize after_size) {
  ValidateSize(before_size, "Resize before size");
  ValidateSize(after_size, "Resize after size");

  GeometryStep step;
  step.kind = GeometryStepKind::kResize;
  step.before_size = before_size;
  step.after_size = after_size;
  step.resize.scale_x = static_cast<float>(after_size.width) /
                        static_cast<float>(before_size.width);
  step.resize.scale_y = static_cast<float>(after_size.height) /
                        static_cast<float>(before_size.height);
  steps_.push_back(step);
}

void GeometryTrace::AddPad(ImageSize before_size, Padding padding) {
  ValidateSize(before_size, "Pad before size");
  ValidatePadding(padding);

  GeometryStep step;
  step.kind = GeometryStepKind::kPad;
  step.before_size = before_size;
  step.after_size =
      ImageSize{before_size.width + padding.left + padding.right,
                before_size.height + padding.top + padding.bottom};
  step.pad.padding = padding;
  steps_.push_back(step);
}

void GeometryTrace::AddCrop(ImageSize before_size, Rect crop_rect) {
  ValidateSize(before_size, "Crop before size");
  ValidateCropRect(before_size, crop_rect);

  GeometryStep step;
  step.kind = GeometryStepKind::kCrop;
  step.before_size = before_size;
  step.after_size = ImageSize{crop_rect.width, crop_rect.height};
  step.crop.crop_rect = crop_rect;
  steps_.push_back(step);
}

void GeometryTrace::AddLetterBox(ImageSize before_size, ImageSize after_size,
                                 ImageSize resized_size, Padding padding) {
  ValidateSize(before_size, "LetterBox before size");
  ValidateSize(after_size, "LetterBox after size");
  ValidateSize(resized_size, "LetterBox resized size");
  ValidatePadding(padding);

  if (resized_size.width + padding.left + padding.right != after_size.width ||
      resized_size.height + padding.top + padding.bottom != after_size.height) {
    throw std::invalid_argument(
        "LetterBox resized size and padding do not match after size");
  }

  GeometryStep step;
  step.kind = GeometryStepKind::kLetterBox;
  step.before_size = before_size;
  step.after_size = after_size;
  step.letterbox.resized_size = resized_size;
  step.letterbox.scale_x = static_cast<float>(resized_size.width) /
                           static_cast<float>(before_size.width);
  step.letterbox.scale_y = static_cast<float>(resized_size.height) /
                           static_cast<float>(before_size.height);
  step.letterbox.padding = padding;
  steps_.push_back(step);
}

Point2f GeometryTrace::RestorePoint(Point2f point) const {
  for (auto it = steps_.rbegin(); it != steps_.rend(); ++it) {
    point = RestorePointByStep(point, *it);
  }
  return point;
}

Rect2f GeometryTrace::RestoreRect(Rect2f rect) const {
  const Point2f top_left = RestorePoint(Point2f{rect.x, rect.y});
  const Point2f bottom_right =
      RestorePoint(Point2f{rect.x + rect.width, rect.y + rect.height});
  const float left = std::min(top_left.x, bottom_right.x);
  const float top = std::min(top_left.y, bottom_right.y);
  const float right = std::max(top_left.x, bottom_right.x);
  const float bottom = std::max(top_left.y, bottom_right.y);
  return Rect2f{left, top, right - left, bottom - top};
}

std::vector<Point2f> GeometryTrace::RestorePolygon(
    const std::vector<Point2f>& polygon) const {
  std::vector<Point2f> restored;
  restored.reserve(polygon.size());
  for (Point2f point : polygon) {
    restored.push_back(RestorePoint(point));
  }
  return restored;
}

std::string GeometryTrace::Dump() const {
  fmt::memory_buffer buffer;
  for (std::size_t index = 0; index < steps_.size(); ++index) {
    if (index > 0) {
      fmt::format_to(std::back_inserter(buffer), "\n");
    }
    AppendStep(buffer, index, steps_[index]);
  }
  return fmt::to_string(buffer);
}

GeometryResult::GeometryResult(RawImageBatch images)
    : images_(std::move(images)), traces_(images_.size()) {}

GeometryResult::GeometryResult(RawImageBatch images,
                               std::vector<GeometryTrace> traces)
    : images_(std::move(images)), traces_(std::move(traces)) {
  if (traces_.size() != images_.size()) {
    throw std::invalid_argument(
        "GeometryResult image count and trace count mismatch");
  }
}

GeometryResult::GeometryResult(RawImageBatch images,
                               std::vector<GeometryTrace> traces,
                               std::vector<RawImage> keep_alive)
    : images_(std::move(images)),
      traces_(std::move(traces)),
      keep_alive_(std::move(keep_alive)) {
  if (traces_.size() != images_.size()) {
    throw std::invalid_argument(
        "GeometryResult image count and trace count mismatch");
  }
}

bool GeometryResult::empty() const { return images_.empty(); }

std::size_t GeometryResult::size() const { return images_.size(); }

const RawImageBatch& GeometryResult::images() const { return images_; }

const std::vector<GeometryTrace>& GeometryResult::traces() const {
  return traces_;
}

const GeometryTrace& GeometryResult::trace(std::size_t index) const {
  return traces_.at(index);
}

ImageSize GeometryResult::original_size(std::size_t index) const {
  const RawImage& image = images_.image(index);
  const GeometryTrace& geometry_trace = traces_.at(index);
  if (geometry_trace.empty()) {
    return image.size();
  }
  return geometry_trace.step(0).before_size;
}

ImageSize GeometryResult::transformed_size(std::size_t index) const {
  return images_.image(index).size();
}

GeometryTransformer::GeometryTransformer() {
#if defined(MW_INFER_HAS_OPENCV_GEOMETRY_ADAPTER)
  AddAdapter(CreateOpenCvMatGeometryAdapter());
#endif
#if defined(MW_INFER_HAS_OPENCV_CUDA_ADAPTER)
  AddAdapter(CreateOpenCvCudaGeometryAdapter());
#endif
}

void GeometryTransformer::AddAdapter(std::unique_ptr<GeometryAdapter> adapter) {
  if (!adapter) {
    throw std::invalid_argument("Geometry adapter is null");
  }
  adapters_.push_back(std::move(adapter));
}

bool GeometryTransformer::Supports(const RawImage& image) const {
  for (const auto& adapter : adapters_) {
    if (adapter->Supports(image)) {
      return true;
    }
  }
  return false;
}

GeometryResult GeometryTransformer::ResizeImpl(
    GeometryResult result, ImageSize size, Interpolation interpolation,
    ExecutionStream* stream) const {
  ValidateSize(size, "Resize size");
  std::vector<RawImage> output_images;
  output_images.reserve(result.size());
  std::vector<GeometryTrace> traces = result.traces();
  std::vector<RawImage> keep_alive = std::move(result.keep_alive_);
  const std::vector<RawImage>& input_images = result.images().images();
  if (stream != nullptr) {
    keep_alive.insert(keep_alive.end(), input_images.begin(),
                      input_images.end());
  }
  for (std::size_t index = 0; index < input_images.size(); ++index) {
    const RawImage& image = input_images[index];
    const ImageSize before_size = image.size();
    ValidateSize(before_size, "Resize input size");

    const GeometryAdapter& adapter = SelectAdapter(image);
    if (stream == nullptr) {
      output_images.push_back(adapter.Resize(image, size, interpolation));
    } else {
      ValidateStreamDevice(image, *stream, "Resize");
      output_images.push_back(
          adapter.Resize(image, size, *stream, interpolation));
    }
    traces[index].AddResize(before_size, size);
  }
  return GeometryResult(MakeRawImageBatch(std::move(output_images)),
                        std::move(traces), std::move(keep_alive));
}

GeometryResult GeometryTransformer::Resize(GeometryResult result,
                                           ImageSize size,
                                           Interpolation interpolation) const {
  return ResizeImpl(std::move(result), size, interpolation, nullptr);
}

GeometryResult GeometryTransformer::Resize(RawImageBatch images, ImageSize size,
                                           Interpolation interpolation) const {
  return this->Resize(GeometryResult(std::move(images)), size, interpolation);
}

GeometryResult GeometryTransformer::Resize(
    GeometryResult result, ImageSize size, ExecutionStream& stream,
    Interpolation interpolation) const {
  return ResizeImpl(std::move(result), size, interpolation, &stream);
}

GeometryResult GeometryTransformer::Resize(
    RawImageBatch images, ImageSize size, ExecutionStream& stream,
    Interpolation interpolation) const {
  return ResizeImpl(GeometryResult(std::move(images)), size, interpolation,
                    &stream);
}

GeometryResult GeometryTransformer::ResizeShortSideImpl(
    GeometryResult result, int short_side, Interpolation interpolation,
    ExecutionStream* stream) const {
  std::vector<RawImage> output_images;
  output_images.reserve(result.size());
  std::vector<GeometryTrace> traces = result.traces();
  std::vector<RawImage> keep_alive = std::move(result.keep_alive_);
  const std::vector<RawImage>& input_images = result.images().images();
  if (stream != nullptr) {
    keep_alive.insert(keep_alive.end(), input_images.begin(),
                      input_images.end());
  }
  for (std::size_t index = 0; index < input_images.size(); ++index) {
    const RawImage& image = input_images[index];
    const ImageSize before_size = image.size();
    const ImageSize resized_size = ResizeShortSideSize(before_size, short_side);

    const GeometryAdapter& adapter = SelectAdapter(image);
    if (stream == nullptr) {
      output_images.push_back(
          adapter.Resize(image, resized_size, interpolation));
    } else {
      ValidateStreamDevice(image, *stream, "ResizeShortSide");
      output_images.push_back(
          adapter.Resize(image, resized_size, *stream, interpolation));
    }
    traces[index].AddResize(before_size, resized_size);
  }
  return GeometryResult(MakeRawImageBatch(std::move(output_images)),
                        std::move(traces), std::move(keep_alive));
}

GeometryResult GeometryTransformer::ResizeShortSide(
    GeometryResult result, int short_side, Interpolation interpolation) const {
  return ResizeShortSideImpl(std::move(result), short_side, interpolation,
                             nullptr);
}

GeometryResult GeometryTransformer::ResizeShortSide(
    RawImageBatch images, int short_side, Interpolation interpolation) const {
  return this->ResizeShortSide(GeometryResult(std::move(images)), short_side,
                               interpolation);
}

GeometryResult GeometryTransformer::ResizeShortSide(
    GeometryResult result, int short_side, ExecutionStream& stream,
    Interpolation interpolation) const {
  return ResizeShortSideImpl(std::move(result), short_side, interpolation,
                             &stream);
}

GeometryResult GeometryTransformer::ResizeShortSide(
    RawImageBatch images, int short_side, ExecutionStream& stream,
    Interpolation interpolation) const {
  return ResizeShortSideImpl(GeometryResult(std::move(images)), short_side,
                             interpolation, &stream);
}

GeometryResult GeometryTransformer::PadImpl(GeometryResult result,
                                            Padding padding, FillValue value,
                                            ExecutionStream* stream) const {
  ValidatePadding(padding);
  std::vector<RawImage> output_images;
  output_images.reserve(result.size());
  std::vector<GeometryTrace> traces = result.traces();
  std::vector<RawImage> keep_alive = std::move(result.keep_alive_);
  const std::vector<RawImage>& input_images = result.images().images();
  if (stream != nullptr) {
    keep_alive.insert(keep_alive.end(), input_images.begin(),
                      input_images.end());
  }
  for (std::size_t index = 0; index < input_images.size(); ++index) {
    const RawImage& image = input_images[index];
    const ImageSize before_size = image.size();
    ValidateSize(before_size, "Pad input size");

    const GeometryAdapter& adapter = SelectAdapter(image);
    if (stream == nullptr) {
      output_images.push_back(adapter.Pad(image, padding, value));
    } else {
      ValidateStreamDevice(image, *stream, "Pad");
      output_images.push_back(adapter.Pad(image, padding, *stream, value));
    }
    traces[index].AddPad(before_size, padding);
  }
  return GeometryResult(MakeRawImageBatch(std::move(output_images)),
                        std::move(traces), std::move(keep_alive));
}

GeometryResult GeometryTransformer::Pad(GeometryResult result,
                                        Padding padding,
                                        FillValue value) const {
  return PadImpl(std::move(result), padding, std::move(value), nullptr);
}

GeometryResult GeometryTransformer::Pad(RawImageBatch images, Padding padding,
                                        FillValue value) const {
  return this->Pad(GeometryResult(std::move(images)), padding,
                   std::move(value));
}

GeometryResult GeometryTransformer::Pad(GeometryResult result,
                                        Padding padding,
                                        ExecutionStream& stream,
                                        FillValue value) const {
  return PadImpl(std::move(result), padding, std::move(value), &stream);
}

GeometryResult GeometryTransformer::Pad(RawImageBatch images, Padding padding,
                                        ExecutionStream& stream,
                                        FillValue value) const {
  return PadImpl(GeometryResult(std::move(images)), padding, std::move(value),
                 &stream);
}

GeometryResult GeometryTransformer::CropImpl(GeometryResult result, Rect rect,
                                             ExecutionStream* stream) const {
  std::vector<RawImage> output_images;
  output_images.reserve(result.size());
  std::vector<GeometryTrace> traces = result.traces();
  std::vector<RawImage> keep_alive = std::move(result.keep_alive_);
  const std::vector<RawImage>& input_images = result.images().images();
  if (stream != nullptr) {
    keep_alive.insert(keep_alive.end(), input_images.begin(),
                      input_images.end());
  }
  for (std::size_t index = 0; index < input_images.size(); ++index) {
    const RawImage& image = input_images[index];
    const ImageSize before_size = image.size();
    ValidateSize(before_size, "Crop input size");
    ValidateCropRect(before_size, rect);

    const GeometryAdapter& adapter = SelectAdapter(image);
    if (stream == nullptr) {
      output_images.push_back(adapter.Crop(image, rect));
    } else {
      ValidateStreamDevice(image, *stream, "Crop");
      output_images.push_back(adapter.Crop(image, rect, *stream));
    }
    traces[index].AddCrop(before_size, rect);
  }
  return GeometryResult(MakeRawImageBatch(std::move(output_images)),
                        std::move(traces), std::move(keep_alive));
}

GeometryResult GeometryTransformer::Crop(GeometryResult result,
                                         Rect rect) const {
  return CropImpl(std::move(result), rect, nullptr);
}

GeometryResult GeometryTransformer::Crop(RawImageBatch images,
                                         Rect rect) const {
  return this->Crop(GeometryResult(std::move(images)), rect);
}

GeometryResult GeometryTransformer::Crop(GeometryResult result, Rect rect,
                                         ExecutionStream& stream) const {
  return CropImpl(std::move(result), rect, &stream);
}

GeometryResult GeometryTransformer::Crop(RawImageBatch images, Rect rect,
                                         ExecutionStream& stream) const {
  return CropImpl(GeometryResult(std::move(images)), rect, &stream);
}

GeometryResult GeometryTransformer::CenterCropImpl(
    GeometryResult result, ImageSize size, ExecutionStream* stream) const {
  ValidateSize(size, "CenterCrop size");
  std::vector<RawImage> output_images;
  output_images.reserve(result.size());
  std::vector<GeometryTrace> traces = result.traces();
  std::vector<RawImage> keep_alive = std::move(result.keep_alive_);
  const std::vector<RawImage>& input_images = result.images().images();
  if (stream != nullptr) {
    keep_alive.insert(keep_alive.end(), input_images.begin(),
                      input_images.end());
  }
  for (std::size_t index = 0; index < input_images.size(); ++index) {
    const RawImage& image = input_images[index];
    const ImageSize before_size = image.size();
    ValidateSize(before_size, "CenterCrop input size");

    const Rect crop_rect = CenterCropRect(before_size, size);
    const GeometryAdapter& adapter = SelectAdapter(image);
    if (stream == nullptr) {
      output_images.push_back(adapter.Crop(image, crop_rect));
    } else {
      ValidateStreamDevice(image, *stream, "CenterCrop");
      output_images.push_back(adapter.Crop(image, crop_rect, *stream));
    }
    traces[index].AddCrop(before_size, crop_rect);
  }
  return GeometryResult(MakeRawImageBatch(std::move(output_images)),
                        std::move(traces), std::move(keep_alive));
}

GeometryResult GeometryTransformer::CenterCrop(GeometryResult result,
                                               ImageSize size) const {
  return CenterCropImpl(std::move(result), size, nullptr);
}

GeometryResult GeometryTransformer::CenterCrop(RawImageBatch images,
                                               ImageSize size) const {
  return this->CenterCrop(GeometryResult(std::move(images)), size);
}

GeometryResult GeometryTransformer::CenterCrop(GeometryResult result,
                                               ImageSize size,
                                               ExecutionStream& stream) const {
  return CenterCropImpl(std::move(result), size, &stream);
}

GeometryResult GeometryTransformer::CenterCrop(RawImageBatch images,
                                               ImageSize size,
                                               ExecutionStream& stream) const {
  return CenterCropImpl(GeometryResult(std::move(images)), size, &stream);
}

GeometryResult GeometryTransformer::LetterBoxImpl(
    GeometryResult result, ImageSize size, Interpolation interpolation,
    FillValue value, ExecutionStream* stream) const {
  ValidateSize(size, "LetterBox size");
  std::vector<RawImage> output_images;
  output_images.reserve(result.size());
  std::vector<GeometryTrace> traces = result.traces();
  std::vector<RawImage> keep_alive = std::move(result.keep_alive_);
  const std::vector<RawImage>& input_images = result.images().images();
  if (stream != nullptr) {
    keep_alive.insert(keep_alive.end(), input_images.begin(),
                      input_images.end());
    keep_alive.reserve(keep_alive.size() + input_images.size());
  }
  for (std::size_t index = 0; index < input_images.size(); ++index) {
    const RawImage& image = input_images[index];
    const ImageSize before_size = image.size();
    ValidateSize(before_size, "LetterBox input size");

    const ImageSize resized_size = ResizeKeepRatioSize(before_size, size);
    const Padding padding = CenterPadding(resized_size, size);
    const GeometryAdapter& adapter = SelectAdapter(image);

    RawImage resized;
    if (stream == nullptr) {
      resized = adapter.Resize(image, resized_size, interpolation);
    } else {
      ValidateStreamDevice(image, *stream, "LetterBox");
      resized = adapter.Resize(image, resized_size, *stream, interpolation);
    }
    const GeometryAdapter& pad_adapter = SelectAdapter(resized);
    if (stream == nullptr) {
      output_images.push_back(pad_adapter.Pad(resized, padding, value));
    } else {
      output_images.push_back(
          pad_adapter.Pad(resized, padding, *stream, value));
      keep_alive.push_back(resized);
    }
    traces[index].AddLetterBox(before_size, size, resized_size, padding);
  }
  return GeometryResult(MakeRawImageBatch(std::move(output_images)),
                        std::move(traces), std::move(keep_alive));
}

GeometryResult GeometryTransformer::LetterBox(GeometryResult result,
                                              ImageSize size,
                                              Interpolation interpolation,
                                              FillValue value) const {
  return LetterBoxImpl(std::move(result), size, interpolation,
                       std::move(value), nullptr);
}

GeometryResult GeometryTransformer::LetterBox(RawImageBatch images,
                                              ImageSize size,
                                              Interpolation interpolation,
                                              FillValue value) const {
  return this->LetterBox(GeometryResult(std::move(images)), size, interpolation,
                         std::move(value));
}

GeometryResult GeometryTransformer::LetterBox(
    GeometryResult result, ImageSize size, ExecutionStream& stream,
    Interpolation interpolation, FillValue value) const {
  return LetterBoxImpl(std::move(result), size, interpolation,
                       std::move(value), &stream);
}

GeometryResult GeometryTransformer::LetterBox(
    RawImageBatch images, ImageSize size, ExecutionStream& stream,
    Interpolation interpolation, FillValue value) const {
  return LetterBoxImpl(GeometryResult(std::move(images)), size, interpolation,
                       std::move(value), &stream);
}

const GeometryAdapter& GeometryTransformer::SelectAdapter(
    const RawImage& image) const {
  for (const auto& adapter : adapters_) {
    if (adapter->Supports(image)) {
      return *adapter;
    }
  }
  throw std::invalid_argument("No geometry adapter supports this RawImage");
}

}  // namespace mw::infer
