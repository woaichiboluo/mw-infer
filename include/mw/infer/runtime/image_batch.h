#ifndef MW_INFER_RUNTIME_IMAGE_BATCH_H_
#define MW_INFER_RUNTIME_IMAGE_BATCH_H_

#include <algorithm>
#include <cstddef>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mw/infer/common/geometry.h"
#include "mw/infer/runtime/input.h"

namespace mw::infer {

enum class GeometryStepKind {
  kResize,
  kPad,
  kCrop,
  kLetterBox,
};

struct ResizeStep {
  float scale_x = 1.0F;
  float scale_y = 1.0F;
};

struct PadStep {
  Padding padding;
};

struct CropStep {
  Rect crop_rect;
};

struct LetterBoxStep {
  ImageSize resized_size;
  float scale_x = 1.0F;
  float scale_y = 1.0F;
  Padding padding;
};

struct GeometryStep {
  GeometryStepKind kind = GeometryStepKind::kResize;
  ImageSize before_size;
  ImageSize after_size;
  ResizeStep resize;
  PadStep pad;
  CropStep crop;
  LetterBoxStep letterbox;
};

class GeometryTrace {
 public:
  bool empty() const { return steps_.empty(); }
  std::size_t size() const { return steps_.size(); }
  const std::vector<GeometryStep>& steps() const { return steps_; }
  const GeometryStep& step(std::size_t index) const { return steps_.at(index); }

  void AddResize(ImageSize before_size, ImageSize after_size) {
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

  void AddPad(ImageSize before_size, Padding padding) {
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

  void AddCrop(ImageSize before_size, Rect crop_rect) {
    ValidateSize(before_size, "Crop before size");
    ValidateCropRect(before_size, crop_rect);

    GeometryStep step;
    step.kind = GeometryStepKind::kCrop;
    step.before_size = before_size;
    step.after_size = ImageSize{crop_rect.width, crop_rect.height};
    step.crop.crop_rect = crop_rect;
    steps_.push_back(step);
  }

  void AddLetterBox(ImageSize before_size, ImageSize after_size,
                    ImageSize resized_size, Padding padding) {
    ValidateSize(before_size, "LetterBox before size");
    ValidateSize(after_size, "LetterBox after size");
    ValidateSize(resized_size, "LetterBox resized size");
    ValidatePadding(padding);

    if (resized_size.width + padding.left + padding.right != after_size.width ||
        resized_size.height + padding.top + padding.bottom !=
            after_size.height) {
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

  Point2f RestorePoint(Point2f point) const {
    for (auto it = steps_.rbegin(); it != steps_.rend(); ++it) {
      point = RestorePointByStep(point, *it);
    }
    return point;
  }

  Rect2f RestoreRect(Rect2f rect) const {
    const Point2f top_left = RestorePoint(Point2f{rect.x, rect.y});
    const Point2f bottom_right =
        RestorePoint(Point2f{rect.x + rect.width, rect.y + rect.height});
    const float left = std::min(top_left.x, bottom_right.x);
    const float top = std::min(top_left.y, bottom_right.y);
    const float right = std::max(top_left.x, bottom_right.x);
    const float bottom = std::max(top_left.y, bottom_right.y);
    return Rect2f{left, top, right - left, bottom - top};
  }

  std::vector<Point2f> RestorePolygon(
      const std::vector<Point2f>& polygon) const {
    std::vector<Point2f> restored;
    restored.reserve(polygon.size());
    for (Point2f point : polygon) {
      restored.push_back(RestorePoint(point));
    }
    return restored;
  }

  std::string Dump() const {
    std::ostringstream oss;
    for (std::size_t index = 0; index < steps_.size(); ++index) {
      if (index > 0) {
        oss << '\n';
      }
      DumpStep(oss, index, steps_[index]);
    }
    return oss.str();
  }

 private:
  static void DumpSize(std::ostream& os, ImageSize size) {
    os << size.width << "x" << size.height;
  }

  static void DumpPadding(std::ostream& os, Padding padding) {
    os << "{left=" << padding.left << ",top=" << padding.top
       << ",right=" << padding.right << ",bottom=" << padding.bottom << "}";
  }

  static void DumpRect(std::ostream& os, Rect rect) {
    os << "{x=" << rect.x << ",y=" << rect.y << ",width=" << rect.width
       << ",height=" << rect.height << "}";
  }

  static void DumpStep(std::ostream& os, std::size_t index,
                       const GeometryStep& step) {
    os << "#" << index << " ";
    switch (step.kind) {
      case GeometryStepKind::kResize:
        os << "Resize";
        break;
      case GeometryStepKind::kPad:
        os << "Pad";
        break;
      case GeometryStepKind::kCrop:
        os << "Crop";
        break;
      case GeometryStepKind::kLetterBox:
        os << "LetterBox";
        break;
    }

    os << " before=";
    DumpSize(os, step.before_size);
    os << " after=";
    DumpSize(os, step.after_size);

    switch (step.kind) {
      case GeometryStepKind::kResize:
        os << " scale_x=" << step.resize.scale_x
           << " scale_y=" << step.resize.scale_y;
        break;
      case GeometryStepKind::kPad:
        os << " padding=";
        DumpPadding(os, step.pad.padding);
        break;
      case GeometryStepKind::kCrop:
        os << " rect=";
        DumpRect(os, step.crop.crop_rect);
        break;
      case GeometryStepKind::kLetterBox:
        os << " resized=";
        DumpSize(os, step.letterbox.resized_size);
        os << " scale_x=" << step.letterbox.scale_x
           << " scale_y=" << step.letterbox.scale_y << " padding=";
        DumpPadding(os, step.letterbox.padding);
        break;
    }
  }

  static void ValidateSize(ImageSize size, const char* name) {
    if (size.width <= 0 || size.height <= 0) {
      throw std::invalid_argument(std::string(name) + " must be positive");
    }
  }

  static void ValidatePadding(Padding padding) {
    if (padding.left < 0 || padding.top < 0 || padding.right < 0 ||
        padding.bottom < 0) {
      throw std::invalid_argument("Padding values must be non-negative");
    }
  }

  static void ValidateCropRect(ImageSize size, Rect rect) {
    if (rect.width <= 0 || rect.height <= 0) {
      throw std::invalid_argument("Crop rect size must be positive");
    }
    if (rect.x < 0 || rect.y < 0 || rect.x + rect.width > size.width ||
        rect.y + rect.height > size.height) {
      throw std::invalid_argument("Crop rect is outside image bounds");
    }
  }

  static Point2f RestorePointByStep(Point2f point, const GeometryStep& step) {
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

  std::vector<GeometryStep> steps_;
};

struct ImageFrame {
  RawImage image;
  ImageDesc original_desc;
  GeometryTrace geometry_trace;
};

class ImageBatch {
 public:
  ImageBatch() = default;

  explicit ImageBatch(RawImageBatch images) {
    frames_.reserve(images.size());
    for (const RawImage& image : images.images()) {
      frames_.push_back(ImageFrame{image, image.desc(), GeometryTrace{}});
    }
  }

  explicit ImageBatch(std::vector<ImageFrame> frames)
      : frames_(std::move(frames)) {}

  bool empty() const { return frames_.empty(); }
  std::size_t size() const { return frames_.size(); }
  const std::vector<ImageFrame>& frames() const { return frames_; }
  std::vector<ImageFrame>& mutable_frames() { return frames_; }
  const ImageFrame& frame(std::size_t index) const { return frames_.at(index); }
  ImageFrame& mutable_frame(std::size_t index) { return frames_.at(index); }

  RawImageBatch ToRawImageBatch() const {
    std::vector<RawImage> images;
    images.reserve(frames_.size());
    for (const ImageFrame& frame : frames_) {
      images.push_back(frame.image);
    }
    return RawImageBatch(std::move(images));
  }

 private:
  std::vector<ImageFrame> frames_;
};

}  // namespace mw::infer

#endif  // MW_INFER_RUNTIME_IMAGE_BATCH_H_
