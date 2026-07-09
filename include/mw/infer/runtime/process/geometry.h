#ifndef MW_INFER_RUNTIME_PROCESS_GEOMETRY_H_
#define MW_INFER_RUNTIME_PROCESS_GEOMETRY_H_

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "mw/infer/runtime/input/input.h"

namespace mw::infer {

struct Point2f {
  float x = 0.0F;
  float y = 0.0F;
};

struct Rect {
  int x = 0;
  int y = 0;
  int width = 0;
  int height = 0;
};

struct Rect2f {
  float x = 0.0F;
  float y = 0.0F;
  float width = 0.0F;
  float height = 0.0F;
};

struct Padding {
  int left = 0;
  int top = 0;
  int right = 0;
  int bottom = 0;
};

enum class Interpolation {
  kNearest,
  kLinear,
  kCubic,
  kArea,
};

struct FillValue {
  std::vector<double> channels;
};

ImageSize ResizeShortSideSize(ImageSize input_size, int short_side);

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
  bool empty() const;
  std::size_t size() const;
  const std::vector<GeometryStep>& steps() const;
  const GeometryStep& step(std::size_t index) const;

  void AddResize(ImageSize before_size, ImageSize after_size);
  void AddPad(ImageSize before_size, Padding padding);
  void AddCrop(ImageSize before_size, Rect crop_rect);
  void AddLetterBox(ImageSize before_size, ImageSize after_size,
                    ImageSize resized_size, Padding padding);

  Point2f RestorePoint(Point2f point) const;
  Rect2f RestoreRect(Rect2f rect) const;
  std::vector<Point2f> RestorePolygon(
      const std::vector<Point2f>& polygon) const;
  std::string Dump() const;

 private:
  std::vector<GeometryStep> steps_;
};

class GeometryResult {
 public:
  GeometryResult() = default;
  explicit GeometryResult(RawImageBatch images);
  GeometryResult(RawImageBatch images, std::vector<GeometryTrace> traces);

  bool empty() const;
  std::size_t size() const;
  const RawImageBatch& images() const;
  const std::vector<GeometryTrace>& traces() const;
  const GeometryTrace& trace(std::size_t index) const;
  ImageSize original_size(std::size_t index) const;
  ImageSize transformed_size(std::size_t index) const;

 private:
  RawImageBatch images_;
  std::vector<GeometryTrace> traces_;
};

class GeometryAdapter {
 public:
  virtual ~GeometryAdapter() = default;

  virtual bool Supports(const RawImage& image) const = 0;
  virtual RawImage Resize(const RawImage& image, ImageSize size,
                          Interpolation interpolation) const = 0;
  virtual RawImage Pad(const RawImage& image, Padding padding,
                       const FillValue& value) const = 0;
  virtual RawImage Crop(const RawImage& image, Rect rect) const = 0;
};

class GeometryTransformer {
 public:
  GeometryTransformer();
  GeometryTransformer(const GeometryTransformer&) = delete;
  GeometryTransformer& operator=(const GeometryTransformer&) = delete;
  GeometryTransformer(GeometryTransformer&&) noexcept = default;
  GeometryTransformer& operator=(GeometryTransformer&&) noexcept = default;

  bool Supports(const RawImage& image) const;

  GeometryResult Resize(
      GeometryResult result, ImageSize size,
      Interpolation interpolation = Interpolation::kLinear) const;
  GeometryResult Resize(
      RawImageBatch images, ImageSize size,
      Interpolation interpolation = Interpolation::kLinear) const;
  GeometryResult ResizeShortSide(
      GeometryResult result, int short_side,
      Interpolation interpolation = Interpolation::kLinear) const;
  GeometryResult ResizeShortSide(
      RawImageBatch images, int short_side,
      Interpolation interpolation = Interpolation::kLinear) const;
  GeometryResult Pad(GeometryResult result, Padding padding,
                     FillValue value = {}) const;
  GeometryResult Pad(RawImageBatch images, Padding padding,
                     FillValue value = {}) const;
  GeometryResult Crop(GeometryResult result, Rect rect) const;
  GeometryResult Crop(RawImageBatch images, Rect rect) const;
  GeometryResult CenterCrop(GeometryResult result, ImageSize size) const;
  GeometryResult CenterCrop(RawImageBatch images, ImageSize size) const;
  GeometryResult LetterBox(GeometryResult result, ImageSize size,
                           Interpolation interpolation = Interpolation::kLinear,
                           FillValue value = {}) const;
  GeometryResult LetterBox(RawImageBatch images, ImageSize size,
                           Interpolation interpolation = Interpolation::kLinear,
                           FillValue value = {}) const;

 private:
  void AddAdapter(std::unique_ptr<GeometryAdapter> adapter);
  const GeometryAdapter& SelectAdapter(const RawImage& image) const;

  std::vector<std::unique_ptr<GeometryAdapter>> adapters_;
};

}  // namespace mw::infer

#endif  // MW_INFER_RUNTIME_PROCESS_GEOMETRY_H_
