#ifndef MW_INFER_GEOMETRY_H_
#define MW_INFER_GEOMETRY_H_

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mw::infer {

struct ImageSize {
  int32_t width = 0;
  int32_t height = 0;
};

struct RectF {
  float x = 0.0F;
  float y = 0.0F;
  float width = 0.0F;
  float height = 0.0F;
};

struct PointF {
  float x = 0.0F;
  float y = 0.0F;
};

struct AffineTransform {
  float a = 1.0F;
  float b = 0.0F;
  float c = 0.0F;
  float d = 0.0F;
  float e = 1.0F;
  float f = 0.0F;
};

enum class GeometryUpdateStepKind : int32_t {
  kResize,
  kCrop,
  kPad,
};

struct GeometryUpdateStep {
  GeometryUpdateStepKind kind = GeometryUpdateStepKind::kResize;
  std::vector<ImageSize> sizes;
  std::vector<RectF> crops;
  std::vector<PointF> offsets;
};

struct GeometryUpdate {
  std::vector<ImageSize> source_sizes;
  std::vector<GeometryUpdateStep> steps;

  static GeometryUpdate FromSource(std::vector<ImageSize> sizes) {
    GeometryUpdate update;
    update.source_sizes = std::move(sizes);
    return update;
  }

  GeometryUpdate& ThenResize(std::vector<ImageSize> sizes) {
    GeometryUpdateStep step;
    step.kind = GeometryUpdateStepKind::kResize;
    step.sizes = std::move(sizes);
    steps.push_back(std::move(step));
    return *this;
  }

  GeometryUpdate& ThenCrop(std::vector<RectF> crops) {
    GeometryUpdateStep step;
    step.kind = GeometryUpdateStepKind::kCrop;
    step.crops = std::move(crops);
    steps.push_back(std::move(step));
    return *this;
  }

  GeometryUpdate& ThenPad(std::vector<ImageSize> sizes,
                          std::vector<PointF> offsets) {
    GeometryUpdateStep step;
    step.kind = GeometryUpdateStepKind::kPad;
    step.sizes = std::move(sizes);
    step.offsets = std::move(offsets);
    steps.push_back(std::move(step));
    return *this;
  }
};

class ImageGeometry {
 public:
  ImageGeometry() = default;
  explicit ImageGeometry(ImageSize size)
      : original_size_(size), current_size_(size) {}

  ImageSize original_size() const { return original_size_; }
  ImageSize current_size() const { return current_size_; }

  PointF MapPointToOriginal(PointF point) const;
  RectF MapRectToOriginal(const RectF& rect) const;
  PointF MapPointFromOriginal(PointF point) const;
  RectF MapRectFromOriginal(const RectF& rect) const;

 private:
  friend ImageGeometry MakeImageGeometry(ImageSize size);
  friend ImageGeometry ApplyImageTransform(
      ImageGeometry geometry, ImageSize new_size,
      const AffineTransform& previous_to_current);

  ImageSize original_size_;
  ImageSize current_size_;
  AffineTransform current_to_original_;
};

inline float Area(const RectF& rect) {
  return std::max(rect.width, 0.0F) * std::max(rect.height, 0.0F);
}

inline RectF Expand(const RectF& rect, float ratio) {
  const float normalized_ratio = std::max(ratio, 0.0F);
  const float center_x = rect.x + rect.width * 0.5F;
  const float center_y = rect.y + rect.height * 0.5F;
  const float width = rect.width * normalized_ratio;
  const float height = rect.height * normalized_ratio;
  return RectF{center_x - width * 0.5F, center_y - height * 0.5F, width,
               height};
}

inline RectF Clamp(const RectF& rect, const ImageSize& size) {
  const float image_width = static_cast<float>(std::max(size.width, 0));
  const float image_height = static_cast<float>(std::max(size.height, 0));
  const float x1 = std::clamp(rect.x, 0.0F, image_width);
  const float y1 = std::clamp(rect.y, 0.0F, image_height);
  const float x2 = std::clamp(rect.x + rect.width, 0.0F, image_width);
  const float y2 = std::clamp(rect.y + rect.height, 0.0F, image_height);
  return RectF{x1, y1, std::max(x2 - x1, 0.0F), std::max(y2 - y1, 0.0F)};
}

inline AffineTransform IdentityAffine() { return AffineTransform{}; }

inline AffineTransform ScaleAffine(float scale_x, float scale_y) {
  return AffineTransform{scale_x, 0.0F, 0.0F, 0.0F, scale_y, 0.0F};
}

inline AffineTransform TranslateAffine(float x, float y) {
  return AffineTransform{1.0F, 0.0F, x, 0.0F, 1.0F, y};
}

inline PointF MapPoint(const AffineTransform& transform, PointF point) {
  return PointF{transform.a * point.x + transform.b * point.y + transform.c,
                transform.d * point.x + transform.e * point.y + transform.f};
}

inline RectF MapRect(const AffineTransform& transform, const RectF& rect) {
  const PointF p0 = MapPoint(transform, PointF{rect.x, rect.y});
  const PointF p1 = MapPoint(transform, PointF{rect.x + rect.width, rect.y});
  const PointF p2 = MapPoint(transform, PointF{rect.x, rect.y + rect.height});
  const PointF p3 =
      MapPoint(transform, PointF{rect.x + rect.width, rect.y + rect.height});

  const float x1 = std::min({p0.x, p1.x, p2.x, p3.x});
  const float y1 = std::min({p0.y, p1.y, p2.y, p3.y});
  const float x2 = std::max({p0.x, p1.x, p2.x, p3.x});
  const float y2 = std::max({p0.y, p1.y, p2.y, p3.y});
  return RectF{x1, y1, x2 - x1, y2 - y1};
}

inline AffineTransform ComposeAffine(const AffineTransform& lhs,
                                     const AffineTransform& rhs) {
  return AffineTransform{
      lhs.a * rhs.a + lhs.b * rhs.d,
      lhs.a * rhs.b + lhs.b * rhs.e,
      lhs.a * rhs.c + lhs.b * rhs.f + lhs.c,
      lhs.d * rhs.a + lhs.e * rhs.d,
      lhs.d * rhs.b + lhs.e * rhs.e,
      lhs.d * rhs.c + lhs.e * rhs.f + lhs.f,
  };
}

inline AffineTransform InvertAffine(const AffineTransform& transform) {
  const float determinant =
      transform.a * transform.e - transform.b * transform.d;
  if (determinant == 0.0F) {
    throw std::invalid_argument("affine transform is not invertible");
  }

  const float inv_det = 1.0F / determinant;
  const float a = transform.e * inv_det;
  const float b = -transform.b * inv_det;
  const float d = -transform.d * inv_det;
  const float e = transform.a * inv_det;
  return AffineTransform{
      a, b, -(a * transform.c + b * transform.f),
      d, e, -(d * transform.c + e * transform.f),
  };
}

inline PointF ImageGeometry::MapPointToOriginal(PointF point) const {
  return MapPoint(current_to_original_, point);
}

inline RectF ImageGeometry::MapRectToOriginal(const RectF& rect) const {
  return MapRect(current_to_original_, rect);
}

inline PointF ImageGeometry::MapPointFromOriginal(PointF point) const {
  return MapPoint(InvertAffine(current_to_original_), point);
}

inline RectF ImageGeometry::MapRectFromOriginal(const RectF& rect) const {
  return MapRect(InvertAffine(current_to_original_), rect);
}

inline ImageGeometry MakeImageGeometry(ImageSize size) {
  return ImageGeometry(size);
}

inline ImageGeometry ApplyImageTransform(
    ImageGeometry geometry, ImageSize new_size,
    const AffineTransform& previous_to_current) {
  geometry.current_size_ = new_size;
  geometry.current_to_original_ = ComposeAffine(
      geometry.current_to_original_, InvertAffine(previous_to_current));
  return geometry;
}

}  // namespace mw::infer

#endif  // MW_INFER_GEOMETRY_H_
