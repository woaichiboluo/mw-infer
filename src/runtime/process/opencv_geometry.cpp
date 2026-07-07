#include <algorithm>
#include <memory>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <utility>

#include "mw/infer/runtime/input/opencv_input.h"
#include "mw/infer/runtime/process/geometry.h"

namespace mw::infer {
namespace {

int ToOpenCvInterpolation(Interpolation interpolation) {
  switch (interpolation) {
    case Interpolation::kNearest:
      return cv::INTER_NEAREST;
    case Interpolation::kLinear:
      return cv::INTER_LINEAR;
    case Interpolation::kCubic:
      return cv::INTER_CUBIC;
    case Interpolation::kArea:
      return cv::INTER_AREA;
  }
  throw std::invalid_argument("Unsupported interpolation mode");
}

cv::Size ToOpenCvSize(ImageSize size) {
  return cv::Size(size.width, size.height);
}

cv::Rect ToOpenCvRect(Rect rect) {
  return cv::Rect(rect.x, rect.y, rect.width, rect.height);
}

cv::Scalar ToOpenCvScalar(const FillValue& value) {
  cv::Scalar scalar;
  const std::size_t count = std::min<std::size_t>(value.channels.size(), 4);
  for (std::size_t index = 0; index < count; ++index) {
    scalar[static_cast<int>(index)] = value.channels[index];
  }
  return scalar;
}

class OpenCvMatGeometryAdapter final : public GeometryAdapter {
 public:
  bool Supports(const RawImage& image) const override {
    return image.handle_kind() == ImageHandleKind::kOpenCvMat;
  }

  RawImage Resize(const RawImage& image, ImageSize size,
                  Interpolation interpolation) const override {
    const cv::Mat& source = GetOpenCvMat(image);
    cv::Mat output;
    cv::resize(source, output, ToOpenCvSize(size), 0.0, 0.0,
               ToOpenCvInterpolation(interpolation));
    return ToRawImage(std::move(output));
  }

  RawImage Pad(const RawImage& image, Padding padding,
               const FillValue& value) const override {
    const cv::Mat& source = GetOpenCvMat(image);
    cv::Mat output;
    cv::copyMakeBorder(source, output, padding.top, padding.bottom,
                       padding.left, padding.right, cv::BORDER_CONSTANT,
                       ToOpenCvScalar(value));
    return ToRawImage(std::move(output));
  }

  RawImage Crop(const RawImage& image, Rect rect) const override {
    const cv::Mat& source = GetOpenCvMat(image);
    return ToRawImage(source(ToOpenCvRect(rect)).clone());
  }
};

}  // namespace

std::unique_ptr<GeometryAdapter> CreateOpenCvMatGeometryAdapter() {
  return std::make_unique<OpenCvMatGeometryAdapter>();
}

}  // namespace mw::infer
