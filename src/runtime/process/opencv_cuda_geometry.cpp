#include <algorithm>
#include <cstddef>
#include <memory>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

#include "mw/infer/runtime/input/opencv_cuda_input.h"
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

cv::cuda::GpuMat ResizeGpuMat(const cv::cuda::GpuMat& source, ImageSize size,
                              Interpolation interpolation) {
  const int channels = source.channels();
  if (channels > 4) {
    throw std::invalid_argument("OpenCV CUDA resize supports up to 4 channels");
  }
  if (channels != 2) {
    cv::cuda::GpuMat output;
    cv::cuda::resize(source, output, ToOpenCvSize(size), 0.0, 0.0,
                     ToOpenCvInterpolation(interpolation));
    return output;
  }

  std::vector<cv::cuda::GpuMat> channel_mats;
  cv::cuda::split(source, channel_mats);
  for (cv::cuda::GpuMat& channel : channel_mats) {
    cv::cuda::GpuMat resized_channel;
    cv::cuda::resize(channel, resized_channel, ToOpenCvSize(size), 0.0, 0.0,
                     ToOpenCvInterpolation(interpolation));
    channel = std::move(resized_channel);
  }

  cv::cuda::GpuMat output;
  cv::cuda::merge(channel_mats, output);
  return output;
}

cv::cuda::GpuMat CopyMakeBorderGpuMat(const cv::cuda::GpuMat& source,
                                      Padding padding,
                                      const FillValue& fill_value) {
  const int channels = source.channels();
  if (channels > 4) {
    throw std::invalid_argument(
        "OpenCV CUDA copyMakeBorder supports up to 4 channels");
  }

  const cv::Scalar value = ToOpenCvScalar(fill_value);
  if (channels != 2) {
    cv::cuda::GpuMat output;
    cv::cuda::copyMakeBorder(source, output, padding.top, padding.bottom,
                             padding.left, padding.right, cv::BORDER_CONSTANT,
                             value);
    return output;
  }

  std::vector<cv::cuda::GpuMat> channel_mats;
  cv::cuda::split(source, channel_mats);
  for (std::size_t index = 0; index < channel_mats.size(); ++index) {
    cv::cuda::GpuMat padded_channel;
    cv::cuda::copyMakeBorder(channel_mats[index], padded_channel, padding.top,
                             padding.bottom, padding.left, padding.right,
                             cv::BORDER_CONSTANT,
                             cv::Scalar(value[static_cast<int>(index)]));
    channel_mats[index] = std::move(padded_channel);
  }

  cv::cuda::GpuMat output;
  cv::cuda::merge(channel_mats, output);
  return output;
}

class OpenCvCudaGeometryAdapter final : public GeometryAdapter {
 public:
  bool Supports(const RawImage& image) const override {
    return image.handle_kind() == ImageHandleKind::kOpenCvCudaGpuMat;
  }

  RawImage Resize(const RawImage& image, ImageSize size,
                  Interpolation interpolation) const override {
    return ToRawImage(
        ResizeGpuMat(GetOpenCvCudaGpuMat(image), size, interpolation));
  }

  RawImage Pad(const RawImage& image, Padding padding,
               const FillValue& value) const override {
    return ToRawImage(
        CopyMakeBorderGpuMat(GetOpenCvCudaGpuMat(image), padding, value));
  }

  RawImage Crop(const RawImage& image, Rect rect) const override {
    const cv::cuda::GpuMat& source = GetOpenCvCudaGpuMat(image);
    return ToRawImage(source(ToOpenCvRect(rect)).clone());
  }
};

}  // namespace

std::unique_ptr<GeometryAdapter> CreateOpenCvCudaGeometryAdapter() {
  return std::make_unique<OpenCvCudaGeometryAdapter>();
}

}  // namespace mw::infer
