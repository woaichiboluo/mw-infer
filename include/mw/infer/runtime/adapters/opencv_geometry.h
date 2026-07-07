#ifndef MW_INFER_RUNTIME_ADAPTERS_OPENCV_GEOMETRY_H_
#define MW_INFER_RUNTIME_ADAPTERS_OPENCV_GEOMETRY_H_

#include <algorithm>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#ifdef MW_INFER_HAS_OPENCV_CUDA_ADAPTER
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#endif

#include "mw/infer/runtime/adapters/opencv_input.h"
#ifdef MW_INFER_HAS_OPENCV_CUDA_ADAPTER
#include "mw/infer/runtime/adapters/opencv_cuda_input.h"
#endif
#include "mw/infer/runtime/image_batch.h"

namespace mw::infer {

namespace detail {

inline void ValidateTargetSize(ImageSize size, const char* name) {
  if (size.width <= 0 || size.height <= 0) {
    throw std::invalid_argument(std::string(name) + " must be positive");
  }
}

inline void ValidatePadding(Padding padding) {
  if (padding.left < 0 || padding.top < 0 || padding.right < 0 ||
      padding.bottom < 0) {
    throw std::invalid_argument("Padding values must be non-negative");
  }
}

inline void ValidateCropRect(ImageSize image_size, Rect rect) {
  if (rect.width <= 0 || rect.height <= 0) {
    throw std::invalid_argument("Crop rect size must be positive");
  }
  if (rect.x < 0 || rect.y < 0 || rect.x + rect.width > image_size.width ||
      rect.y + rect.height > image_size.height) {
    throw std::invalid_argument("Crop rect is outside image bounds");
  }
}

inline cv::Size ToCvSize(ImageSize size) {
  return cv::Size(size.width, size.height);
}

inline cv::Rect ToCvRect(Rect rect) {
  return cv::Rect(rect.x, rect.y, rect.width, rect.height);
}

#ifdef MW_INFER_HAS_OPENCV_CUDA_ADAPTER
inline cv::cuda::GpuMat ResizeGpuMat(const cv::cuda::GpuMat& source,
                                     ImageSize size, int interpolation) {
  const int channels = source.channels();
  if (channels > 4) {
    throw std::invalid_argument("OpenCV CUDA resize supports up to 4 channels");
  }
  if (channels != 2) {
    cv::cuda::GpuMat output;
    cv::cuda::resize(source, output, ToCvSize(size), 0.0, 0.0, interpolation);
    return output;
  }

  std::vector<cv::cuda::GpuMat> channel_mats;
  cv::cuda::split(source, channel_mats);
  for (cv::cuda::GpuMat& channel : channel_mats) {
    cv::cuda::GpuMat resized_channel;
    cv::cuda::resize(channel, resized_channel, ToCvSize(size), 0.0, 0.0,
                     interpolation);
    channel = std::move(resized_channel);
  }

  cv::cuda::GpuMat output;
  cv::cuda::merge(channel_mats, output);
  return output;
}

inline cv::cuda::GpuMat CopyMakeBorderGpuMat(const cv::cuda::GpuMat& source,
                                             Padding padding,
                                             const cv::Scalar& value) {
  const int channels = source.channels();
  if (channels > 4) {
    throw std::invalid_argument(
        "OpenCV CUDA copyMakeBorder supports up to 4 channels");
  }
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
                             cv::BORDER_CONSTANT, cv::Scalar(value[index]));
    channel_mats[index] = std::move(padded_channel);
  }

  cv::cuda::GpuMat output;
  cv::cuda::merge(channel_mats, output);
  return output;
}
#endif

inline ImageSize ResizeKeepRatioSize(ImageSize input_size,
                                     ImageSize target_size) {
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

inline Padding CenterPadding(ImageSize resized_size, ImageSize target_size) {
  const int pad_width = target_size.width - resized_size.width;
  const int pad_height = target_size.height - resized_size.height;
  if (pad_width < 0 || pad_height < 0) {
    throw std::invalid_argument("Resized image is larger than target size");
  }
  return Padding{pad_width / 2, pad_height / 2, pad_width - pad_width / 2,
                 pad_height - pad_height / 2};
}

inline Rect CenterCropRect(ImageSize image_size, ImageSize crop_size) {
  ValidateTargetSize(crop_size, "CenterCrop size");
  if (crop_size.width > image_size.width ||
      crop_size.height > image_size.height) {
    throw std::invalid_argument("CenterCrop size is larger than image size");
  }
  return Rect{(image_size.width - crop_size.width) / 2,
              (image_size.height - crop_size.height) / 2, crop_size.width,
              crop_size.height};
}

inline void ResizeFrame(ImageFrame& frame, ImageSize size, int interpolation) {
  const ImageSize before_size = frame.image.size();
  ValidateTargetSize(before_size, "Resize input size");

  if (frame.image.handle_kind() == ImageHandleKind::kOpenCvMat) {
    const cv::Mat& source = GetOpenCvMat(frame.image);
    cv::Mat output;
    cv::resize(source, output, ToCvSize(size), 0.0, 0.0, interpolation);
    frame.geometry_trace.AddResize(before_size, size);
    frame.image = ToRawImage(std::move(output));
    return;
  }

#ifdef MW_INFER_HAS_OPENCV_CUDA_ADAPTER
  if (frame.image.handle_kind() == ImageHandleKind::kOpenCvCudaGpuMat) {
    const cv::cuda::GpuMat& source = GetOpenCvCudaGpuMat(frame.image);
    cv::cuda::GpuMat output = ResizeGpuMat(source, size, interpolation);
    frame.geometry_trace.AddResize(before_size, size);
    frame.image = ToRawImage(std::move(output));
    return;
  }
#endif

  throw std::invalid_argument("Resize only supports OpenCV Mat and GpuMat");
}

inline void PadFrame(ImageFrame& frame, Padding padding,
                     const cv::Scalar& value) {
  const ImageSize before_size = frame.image.size();
  ValidateTargetSize(before_size, "Pad input size");

  if (frame.image.handle_kind() == ImageHandleKind::kOpenCvMat) {
    const cv::Mat& source = GetOpenCvMat(frame.image);
    cv::Mat output;
    cv::copyMakeBorder(source, output, padding.top, padding.bottom,
                       padding.left, padding.right, cv::BORDER_CONSTANT, value);
    frame.geometry_trace.AddPad(before_size, padding);
    frame.image = ToRawImage(std::move(output));
    return;
  }

#ifdef MW_INFER_HAS_OPENCV_CUDA_ADAPTER
  if (frame.image.handle_kind() == ImageHandleKind::kOpenCvCudaGpuMat) {
    const cv::cuda::GpuMat& source = GetOpenCvCudaGpuMat(frame.image);
    cv::cuda::GpuMat output = CopyMakeBorderGpuMat(source, padding, value);
    frame.geometry_trace.AddPad(before_size, padding);
    frame.image = ToRawImage(std::move(output));
    return;
  }
#endif

  throw std::invalid_argument("Pad only supports OpenCV Mat and GpuMat");
}

inline void CropFrame(ImageFrame& frame, Rect rect) {
  const ImageSize before_size = frame.image.size();
  ValidateTargetSize(before_size, "Crop input size");
  ValidateCropRect(before_size, rect);

  if (frame.image.handle_kind() == ImageHandleKind::kOpenCvMat) {
    const cv::Mat& source = GetOpenCvMat(frame.image);
    cv::Mat output = source(ToCvRect(rect)).clone();
    frame.geometry_trace.AddCrop(before_size, rect);
    frame.image = ToRawImage(std::move(output));
    return;
  }

#ifdef MW_INFER_HAS_OPENCV_CUDA_ADAPTER
  if (frame.image.handle_kind() == ImageHandleKind::kOpenCvCudaGpuMat) {
    const cv::cuda::GpuMat& source = GetOpenCvCudaGpuMat(frame.image);
    cv::cuda::GpuMat output = source(ToCvRect(rect)).clone();
    frame.geometry_trace.AddCrop(before_size, rect);
    frame.image = ToRawImage(std::move(output));
    return;
  }
#endif

  throw std::invalid_argument("Crop only supports OpenCV Mat and GpuMat");
}

inline void LetterBoxFrame(ImageFrame& frame, ImageSize size,
                           const cv::Scalar& pad_value, int interpolation) {
  const ImageSize before_size = frame.image.size();
  ValidateTargetSize(before_size, "LetterBox input size");
  const ImageSize resized_size = ResizeKeepRatioSize(before_size, size);
  const Padding padding = CenterPadding(resized_size, size);

  if (frame.image.handle_kind() == ImageHandleKind::kOpenCvMat) {
    const cv::Mat& source = GetOpenCvMat(frame.image);
    cv::Mat resized;
    cv::resize(source, resized, ToCvSize(resized_size), 0.0, 0.0,
               interpolation);
    cv::Mat output;
    cv::copyMakeBorder(resized, output, padding.top, padding.bottom,
                       padding.left, padding.right, cv::BORDER_CONSTANT,
                       pad_value);
    frame.geometry_trace.AddLetterBox(before_size, size, resized_size, padding);
    frame.image = ToRawImage(std::move(output));
    return;
  }

#ifdef MW_INFER_HAS_OPENCV_CUDA_ADAPTER
  if (frame.image.handle_kind() == ImageHandleKind::kOpenCvCudaGpuMat) {
    const cv::cuda::GpuMat& source = GetOpenCvCudaGpuMat(frame.image);
    cv::cuda::GpuMat resized =
        ResizeGpuMat(source, resized_size, interpolation);
    cv::cuda::GpuMat output = CopyMakeBorderGpuMat(resized, padding, pad_value);
    frame.geometry_trace.AddLetterBox(before_size, size, resized_size, padding);
    frame.image = ToRawImage(std::move(output));
    return;
  }
#endif

  throw std::invalid_argument("LetterBox only supports OpenCV Mat and GpuMat");
}

}  // namespace detail

inline ImageBatch Resize(ImageBatch batch, ImageSize size,
                         int interpolation = cv::INTER_LINEAR) {
  detail::ValidateTargetSize(size, "Resize size");
  for (ImageFrame& frame : batch.mutable_frames()) {
    detail::ResizeFrame(frame, size, interpolation);
  }
  return batch;
}

inline ImageBatch Resize(RawImageBatch images, ImageSize size,
                         int interpolation = cv::INTER_LINEAR) {
  return Resize(ImageBatch(std::move(images)), size, interpolation);
}

inline ImageBatch Pad(ImageBatch batch, Padding padding,
                      const cv::Scalar& pad_value = cv::Scalar()) {
  detail::ValidatePadding(padding);
  for (ImageFrame& frame : batch.mutable_frames()) {
    detail::PadFrame(frame, padding, pad_value);
  }
  return batch;
}

inline ImageBatch Pad(RawImageBatch images, Padding padding,
                      const cv::Scalar& pad_value = cv::Scalar()) {
  return Pad(ImageBatch(std::move(images)), padding, pad_value);
}

inline ImageBatch Crop(ImageBatch batch, Rect rect) {
  for (ImageFrame& frame : batch.mutable_frames()) {
    detail::CropFrame(frame, rect);
  }
  return batch;
}

inline ImageBatch Crop(RawImageBatch images, Rect rect) {
  return Crop(ImageBatch(std::move(images)), rect);
}

inline ImageBatch CenterCrop(ImageBatch batch, ImageSize size) {
  detail::ValidateTargetSize(size, "CenterCrop size");
  for (ImageFrame& frame : batch.mutable_frames()) {
    detail::CropFrame(frame, detail::CenterCropRect(frame.image.size(), size));
  }
  return batch;
}

inline ImageBatch CenterCrop(RawImageBatch images, ImageSize size) {
  return CenterCrop(ImageBatch(std::move(images)), size);
}

inline ImageBatch LetterBox(ImageBatch batch, ImageSize size,
                            const cv::Scalar& pad_value = cv::Scalar(114, 114,
                                                                     114, 114),
                            int interpolation = cv::INTER_LINEAR) {
  detail::ValidateTargetSize(size, "LetterBox size");
  for (ImageFrame& frame : batch.mutable_frames()) {
    detail::LetterBoxFrame(frame, size, pad_value, interpolation);
  }
  return batch;
}

inline ImageBatch LetterBox(RawImageBatch images, ImageSize size,
                            const cv::Scalar& pad_value = cv::Scalar(114, 114,
                                                                     114, 114),
                            int interpolation = cv::INTER_LINEAR) {
  return LetterBox(ImageBatch(std::move(images)), size, pad_value,
                   interpolation);
}

}  // namespace mw::infer

#endif  // MW_INFER_RUNTIME_ADAPTERS_OPENCV_GEOMETRY_H_
