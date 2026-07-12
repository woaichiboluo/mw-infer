#if defined(MW_INFER_HAS_CUDA_RUNTIME)
#include <cuda_runtime_api.h>
#include <opencv2/core/cuda_stream_accessor.hpp>
#endif

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <string>
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

void SetImageDevice(const RawImage& image) {
  const Device device = image.device();
  if (device.type != DeviceType::kCuda) {
    throw std::invalid_argument("OpenCV CUDA image must be on a CUDA device");
  }
  cv::cuda::setDevice(device.id);
}

#if defined(MW_INFER_HAS_CUDA_RUNTIME)
void CheckCuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) +
                             " failed: " + cudaGetErrorString(status));
  }
}

bool SameDevice(Device lhs, Device rhs) {
  return lhs.type == rhs.type && lhs.id == rhs.id;
}

void SynchronizeNoThrow(cudaStream_t stream) noexcept {
  static_cast<void>(cudaStreamSynchronize(stream));
}

struct AsyncGpuMatStorage {
  Device device;
  void* data = nullptr;
  cv::cuda::GpuMat image;
  std::vector<std::shared_ptr<AsyncGpuMatStorage>> dependencies;

  ~AsyncGpuMatStorage() {
    image.release();
    if (data != nullptr) {
      static_cast<void>(cudaSetDevice(device.id));
      static_cast<void>(cudaFree(data));
    }
  }
};

std::shared_ptr<AsyncGpuMatStorage> AllocateAsyncGpuMat(
    int rows, int cols, int type, Device device, cudaStream_t stream) {
  if (rows <= 0 || cols <= 0) {
    throw std::invalid_argument("OpenCV CUDA output size must be positive");
  }
  const std::size_t row_bytes =
      static_cast<std::size_t>(cols) * CV_ELEM_SIZE(type);
  if (static_cast<std::size_t>(rows) >
      std::numeric_limits<std::size_t>::max() / row_bytes) {
    throw std::invalid_argument("OpenCV CUDA output size overflows");
  }

  CheckCuda(cudaSetDevice(device.id), "cudaSetDevice");
  auto storage = std::make_shared<AsyncGpuMatStorage>();
  storage->device = device;
  CheckCuda(cudaMallocAsync(&storage->data,
                            static_cast<std::size_t>(rows) * row_bytes, stream),
            "cudaMallocAsync");
  try {
    storage->image =
        cv::cuda::GpuMat(rows, cols, type, storage->data, row_bytes);
  } catch (...) {
    SynchronizeNoThrow(stream);
    throw;
  }
  return storage;
}

RawImage ToAsyncRawImage(
    const RawImage& source,
    const std::shared_ptr<AsyncGpuMatStorage>& storage) {
  ImageDesc desc = source.desc();
  desc.size = ImageSize{storage->image.cols, storage->image.rows};
  desc.planes = {ImagePlaneDesc{storage->image.step,
                                storage->image.elemSize()}};
  auto handle = std::shared_ptr<cv::cuda::GpuMat>(storage, &storage->image);
  return RawImage::FromSharedHandle(
      std::move(desc), ImageHandleKind::kOpenCvCudaGpuMat, std::move(handle));
}

std::shared_ptr<AsyncGpuMatStorage> ResizeGpuMatAsync(
    const cv::cuda::GpuMat& source, ImageSize size,
    Interpolation interpolation, Device device, cudaStream_t cuda_stream,
    cv::cuda::Stream& stream) {
  const int channels = source.channels();
  if (channels > 4) {
    throw std::invalid_argument("OpenCV CUDA resize supports up to 4 channels");
  }
  auto output = AllocateAsyncGpuMat(size.height, size.width, source.type(),
                                    device, cuda_stream);
  try {
    if (channels != 2) {
      cv::cuda::resize(source, output->image, ToOpenCvSize(size), 0.0, 0.0,
                       ToOpenCvInterpolation(interpolation), stream);
      return output;
    }

    std::vector<cv::cuda::GpuMat> source_channels;
    std::vector<cv::cuda::GpuMat> resized_channels;
    source_channels.reserve(2);
    resized_channels.reserve(2);
    output->dependencies.reserve(4);
    const int channel_type = CV_MAKETYPE(source.depth(), 1);
    for (int channel = 0; channel < 2; ++channel) {
      auto source_channel = AllocateAsyncGpuMat(
          source.rows, source.cols, channel_type, device, cuda_stream);
      source_channels.push_back(source_channel->image);
      output->dependencies.push_back(std::move(source_channel));
      auto resized_channel = AllocateAsyncGpuMat(
          size.height, size.width, channel_type, device, cuda_stream);
      resized_channels.push_back(resized_channel->image);
      output->dependencies.push_back(std::move(resized_channel));
    }
    cv::cuda::split(source, source_channels, stream);
    for (int channel = 0; channel < 2; ++channel) {
      cv::cuda::resize(source_channels[static_cast<std::size_t>(channel)],
                       resized_channels[static_cast<std::size_t>(channel)],
                       ToOpenCvSize(size), 0.0, 0.0,
                       ToOpenCvInterpolation(interpolation), stream);
    }
    cv::cuda::merge(resized_channels, output->image, stream);
    return output;
  } catch (...) {
    SynchronizeNoThrow(cuda_stream);
    throw;
  }
}

std::shared_ptr<AsyncGpuMatStorage> CopyMakeBorderGpuMatAsync(
    const cv::cuda::GpuMat& source, Padding padding,
    const FillValue& fill_value, Device device, cudaStream_t cuda_stream,
    cv::cuda::Stream& stream) {
  const int channels = source.channels();
  if (channels > 4) {
    throw std::invalid_argument(
        "OpenCV CUDA copyMakeBorder supports up to 4 channels");
  }
  const int output_rows = source.rows + padding.top + padding.bottom;
  const int output_cols = source.cols + padding.left + padding.right;
  auto output = AllocateAsyncGpuMat(output_rows, output_cols, source.type(),
                                    device, cuda_stream);
  const cv::Scalar value = ToOpenCvScalar(fill_value);
  try {
    if (channels != 2) {
      cv::cuda::copyMakeBorder(
          source, output->image, padding.top, padding.bottom, padding.left,
          padding.right, cv::BORDER_CONSTANT, value, stream);
      return output;
    }

    std::vector<cv::cuda::GpuMat> source_channels;
    std::vector<cv::cuda::GpuMat> padded_channels;
    source_channels.reserve(2);
    padded_channels.reserve(2);
    output->dependencies.reserve(4);
    const int channel_type = CV_MAKETYPE(source.depth(), 1);
    for (int channel = 0; channel < 2; ++channel) {
      auto source_channel = AllocateAsyncGpuMat(
          source.rows, source.cols, channel_type, device, cuda_stream);
      source_channels.push_back(source_channel->image);
      output->dependencies.push_back(std::move(source_channel));
      auto padded_channel = AllocateAsyncGpuMat(
          output_rows, output_cols, channel_type, device, cuda_stream);
      padded_channels.push_back(padded_channel->image);
      output->dependencies.push_back(std::move(padded_channel));
    }
    cv::cuda::split(source, source_channels, stream);
    for (int channel = 0; channel < 2; ++channel) {
      cv::cuda::copyMakeBorder(
          source_channels[static_cast<std::size_t>(channel)],
          padded_channels[static_cast<std::size_t>(channel)], padding.top,
          padding.bottom, padding.left, padding.right, cv::BORDER_CONSTANT,
          cv::Scalar(value[channel]), stream);
    }
    cv::cuda::merge(padded_channels, output->image, stream);
    return output;
  } catch (...) {
    SynchronizeNoThrow(cuda_stream);
    throw;
  }
}
#endif

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
    const Device device = image.device();
    return image.handle_kind() == ImageHandleKind::kOpenCvCudaGpuMat &&
           device.type == DeviceType::kCuda && device.id >= 0;
  }

  RawImage Resize(const RawImage& image, ImageSize size,
                   Interpolation interpolation) const override {
    SetImageDevice(image);
    return ToRawImage(
        ResizeGpuMat(GetOpenCvCudaGpuMat(image), size, interpolation));
  }

  RawImage Resize(const RawImage& image, ImageSize size,
                  ExecutionStream& execution_stream,
                  Interpolation interpolation) const override {
#if defined(MW_INFER_HAS_CUDA_RUNTIME)
    SetImageDevice(image);
    if (!SameDevice(image.device(), execution_stream.device())) {
      throw std::invalid_argument(
          "OpenCV CUDA resize stream device does not match image");
    }
    const cudaStream_t cuda_stream = execution_stream.cuda_handle();
    cv::cuda::Stream stream =
        cv::cuda::StreamAccessor::wrapStream(cuda_stream);
    return ToAsyncRawImage(
        image, ResizeGpuMatAsync(GetOpenCvCudaGpuMat(image), size,
                                 interpolation, image.device(), cuda_stream,
                                 stream));
#else
    static_cast<void>(image);
    static_cast<void>(size);
    static_cast<void>(execution_stream);
    static_cast<void>(interpolation);
    throw std::runtime_error(
        "OpenCV CUDA geometry execution stream requires CUDA runtime");
#endif
  }

  RawImage Pad(const RawImage& image, Padding padding,
                const FillValue& value) const override {
    SetImageDevice(image);
    return ToRawImage(
        CopyMakeBorderGpuMat(GetOpenCvCudaGpuMat(image), padding, value));
  }

  RawImage Pad(const RawImage& image, Padding padding,
               ExecutionStream& execution_stream,
               const FillValue& value) const override {
#if defined(MW_INFER_HAS_CUDA_RUNTIME)
    SetImageDevice(image);
    if (!SameDevice(image.device(), execution_stream.device())) {
      throw std::invalid_argument(
          "OpenCV CUDA pad stream device does not match image");
    }
    const cudaStream_t cuda_stream = execution_stream.cuda_handle();
    cv::cuda::Stream stream =
        cv::cuda::StreamAccessor::wrapStream(cuda_stream);
    return ToAsyncRawImage(
        image, CopyMakeBorderGpuMatAsync(GetOpenCvCudaGpuMat(image), padding,
                                         value, image.device(), cuda_stream,
                                         stream));
#else
    static_cast<void>(image);
    static_cast<void>(padding);
    static_cast<void>(execution_stream);
    static_cast<void>(value);
    throw std::runtime_error(
        "OpenCV CUDA geometry execution stream requires CUDA runtime");
#endif
  }

  RawImage Crop(const RawImage& image, Rect rect) const override {
    SetImageDevice(image);
    const cv::cuda::GpuMat& source = GetOpenCvCudaGpuMat(image);
    return ToRawImage(source(ToOpenCvRect(rect)).clone());
  }

  RawImage Crop(const RawImage& image, Rect rect,
                ExecutionStream& execution_stream) const override {
#if defined(MW_INFER_HAS_CUDA_RUNTIME)
    SetImageDevice(image);
    if (!SameDevice(image.device(), execution_stream.device())) {
      throw std::invalid_argument(
          "OpenCV CUDA crop stream device does not match image");
    }
    const cudaStream_t cuda_stream = execution_stream.cuda_handle();
    cv::cuda::Stream stream =
        cv::cuda::StreamAccessor::wrapStream(cuda_stream);
    const cv::cuda::GpuMat& source = GetOpenCvCudaGpuMat(image);
    auto output = AllocateAsyncGpuMat(rect.height, rect.width, source.type(),
                                      image.device(), cuda_stream);
    source(ToOpenCvRect(rect)).copyTo(output->image, stream);
    return ToAsyncRawImage(image, output);
#else
    static_cast<void>(image);
    static_cast<void>(rect);
    static_cast<void>(execution_stream);
    throw std::runtime_error(
        "OpenCV CUDA geometry execution stream requires CUDA runtime");
#endif
  }
};

}  // namespace

std::unique_ptr<GeometryAdapter> CreateOpenCvCudaGeometryAdapter() {
  return std::make_unique<OpenCvCudaGeometryAdapter>();
}

}  // namespace mw::infer
