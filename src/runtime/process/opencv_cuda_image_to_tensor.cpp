#include <cstddef>
#include <memory>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaarithm.hpp>
#include <stdexcept>
#include <vector>

#include "mw/infer/runtime/input/opencv_cuda_input.h"
#include "mw/infer/runtime/input/opencv_input.h"
#include "mw/infer/runtime/process/image_to_tensor.h"

namespace mw::infer {

#if defined(MW_INFER_HAS_CUDA_PREPROCESS)
namespace process_internal {
void RunImageToTensorOnDevice(const void* source, std::size_t source_step,
                              DataType source_type, int rows, int cols,
                              int channels, PixelFormat pixel_format,
                              Tensor* output, std::size_t batch_index,
                              TensorLayout layout, cudaStream_t stream);
}  // namespace process_internal
#endif

namespace {

int ToOpenCvDepth(DataType data_type) {
  switch (data_type) {
    case DataType::kUInt8:
      return CV_8U;
    case DataType::kFloat32:
      return CV_32F;
    case DataType::kUnknown:
    case DataType::kInt8:
    case DataType::kUInt16:
    case DataType::kInt16:
    case DataType::kInt32:
    case DataType::kInt64:
    case DataType::kFloat16:
    case DataType::kFloat64:
      throw std::invalid_argument("Unsupported tensor data type");
  }
  throw std::invalid_argument("Unsupported tensor data type");
}

bool IsSupportedTargetDataType(DataType data_type) {
  try {
    static_cast<void>(ToOpenCvDepth(data_type));
  } catch (const std::invalid_argument&) {
    return false;
  }
  return true;
}

bool IsSupportedSourceDataType(DataType data_type) {
  switch (data_type) {
    case DataType::kUInt8:
    case DataType::kInt8:
    case DataType::kUInt16:
    case DataType::kInt16:
    case DataType::kInt32:
    case DataType::kFloat32:
    case DataType::kFloat64:
      return true;
    case DataType::kUnknown:
    case DataType::kInt64:
    case DataType::kFloat16:
      return false;
  }
  return false;
}

bool IsSupportedLayout(TensorLayout layout) {
  switch (layout) {
    case TensorLayout::kBchw:
    case TensorLayout::kBhwc:
      return true;
  }
  return false;
}

std::size_t ElementSize(DataType data_type) { return DataTypeSize(data_type); }

std::byte* TensorBytes(Tensor* tensor) {
  return static_cast<std::byte*>(tensor->data());
}

int TensorImageType(const Tensor& output, int channels) {
  return CV_MAKETYPE(ToOpenCvDepth(output.data_type()), channels);
}

int RgbSourceChannel(PixelFormat pixel_format, int channel) {
  switch (pixel_format) {
    case PixelFormat::kBgr:
    case PixelFormat::kBgra:
      if (channel == 0) {
        return 2;
      }
      if (channel == 2) {
        return 0;
      }
      return channel;
    case PixelFormat::kUnknown:
    case PixelFormat::kRgb:
    case PixelFormat::kRgba:
    case PixelFormat::kGray:
    case PixelFormat::kNv12:
    case PixelFormat::kNv21:
      return channel;
  }
  return channel;
}

bool NeedsRgbChannelOrder(PixelFormat pixel_format) {
  return pixel_format == PixelFormat::kBgr ||
         pixel_format == PixelFormat::kBgra;
}

bool IsSourceDeviceSupported(const RawImage& image, Device target_device) {
  if (image.handle_kind() != ImageHandleKind::kOpenCvCudaGpuMat) {
    return true;
  }
  const Device source_device = image.device();
  return source_device.type == DeviceType::kCuda &&
         source_device.id == target_device.id;
}

void ConvertTo(const cv::cuda::GpuMat& source, cv::cuda::GpuMat& target,
               int target_type, cv::cuda::Stream* stream) {
  if (stream == nullptr) {
    source.convertTo(target, target_type);
    return;
  }
  source.convertTo(target, target_type, *stream);
}

void Split(const cv::cuda::GpuMat& source,
           std::vector<cv::cuda::GpuMat>& target,
           cv::cuda::Stream* stream) {
  if (stream == nullptr) {
    cv::cuda::split(source, target);
    return;
  }
  cv::cuda::split(source, target, *stream);
}

void Merge(const std::vector<cv::cuda::GpuMat>& source,
           cv::cuda::GpuMat& target, cv::cuda::Stream* stream) {
  if (stream == nullptr) {
    cv::cuda::merge(source, target);
    return;
  }
  cv::cuda::merge(source, target, *stream);
}

void CopyTo(const cv::cuda::GpuMat& source, cv::cuda::GpuMat& target,
            cv::cuda::Stream* stream) {
  if (stream == nullptr) {
    source.copyTo(target);
    return;
  }
  source.copyTo(target, *stream);
}

cv::cuda::GpuMat ToGpuMat(const RawImage& image, Device target_device,
                          cv::cuda::Stream* stream) {
  if (image.handle_kind() == ImageHandleKind::kOpenCvCudaGpuMat) {
    if (!IsSourceDeviceSupported(image, target_device)) {
      throw std::invalid_argument(
          "OpenCV CUDA image device id must match output tensor device id");
    }
    return GetOpenCvCudaGpuMat(image);
  }
  if (image.handle_kind() == ImageHandleKind::kOpenCvMat) {
    cv::cuda::GpuMat gpu_image;
    if (stream == nullptr) {
      gpu_image.upload(GetOpenCvMat(image));
    } else {
      gpu_image.upload(GetOpenCvMat(image), *stream);
    }
    return gpu_image;
  }
  throw std::invalid_argument("RawImage does not store an OpenCV image");
}

void CopyToBhwc(const cv::cuda::GpuMat& image, PixelFormat pixel_format,
                Tensor* output, std::size_t batch_index,
                cv::cuda::Stream* stream) {
  const std::size_t row_bytes = static_cast<std::size_t>(image.cols) *
                                image.channels() *
                                ElementSize(output->data_type());
  const std::size_t image_bytes =
      static_cast<std::size_t>(image.rows) * row_bytes;
  cv::cuda::GpuMat output_view(
      image.rows, image.cols, TensorImageType(*output, image.channels()),
      TensorBytes(output) + batch_index * image_bytes, row_bytes);
  if (!NeedsRgbChannelOrder(pixel_format)) {
    ConvertTo(image, output_view, output_view.type(), stream);
    return;
  }

  cv::cuda::GpuMat converted;
  ConvertTo(image, converted, output_view.type(), stream);
  std::vector<cv::cuda::GpuMat> source_planes;
  Split(converted, source_planes, stream);

  std::vector<cv::cuda::GpuMat> target_planes;
  target_planes.reserve(static_cast<std::size_t>(image.channels()));
  for (int channel = 0; channel < image.channels(); ++channel) {
    target_planes.push_back(source_planes[static_cast<std::size_t>(
        RgbSourceChannel(pixel_format, channel))]);
  }
  Merge(target_planes, output_view, stream);
}

void CopyToBchw(const cv::cuda::GpuMat& image, PixelFormat pixel_format,
                Tensor* output, std::size_t batch_index,
                cv::cuda::Stream* stream) {
  const int channels = image.channels();
  const std::size_t plane_elements =
      static_cast<std::size_t>(image.rows) * image.cols;
  const std::size_t batch_elements = plane_elements * channels;
  const std::size_t element_bytes = ElementSize(output->data_type());
  std::byte* batch_data =
      TensorBytes(output) + batch_index * batch_elements * element_bytes;

  std::vector<cv::cuda::GpuMat> planes;
  planes.reserve(static_cast<std::size_t>(channels));
  const std::size_t plane_step =
      static_cast<std::size_t>(image.cols) * element_bytes;
  for (int channel = 0; channel < channels; ++channel) {
    planes.emplace_back(image.rows, image.cols,
                        ToOpenCvDepth(output->data_type()),
                        batch_data + static_cast<std::size_t>(channel) *
                                         plane_elements * element_bytes,
                        plane_step);
  }

  cv::cuda::GpuMat converted;
  ConvertTo(image, converted, TensorImageType(*output, channels), stream);
  if (!NeedsRgbChannelOrder(pixel_format)) {
    Split(converted, planes, stream);
    return;
  }

  std::vector<cv::cuda::GpuMat> source_planes;
  Split(converted, source_planes, stream);
  for (int channel = 0; channel < channels; ++channel) {
    CopyTo(source_planes[static_cast<std::size_t>(
               RgbSourceChannel(pixel_format, channel))],
           planes[static_cast<std::size_t>(channel)], stream);
  }
}

bool SameDevice(Device lhs, Device rhs) {
  return lhs.type == rhs.type && lhs.id == rhs.id;
}

class OpenCvCudaImageToTensorAdapter final : public ImageToTensorAdapter {
 public:
  bool Supports(const RawImageBatch& images, Device target_device,
                const TensorInfo& input, TensorLayout layout) const override {
    if (target_device.type != DeviceType::kCuda || !IsSupportedLayout(layout) ||
        !IsSupportedTargetDataType(input.data_type)) {
      return false;
    }
    for (const RawImage& image : images.images()) {
      if ((image.handle_kind() != ImageHandleKind::kOpenCvCudaGpuMat &&
           image.handle_kind() != ImageHandleKind::kOpenCvMat) ||
          !IsSupportedSourceDataType(image.data_type()) ||
          !IsSourceDeviceSupported(image, target_device)) {
        return false;
      }
    }
    return true;
  }

  void Convert(const RawImageBatch& images, Tensor* output,
                TensorLayout layout) const override {
    ConvertImpl(images, output, layout, nullptr);
  }

  void Convert(const RawImageBatch& images, Tensor* output,
               ExecutionStream& stream, TensorLayout layout) const override {
    if (output == nullptr || !SameDevice(output->device(), stream.device())) {
      throw std::invalid_argument(
          "OpenCV CUDA image-to-tensor stream device does not match output");
    }
    cv::cuda::setDevice(output->device().id);
    cv::cuda::Stream cv_stream =
        cv::cuda::StreamAccessor::wrapStream(stream.cuda_handle());
    ConvertImpl(images, output, layout, &cv_stream);
  }

 private:
  void ConvertImpl(const RawImageBatch& images, Tensor* output,
                   TensorLayout layout, cv::cuda::Stream* stream) const {
    if (output == nullptr || output->empty()) {
      throw std::invalid_argument("Output tensor is empty");
    }
    if (output->device().type != DeviceType::kCuda) {
      throw std::invalid_argument(
          "OpenCV CUDA image-to-tensor output must be CUDA");
    }

    cv::cuda::setDevice(output->device().id);
    const std::vector<RawImage>& batch = images.images();
    for (std::size_t batch_index = 0; batch_index < batch.size();
         ++batch_index) {
      const RawImage& raw_image = batch[batch_index];
#if defined(MW_INFER_HAS_CUDA_PREPROCESS)
      if (stream != nullptr &&
          raw_image.handle_kind() == ImageHandleKind::kOpenCvCudaGpuMat) {
        const cv::cuda::GpuMat& image = GetOpenCvCudaGpuMat(raw_image);
        process_internal::RunImageToTensorOnDevice(
            image.data, image.step, raw_image.data_type(), image.rows,
            image.cols, image.channels(), raw_image.pixel_format(), output,
            batch_index, layout,
            cv::cuda::StreamAccessor::getStream(*stream));
        continue;
      }
#endif
      cv::cuda::GpuMat image =
          ToGpuMat(raw_image, output->device(), stream);
      switch (layout) {
        case TensorLayout::kBchw:
          CopyToBchw(image, raw_image.pixel_format(), output, batch_index,
                     stream);
          break;
        case TensorLayout::kBhwc:
          CopyToBhwc(image, raw_image.pixel_format(), output, batch_index,
                     stream);
          break;
      }
    }
  }
};

}  // namespace

std::unique_ptr<ImageToTensorAdapter> CreateOpenCvCudaImageToTensorAdapter() {
  return std::make_unique<OpenCvCudaImageToTensorAdapter>();
}

}  // namespace mw::infer
