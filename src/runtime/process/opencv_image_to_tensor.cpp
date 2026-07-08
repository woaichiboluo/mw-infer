#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include "mw/infer/runtime/input/opencv_input.h"
#include "mw/infer/runtime/process/image_to_tensor.h"

namespace mw::infer {
namespace {

int ToOpenCvDepth(DataType data_type) {
  switch (data_type) {
    case DataType::kUInt8:
      return CV_8U;
#ifdef CV_16F
    case DataType::kFloat16:
      return CV_16F;
#endif
    case DataType::kFloat32:
      return CV_32F;
    case DataType::kUnknown:
    case DataType::kInt8:
    case DataType::kUInt16:
    case DataType::kInt16:
    case DataType::kInt32:
    case DataType::kInt64:
    case DataType::kFloat64:
#ifndef CV_16F
    case DataType::kFloat16:
#endif
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

int TensorImageType(const Tensor& output, int channels) {
  return CV_MAKETYPE(ToOpenCvDepth(output.data_type()), channels);
}

std::byte* TensorBytes(Tensor* tensor) {
  return static_cast<std::byte*>(tensor->data());
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

cv::Mat ConvertImage(const cv::Mat& image, int output_type) {
  cv::Mat converted;
  image.convertTo(converted, output_type);
  return converted;
}

void CopyToBhwc(const cv::Mat& image, PixelFormat pixel_format, Tensor* output,
                std::size_t batch_index) {
  const std::size_t image_bytes = static_cast<std::size_t>(image.rows) *
                                  image.cols * image.channels() *
                                  ElementSize(output->data_type());
  cv::Mat output_view(image.rows, image.cols,
                      TensorImageType(*output, image.channels()),
                      TensorBytes(output) + batch_index * image_bytes);
  if (!NeedsRgbChannelOrder(pixel_format)) {
    image.convertTo(output_view, output_view.type());
    return;
  }

  cv::Mat converted = ConvertImage(image, output_view.type());
  std::vector<cv::Mat> source_planes;
  cv::split(converted, source_planes);

  std::vector<cv::Mat> target_planes;
  target_planes.reserve(static_cast<std::size_t>(image.channels()));
  for (int channel = 0; channel < image.channels(); ++channel) {
    target_planes.push_back(source_planes[static_cast<std::size_t>(
        RgbSourceChannel(pixel_format, channel))]);
  }
  cv::merge(target_planes, output_view);
}

void CopyToBchw(const cv::Mat& image, PixelFormat pixel_format, Tensor* output,
                std::size_t batch_index) {
  const int channels = image.channels();
  const std::size_t plane_elements =
      static_cast<std::size_t>(image.rows) * image.cols;
  const std::size_t batch_elements = plane_elements * channels;
  const std::size_t element_bytes = ElementSize(output->data_type());
  std::byte* batch_data =
      TensorBytes(output) + batch_index * batch_elements * element_bytes;

  std::vector<cv::Mat> planes;
  planes.reserve(static_cast<std::size_t>(channels));
  for (int channel = 0; channel < channels; ++channel) {
    planes.emplace_back(image.rows, image.cols,
                        ToOpenCvDepth(output->data_type()),
                        batch_data + static_cast<std::size_t>(channel) *
                                         plane_elements * element_bytes);
  }

  cv::Mat converted = ConvertImage(image, TensorImageType(*output, channels));
  if (!NeedsRgbChannelOrder(pixel_format)) {
    cv::split(converted, planes);
    return;
  }

  std::vector<cv::Mat> source_planes;
  cv::split(converted, source_planes);
  for (int channel = 0; channel < channels; ++channel) {
    source_planes[static_cast<std::size_t>(
                      RgbSourceChannel(pixel_format, channel))]
        .copyTo(planes[static_cast<std::size_t>(channel)]);
  }
}

class OpenCvImageToTensorAdapter final : public ImageToTensorAdapter {
 public:
  bool Supports(const RawImageBatch& images, Device target_device,
                const TensorInfo& input, TensorLayout layout) const override {
    if (target_device.type != DeviceType::kCpu || !IsSupportedLayout(layout) ||
        !IsSupportedTargetDataType(input.data_type)) {
      return false;
    }
    for (const RawImage& image : images.images()) {
      if (image.handle_kind() != ImageHandleKind::kOpenCvMat ||
          !IsSupportedSourceDataType(image.data_type())) {
        return false;
      }
    }
    return true;
  }

  void Convert(const RawImageBatch& images, Tensor* output,
               TensorLayout layout) const override {
    if (output == nullptr || output->empty()) {
      throw std::invalid_argument("Output tensor is empty");
    }
    if (output->device().type != DeviceType::kCpu) {
      throw std::invalid_argument("OpenCV image-to-tensor output must be CPU");
    }

    const std::vector<RawImage>& batch = images.images();
    for (std::size_t batch_index = 0; batch_index < batch.size();
         ++batch_index) {
      const RawImage& raw_image = batch[batch_index];
      const cv::Mat& image = GetOpenCvMat(raw_image);
      switch (layout) {
        case TensorLayout::kBchw:
          CopyToBchw(image, raw_image.pixel_format(), output, batch_index);
          break;
        case TensorLayout::kBhwc:
          CopyToBhwc(image, raw_image.pixel_format(), output, batch_index);
          break;
      }
    }
  }
};

}  // namespace

std::unique_ptr<ImageToTensorAdapter> CreateOpenCvImageToTensorAdapter() {
  return std::make_unique<OpenCvImageToTensorAdapter>();
}

}  // namespace mw::infer
