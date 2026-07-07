#include "mw/infer/runtime/input/opencv_input.h"

#include <stdexcept>
#include <utility>

namespace mw::infer {
namespace {

DataType OpenCvDepthToDataType(int depth) {
  switch (depth) {
    case CV_8U:
      return DataType::kUInt8;
    case CV_8S:
      return DataType::kInt8;
    case CV_16U:
      return DataType::kUInt16;
    case CV_16S:
      return DataType::kInt16;
    case CV_32S:
      return DataType::kInt32;
#ifdef CV_16F
    case CV_16F:
      return DataType::kFloat16;
#endif
    case CV_32F:
      return DataType::kFloat32;
    case CV_64F:
      return DataType::kFloat64;
    default:
      throw std::invalid_argument("OpenCV image has an unsupported depth");
  }
}

PixelFormat OpenCvChannelsToPixelFormat(int channels) {
  switch (channels) {
    case 1:
      return PixelFormat::kGray;
    case 3:
      return PixelFormat::kBgr;
    case 4:
      return PixelFormat::kBgra;
    default:
      return PixelFormat::kUnknown;
  }
}

ImageDesc MakeOpenCvImageDesc(const cv::Mat& image) {
  if (image.empty()) {
    throw std::invalid_argument("OpenCV image is empty");
  }

  const int channels = image.channels();
  return ImageDesc{
      ImageSize{image.cols, image.rows},
      OpenCvChannelsToPixelFormat(channels),
      OpenCvDepthToDataType(image.depth()),
      channels,
      ImageMemoryKind::kHost,
      {ImagePlaneDesc{image.step[0], image.elemSize()}},
  };
}

}  // namespace

RawImage RawImageConverter<cv::Mat>::Convert(cv::Mat image) {
  ImageDesc desc = MakeOpenCvImageDesc(image);
  return RawImage::FromHandle(std::move(desc), ImageHandleKind::kOpenCvMat,
                              std::move(image));
}

const cv::Mat& GetOpenCvMat(const RawImage& image) {
  if (image.memory_kind() != ImageMemoryKind::kHost) {
    throw std::invalid_argument("RawImage does not store a host image");
  }
  if (image.handle_kind() != ImageHandleKind::kOpenCvMat) {
    throw std::invalid_argument("RawImage does not store an OpenCV Mat");
  }
  return *static_cast<const cv::Mat*>(image.handle());
}

std::vector<cv::Mat> GetOpenCvMatBatch(const RawImageBatch& batch) {
  std::vector<cv::Mat> images;
  images.reserve(batch.size());
  for (const RawImage& image : batch.images()) {
    images.push_back(GetOpenCvMat(image));
  }
  return images;
}

}  // namespace mw::infer
