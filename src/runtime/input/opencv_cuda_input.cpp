#include "mw/infer/runtime/input/opencv_cuda_input.h"

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

ImageDesc MakeOpenCvCudaImageDesc(const cv::cuda::GpuMat& image) {
  if (image.empty()) {
    throw std::invalid_argument("OpenCV CUDA image is empty");
  }

  const int channels = image.channels();
  ImageDesc desc{
      ImageSize{image.cols, image.rows},
      OpenCvChannelsToPixelFormat(channels),
      OpenCvDepthToDataType(image.depth()),
      channels,
      ImageMemoryKind::kCuda,
      {ImagePlaneDesc{image.step, image.elemSize()}},
  };
  desc.device_id = cv::cuda::getDevice();
  return desc;
}

}  // namespace

RawImage RawImageConverter<cv::cuda::GpuMat>::Convert(cv::cuda::GpuMat image) {
  ImageDesc desc = MakeOpenCvCudaImageDesc(image);
  return RawImage::FromHandle(
      std::move(desc), ImageHandleKind::kOpenCvCudaGpuMat, std::move(image));
}

const cv::cuda::GpuMat& GetOpenCvCudaGpuMat(const RawImage& image) {
  if (image.memory_kind() != ImageMemoryKind::kCuda) {
    throw std::invalid_argument("RawImage does not store a CUDA image");
  }
  if (image.handle_kind() != ImageHandleKind::kOpenCvCudaGpuMat) {
    throw std::invalid_argument(
        "RawImage does not store an OpenCV CUDA GpuMat");
  }
  return *static_cast<const cv::cuda::GpuMat*>(image.handle());
}

std::vector<cv::cuda::GpuMat> GetOpenCvCudaGpuMatBatch(
    const RawImageBatch& batch) {
  std::vector<cv::cuda::GpuMat> images;
  images.reserve(batch.size());
  for (const RawImage& image : batch.images()) {
    images.push_back(GetOpenCvCudaGpuMat(image));
  }
  return images;
}

}  // namespace mw::infer
