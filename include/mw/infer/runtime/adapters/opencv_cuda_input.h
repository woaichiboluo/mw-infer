#ifndef MW_INFER_RUNTIME_ADAPTERS_OPENCV_CUDA_INPUT_H_
#define MW_INFER_RUNTIME_ADAPTERS_OPENCV_CUDA_INPUT_H_

#include <opencv2/core/cuda.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

#include "mw/infer/runtime/adapters/opencv_input.h"

namespace mw::infer {

using OpenCvCudaGpuMatBatch = std::vector<cv::cuda::GpuMat>;

namespace detail {

inline ImageDesc MakeOpenCvCudaImageDesc(const cv::cuda::GpuMat& image) {
  if (image.empty()) {
    throw std::invalid_argument("OpenCV CUDA image is empty");
  }

  const int channels = image.channels();
  return ImageDesc{
      ImageSize{image.cols, image.rows},
      OpenCvChannelsToPixelFormat(channels),
      OpenCvDepthToDataType(image.depth()),
      channels,
      ImageMemoryKind::kCuda,
      {ImagePlaneDesc{image.step, image.elemSize()}},
  };
}

}  // namespace detail

template <>
struct RawImageConverter<cv::cuda::GpuMat> {
  static RawImage Convert(cv::cuda::GpuMat image) {
    ImageDesc desc = detail::MakeOpenCvCudaImageDesc(image);
    return RawImage::FromHandle(
        std::move(desc), ImageHandleKind::kOpenCvCudaGpuMat, std::move(image));
  }
};

inline const cv::cuda::GpuMat& GetOpenCvCudaGpuMat(const RawImage& image) {
  if (image.memory_kind() != ImageMemoryKind::kCuda) {
    throw std::invalid_argument("RawImage does not store a CUDA image");
  }
  if (image.handle_kind() != ImageHandleKind::kOpenCvCudaGpuMat) {
    throw std::invalid_argument(
        "RawImage does not store an OpenCV CUDA GpuMat");
  }
  return *static_cast<const cv::cuda::GpuMat*>(image.handle());
}

inline std::vector<cv::cuda::GpuMat> GetOpenCvCudaGpuMatBatch(
    const RawImageBatch& batch) {
  std::vector<cv::cuda::GpuMat> images;
  images.reserve(batch.size());
  for (const RawImage& image : batch.images()) {
    images.push_back(GetOpenCvCudaGpuMat(image));
  }
  return images;
}

}  // namespace mw::infer

#endif  // MW_INFER_RUNTIME_ADAPTERS_OPENCV_CUDA_INPUT_H_
