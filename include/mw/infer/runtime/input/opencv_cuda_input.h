#ifndef MW_INFER_RUNTIME_INPUT_OPENCV_CUDA_INPUT_H_
#define MW_INFER_RUNTIME_INPUT_OPENCV_CUDA_INPUT_H_

#include <opencv2/core/cuda.hpp>
#include <vector>

#include "mw/infer/runtime/input/opencv_input.h"

namespace mw::infer {

using OpenCvCudaGpuMatBatch = std::vector<cv::cuda::GpuMat>;

template <>
struct RawImageConverter<cv::cuda::GpuMat> {
  static RawImage Convert(cv::cuda::GpuMat image);
};

const cv::cuda::GpuMat& GetOpenCvCudaGpuMat(const RawImage& image);
std::vector<cv::cuda::GpuMat> GetOpenCvCudaGpuMatBatch(
    const RawImageBatch& batch);

}  // namespace mw::infer

#endif  // MW_INFER_RUNTIME_INPUT_OPENCV_CUDA_INPUT_H_
