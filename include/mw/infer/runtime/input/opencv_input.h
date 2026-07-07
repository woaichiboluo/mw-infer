#ifndef MW_INFER_RUNTIME_INPUT_OPENCV_INPUT_H_
#define MW_INFER_RUNTIME_INPUT_OPENCV_INPUT_H_

#include <opencv2/core.hpp>
#include <vector>

#include "mw/infer/runtime/input/input.h"

namespace mw::infer {

using OpenCvMatBatch = std::vector<cv::Mat>;

template <>
struct RawImageConverter<cv::Mat> {
  static RawImage Convert(cv::Mat image);
};

const cv::Mat& GetOpenCvMat(const RawImage& image);
std::vector<cv::Mat> GetOpenCvMatBatch(const RawImageBatch& batch);

}  // namespace mw::infer

#endif  // MW_INFER_RUNTIME_INPUT_OPENCV_INPUT_H_
