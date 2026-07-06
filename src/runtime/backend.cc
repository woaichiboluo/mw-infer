#include "mw/infer/runtime/backend.h"

#include <stdexcept>

namespace mw::infer {

const InferOutputs& IBackend::InferBatch(const std::vector<cv::cuda::GpuMat>&) {
  throw std::invalid_argument(
      "backend does not support cv::cuda::GpuMat batch input");
}

}  // namespace mw::infer
