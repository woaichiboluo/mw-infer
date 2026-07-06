#ifndef MW_INFER_BACKEND_H_
#define MW_INFER_BACKEND_H_

#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <string_view>
#include <vector>

#include "mw/infer/runtime/infer_outputs.h"
#include "mw/infer/runtime/runtime_config.h"

namespace mw::infer {

class IBackend {
 public:
  virtual ~IBackend() = default;

  virtual BackendKind kind() const = 0;
  virtual const InferOutputs& InferBatch(
      const std::vector<cv::Mat>& inputs) = 0;
  virtual const InferOutputs& InferBatch(
      const std::vector<cv::cuda::GpuMat>& inputs);
  virtual std::unique_ptr<IBackend> Clone() const = 0;
};

std::string_view BackendName(BackendKind backend);
std::unique_ptr<IBackend> CreateBackend(const RuntimeConfig& config);

}  // namespace mw::infer

#endif  // MW_INFER_BACKEND_H_
