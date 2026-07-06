#ifndef MW_INFER_SRC_RUNTIME_BACKENDS_ONNX_RUNTIME_SESSION_H_
#define MW_INFER_SRC_RUNTIME_BACKENDS_ONNX_RUNTIME_SESSION_H_

#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <vector>

#include "mw/infer/runtime/infer_outputs.h"
#include "mw/infer/runtime/runtime_config.h"

namespace mw::infer {

class OnnxRuntimeSession {
 public:
  OnnxRuntimeSession(RuntimeConfig config, bool use_cuda_provider);
  ~OnnxRuntimeSession();

  OnnxRuntimeSession(const OnnxRuntimeSession&) = delete;
  OnnxRuntimeSession& operator=(const OnnxRuntimeSession&) = delete;

  const RuntimeConfig& config() const;
  const InferOutputs& Infer(const std::vector<cv::Mat>& inputs);
  const InferOutputs& Infer(const std::vector<cv::cuda::GpuMat>& inputs);

 private:
  class Impl;

  std::unique_ptr<Impl> impl_;
};

}  // namespace mw::infer

#endif  // MW_INFER_SRC_RUNTIME_BACKENDS_ONNX_RUNTIME_SESSION_H_
