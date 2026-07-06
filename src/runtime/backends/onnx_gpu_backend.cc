#include <utility>

#include "runtime/backends/backend_registry.h"
#include "runtime/backends/onnx_runtime_session.h"

namespace mw::infer {

namespace {

class OnnxGpuBackend final : public IBackend {
 public:
  explicit OnnxGpuBackend(RuntimeConfig config)
      : session_(std::move(config), true) {}

  BackendKind kind() const override { return BackendKind::kOnnxGpu; }

  const InferOutputs& InferBatch(const std::vector<cv::Mat>& inputs) override {
    return session_.Infer(inputs);
  }

  const InferOutputs& InferBatch(
      const std::vector<cv::cuda::GpuMat>& inputs) override {
    return session_.Infer(inputs);
  }

  std::unique_ptr<IBackend> Clone() const override {
    return std::make_unique<OnnxGpuBackend>(session_.config());
  }

 private:
  OnnxRuntimeSession session_;
};

}  // namespace

std::unique_ptr<IBackend> CreateOnnxGpuBackend(const RuntimeConfig& config) {
  return std::make_unique<OnnxGpuBackend>(config);
}

}  // namespace mw::infer
