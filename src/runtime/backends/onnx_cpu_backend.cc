#include <utility>

#include "runtime/backends/backend_registry.h"
#include "runtime/backends/onnx_runtime_session.h"

namespace mw::infer {

namespace {

class OnnxCpuBackend final : public IBackend {
 public:
  explicit OnnxCpuBackend(RuntimeConfig config)
      : session_(std::move(config), false) {}

  BackendKind kind() const override { return BackendKind::kOnnxCpu; }

  const InferOutputs& InferBatch(const std::vector<cv::Mat>& inputs) override {
    return session_.Infer(inputs);
  }

  std::unique_ptr<IBackend> Clone() const override {
    return std::make_unique<OnnxCpuBackend>(session_.config());
  }

 private:
  OnnxRuntimeSession session_;
};

}  // namespace

std::unique_ptr<IBackend> CreateOnnxCpuBackend(const RuntimeConfig& config) {
  return std::make_unique<OnnxCpuBackend>(config);
}

}  // namespace mw::infer
