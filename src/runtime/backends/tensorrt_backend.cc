#include <stdexcept>
#include <utility>

#include "runtime/backends/backend_registry.h"

namespace mw::infer {

namespace {

class TensorRTBackend final : public IBackend {
 public:
  explicit TensorRTBackend(RuntimeConfig config) : config_(std::move(config)) {}

  BackendKind kind() const override { return BackendKind::kTensorRT; }

  const InferOutputs& InferBatch(const std::vector<cv::Mat>&) override {
    throw std::runtime_error("TensorRT backend inference is not implemented");
  }

  std::unique_ptr<IBackend> Clone() const override {
    return std::make_unique<TensorRTBackend>(config_);
  }

 private:
  RuntimeConfig config_;
};

}  // namespace

std::unique_ptr<IBackend> CreateTensorRTBackend(const RuntimeConfig& config) {
  return std::make_unique<TensorRTBackend>(config);
}

}  // namespace mw::infer
