#include <stdexcept>
#include <utility>

#include "runtime/backends/backend_registry.h"

namespace mw::infer {

namespace {

class TensorRTBackend final : public IBackend {
 public:
  explicit TensorRTBackend(RuntimeConfig config) : config_(std::move(config)) {}

  BackendKind kind() const override { return BackendKind::kTensorRT; }

  InferenceResult Forward(const std::vector<Tensor>& inputs) override {
    (void)inputs;
    throw std::runtime_error("TensorRTBackend forward is not implemented yet");
  }

 private:
  RuntimeConfig config_;
};

}  // namespace

std::unique_ptr<IBackend> CreateTensorRTBackend(const RuntimeConfig& config) {
  return std::make_unique<TensorRTBackend>(config);
}

}  // namespace mw::infer
