#include <stdexcept>
#include <utility>

#include "runtime/backends/backend_registry.h"

namespace mw::infer {

namespace {

class OnnxCudaBackend final : public IBackend {
 public:
  explicit OnnxCudaBackend(RuntimeConfig config) : config_(std::move(config)) {}

  BackendKind kind() const override { return BackendKind::kOnnxCuda; }

  InferenceResult Forward(const std::vector<Tensor>& inputs) override {
    (void)inputs;
    throw std::runtime_error("OnnxCudaBackend forward is not implemented yet");
  }

 private:
  RuntimeConfig config_;
};

}  // namespace

std::unique_ptr<IBackend> CreateOnnxCudaBackend(const RuntimeConfig& config) {
  return std::make_unique<OnnxCudaBackend>(config);
}

}  // namespace mw::infer
