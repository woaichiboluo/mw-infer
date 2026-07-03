#include <stdexcept>
#include <utility>

#include "runtime/backends/backend_registry.h"

namespace mw::infer {

namespace {

class OnnxCpuBackend final : public IBackend {
 public:
  explicit OnnxCpuBackend(RuntimeConfig config) : config_(std::move(config)) {}

  BackendKind kind() const override { return BackendKind::kOnnxCpu; }

  InferenceResult Forward(const std::vector<Tensor>& inputs) override {
    (void)inputs;
    throw std::runtime_error("OnnxCpuBackend forward is not implemented yet");
  }

 private:
  RuntimeConfig config_;
};

}  // namespace

std::unique_ptr<IBackend> CreateOnnxCpuBackend(const RuntimeConfig& config) {
  return std::make_unique<OnnxCpuBackend>(config);
}

}  // namespace mw::infer
