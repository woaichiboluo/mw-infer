#include "mw/infer/runtime/session.h"

#include <utility>

namespace mw::infer {

Session::Session(RuntimeConfig config)
    : config_(std::move(config)), backend_(CreateBackend(config_)) {}

Session::Session(RuntimeConfig config, std::unique_ptr<IBackend> backend)
    : config_(std::move(config)), backend_(std::move(backend)) {}

const RuntimeConfig& Session::config() const { return config_; }

BackendKind Session::backend_kind() const { return backend_->kind(); }

const InferOutputs& Session::InferBatch(const std::vector<cv::Mat>& inputs) {
  return backend_->InferBatch(inputs);
}

const InferOutputs& Session::InferBatch(
    const std::vector<cv::cuda::GpuMat>& inputs) {
  return backend_->InferBatch(inputs);
}

std::unique_ptr<Session> Session::Clone() const {
  return std::unique_ptr<Session>(new Session(config_, backend_->Clone()));
}

std::unique_ptr<Session> CreateSession(RuntimeConfig config) {
  return std::make_unique<Session>(std::move(config));
}

}  // namespace mw::infer
