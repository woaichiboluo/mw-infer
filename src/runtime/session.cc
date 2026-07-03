#include "mw/infer/runtime/session.h"

#include <utility>

namespace mw::infer {

Session::Session(RuntimeConfig config)
    : config_(std::move(config)), backend_(CreateBackend(config_)) {}

const RuntimeConfig& Session::config() const { return config_; }

InferenceResult Session::Predict(const std::vector<Tensor>& inputs) {
  return backend_->Forward(inputs);
}

std::unique_ptr<Session> CreateSession(RuntimeConfig config) {
  return std::make_unique<Session>(std::move(config));
}

}  // namespace mw::infer
