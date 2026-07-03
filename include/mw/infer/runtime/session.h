#ifndef MW_INFER_SESSION_H_
#define MW_INFER_SESSION_H_

#include <memory>
#include <vector>

#include "mw/infer/common/tensor.h"
#include "mw/infer/runtime/backend.h"
#include "mw/infer/runtime/result.h"
#include "mw/infer/runtime/runtime_config.h"

namespace mw::infer {

class Session {
 public:
  explicit Session(RuntimeConfig config);

  const RuntimeConfig& config() const;
  InferenceResult Predict(const std::vector<Tensor>& inputs);

 private:
  RuntimeConfig config_;
  std::unique_ptr<IBackend> backend_;
};

std::unique_ptr<Session> CreateSession(RuntimeConfig config);

}  // namespace mw::infer

#endif  // MW_INFER_SESSION_H_
