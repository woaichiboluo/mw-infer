#ifndef MW_INFER_SESSION_H_
#define MW_INFER_SESSION_H_

#include <memory>
#include <vector>

#include "mw/infer/runtime/backend.h"
#include "mw/infer/runtime/runtime_config.h"

namespace mw::infer {

class Session {
 public:
  explicit Session(RuntimeConfig config);

  const RuntimeConfig& config() const;
  BackendKind backend_kind() const;
  const InferOutputs& InferBatch(const std::vector<cv::Mat>& inputs);
  const InferOutputs& InferBatch(const std::vector<cv::cuda::GpuMat>& inputs);
  std::unique_ptr<Session> Clone() const;

 private:
  Session(RuntimeConfig config, std::unique_ptr<IBackend> backend);

  RuntimeConfig config_;
  std::unique_ptr<IBackend> backend_;
};

std::unique_ptr<Session> CreateSession(RuntimeConfig config);

}  // namespace mw::infer

#endif  // MW_INFER_SESSION_H_
