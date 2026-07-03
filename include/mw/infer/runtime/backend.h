#ifndef MW_INFER_BACKEND_H_
#define MW_INFER_BACKEND_H_

#include <memory>
#include <string_view>
#include <vector>

#include "mw/infer/common/tensor.h"
#include "mw/infer/runtime/result.h"
#include "mw/infer/runtime/runtime_config.h"

namespace mw::infer {

class IBackend {
 public:
  virtual ~IBackend() = default;

  virtual BackendKind kind() const = 0;
  virtual InferenceResult Forward(const std::vector<Tensor>& inputs) = 0;
};

std::string_view BackendName(BackendKind backend);
std::unique_ptr<IBackend> CreateBackend(const RuntimeConfig& config);

}  // namespace mw::infer

#endif  // MW_INFER_BACKEND_H_
