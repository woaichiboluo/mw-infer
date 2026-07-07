#ifndef MW_INFER_RUNTIME_BACKEND_H_
#define MW_INFER_RUNTIME_BACKEND_H_

#include <memory>
#include <string>
#include <vector>

#include "mw/infer/common/tensor.h"

namespace mw::infer {

struct TensorInfo {
  std::string name;
  DataType data_type = DataType::kUnknown;
  std::vector<int64_t> shape;
};

struct ModelInfo {
  std::vector<TensorInfo> inputs;
  std::vector<TensorInfo> outputs;
};

inline bool HasDynamicShape(const TensorInfo& info) {
  for (int64_t dim : info.shape) {
    if (dim <= 0) {
      return true;
    }
  }
  return false;
}

class IBackend {
 public:
  virtual ~IBackend() = default;

  virtual const ModelInfo& model_info() const = 0;
  virtual std::vector<Tensor> Infer(const std::vector<Tensor>& inputs) = 0;
};

using BackendPtr = std::unique_ptr<IBackend>;

}  // namespace mw::infer

#endif  // MW_INFER_RUNTIME_BACKEND_H_
