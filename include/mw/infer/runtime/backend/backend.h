#ifndef MW_INFER_RUNTIME_BACKEND_H_
#define MW_INFER_RUNTIME_BACKEND_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "mw/infer/runtime/backend/model.h"
#include "mw/infer/runtime/tensor/tensor.h"

namespace mw::infer {

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

  IBackend(const IBackend&) = delete;
  IBackend& operator=(const IBackend&) = delete;
  IBackend(IBackend&&) noexcept = default;
  IBackend& operator=(IBackend&&) noexcept = default;

  const Model& model() const { return model_; }
  const ModelInfo& model_info() const { return model_.info; }
  Device execution_device() const { return execution_device_; }
  std::vector<Tensor> Infer(const Tensor& input) {
    return Infer(std::vector<Tensor>{input});
  }
  std::vector<Tensor> Infer(const Tensor& input, TensorAllocator& allocator) {
    return Infer(std::vector<Tensor>{input}, allocator);
  }
  virtual std::vector<Tensor> Infer(const std::vector<Tensor>& inputs) = 0;
  // Returns tensors backed by the requested allocator. The default
  // implementation copies backend-native outputs; backends may override it to
  // allocate outputs directly.
  virtual std::vector<Tensor> Infer(const std::vector<Tensor>& inputs,
                                    TensorAllocator& allocator);

 protected:
  IBackend(Model model, Device execution_device)
      : model_(std::move(model)), execution_device_(execution_device) {}

  Model& mutable_model() { return model_; }

 private:
  Model model_;
  Device execution_device_;
};

using BackendPtr = std::unique_ptr<IBackend>;

class BackendAdapter {
 public:
  virtual ~BackendAdapter() = default;

  virtual bool Supports(const Model& model, Device execution_device) const = 0;
  virtual BackendPtr Create(Model model, Device execution_device,
                            std::vector<std::string> output_names) const = 0;
};

class BackendFactory {
 public:
  BackendFactory();
  explicit BackendFactory(
      std::vector<std::unique_ptr<BackendAdapter>> adapters);

  bool Supports(const Model& model,
                Device execution_device = Device{DeviceType::kCpu, 0}) const;
  BackendPtr Create(Model model,
                    Device execution_device = Device{DeviceType::kCpu, 0},
                    std::vector<std::string> output_names = {}) const;

 private:
  void AddAdapter(std::unique_ptr<BackendAdapter> adapter);

  std::vector<std::unique_ptr<BackendAdapter>> adapters_;
};

BackendPtr CreateBackend(Model model,
                         Device execution_device = Device{DeviceType::kCpu, 0},
                         std::vector<std::string> output_names = {});

}  // namespace mw::infer

#endif  // MW_INFER_RUNTIME_BACKEND_H_
