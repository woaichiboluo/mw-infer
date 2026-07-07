#ifndef MW_INFER_RUNTIME_BACKENDS_ONNX_RUNTIME_BACKEND_H_
#define MW_INFER_RUNTIME_BACKENDS_ONNX_RUNTIME_BACKEND_H_

#include <memory>
#include <string>
#include <vector>

#include "mw/infer/common/model.h"
#include "mw/infer/common/tensor.h"
#include "mw/infer/runtime/backend.h"

namespace mw::infer {

struct OnnxCpuBackendOptions {
  Model model;
  std::vector<std::string> output_names;
};

struct OnnxGpuBackendOptions {
  Model model;
  int device_id = 0;
  Device output_device = Device{DeviceType::kCpu, 0};
  std::vector<std::string> output_names;
};

class OnnxCpuBackend final : public IBackend {
 public:
  explicit OnnxCpuBackend(OnnxCpuBackendOptions options);
  ~OnnxCpuBackend() override;

  OnnxCpuBackend(const OnnxCpuBackend&) = delete;
  OnnxCpuBackend& operator=(const OnnxCpuBackend&) = delete;
  OnnxCpuBackend(OnnxCpuBackend&&) noexcept;
  OnnxCpuBackend& operator=(OnnxCpuBackend&&) noexcept;

  const ModelInfo& model_info() const override;
  std::vector<Tensor> Infer(const std::vector<Tensor>& inputs) override;

 private:
  class Impl;

  std::unique_ptr<Impl> impl_;
};

class OnnxGpuBackend final : public IBackend {
 public:
  explicit OnnxGpuBackend(OnnxGpuBackendOptions options);
  ~OnnxGpuBackend() override;

  OnnxGpuBackend(const OnnxGpuBackend&) = delete;
  OnnxGpuBackend& operator=(const OnnxGpuBackend&) = delete;
  OnnxGpuBackend(OnnxGpuBackend&&) noexcept;
  OnnxGpuBackend& operator=(OnnxGpuBackend&&) noexcept;

  const ModelInfo& model_info() const override;
  std::vector<Tensor> Infer(const std::vector<Tensor>& inputs) override;

 private:
  class Impl;

  std::unique_ptr<Impl> impl_;
};

bool IsOnnxGpuBackendAvailable();
BackendPtr CreateOnnxCpuBackend(Model model);
BackendPtr CreateOnnxGpuBackend(Model model, int device_id = 0);

}  // namespace mw::infer

#endif  // MW_INFER_RUNTIME_BACKENDS_ONNX_RUNTIME_BACKEND_H_
