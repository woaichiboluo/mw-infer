#include "mw/infer/runtime/backend/backend.h"

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mw::infer {

#if defined(MW_INFER_HAS_ONNXRUNTIME_BACKEND)
std::unique_ptr<BackendAdapter> CreateOnnxCpuBackendAdapter();
#endif

#if defined(MW_INFER_HAS_ONNXRUNTIME_CUDA_PROVIDER)
std::unique_ptr<BackendAdapter> CreateOnnxGpuBackendAdapter();
#endif

#if defined(MW_INFER_HAS_TENSORRT_BACKEND)
std::unique_ptr<BackendAdapter> CreateTensorRtBackendAdapter();
#endif

BackendFactory::BackendFactory() {
#if defined(MW_INFER_HAS_ONNXRUNTIME_BACKEND)
  AddAdapter(CreateOnnxCpuBackendAdapter());
#endif
#if defined(MW_INFER_HAS_ONNXRUNTIME_CUDA_PROVIDER)
  AddAdapter(CreateOnnxGpuBackendAdapter());
#endif
#if defined(MW_INFER_HAS_TENSORRT_BACKEND)
  AddAdapter(CreateTensorRtBackendAdapter());
#endif
}

BackendFactory::BackendFactory(
    std::vector<std::unique_ptr<BackendAdapter>> adapters)
    : adapters_(std::move(adapters)) {
  if (adapters_.empty()) {
    throw std::invalid_argument("Backend factory has no adapters");
  }
  for (const auto& adapter : adapters_) {
    if (!adapter) {
      throw std::invalid_argument("Backend factory adapter is null");
    }
  }
}

void BackendFactory::AddAdapter(std::unique_ptr<BackendAdapter> adapter) {
  if (!adapter) {
    throw std::invalid_argument("Backend factory adapter is null");
  }
  adapters_.push_back(std::move(adapter));
}

bool BackendFactory::Supports(const Model& model,
                              Device execution_device) const {
  for (const auto& adapter : adapters_) {
    if (adapter->Supports(model, execution_device)) {
      return true;
    }
  }
  return false;
}

BackendPtr BackendFactory::Create(Model model, Device execution_device,
                                  std::vector<std::string> output_names) const {
  for (const auto& adapter : adapters_) {
    if (adapter->Supports(model, execution_device)) {
      return adapter->Create(std::move(model), execution_device,
                             std::move(output_names));
    }
  }

  throw std::invalid_argument("No backend adapter supports model");
}

BackendPtr CreateBackend(Model model, Device execution_device,
                         std::vector<std::string> output_names) {
  return BackendFactory().Create(std::move(model), execution_device,
                                 std::move(output_names));
}

}  // namespace mw::infer
