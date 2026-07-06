#include <stdexcept>
#include <string>
#include <string_view>

#include "mw/infer/runtime/backend.h"
#include "runtime/backends/backend_registry.h"

namespace mw::infer {

namespace {

std::runtime_error DisabledBackendError(BackendKind backend) {
  return std::runtime_error(std::string(BackendName(backend)) +
                            " backend is disabled at build time");
}

}  // namespace

std::string_view BackendName(BackendKind backend) {
  switch (backend) {
    case BackendKind::kOnnxCpu:
      return "onnx_cpu";
    case BackendKind::kOnnxGpu:
      return "onnx_gpu";
    case BackendKind::kTensorRT:
      return "tensorrt";
  }

  return "unknown";
}

std::unique_ptr<IBackend> CreateBackend(const RuntimeConfig& config) {
  ValidateRuntimeConfig(config);

  switch (config.backend) {
    case BackendKind::kOnnxCpu:
#if MW_INFER_WITH_ONNXRUNTIME
      return CreateOnnxCpuBackend(config);
#else
      throw DisabledBackendError(config.backend);
#endif
    case BackendKind::kOnnxGpu:
#if MW_INFER_WITH_ONNXRUNTIME_CUDA
      return CreateOnnxGpuBackend(config);
#else
      throw DisabledBackendError(config.backend);
#endif
    case BackendKind::kTensorRT:
#if MW_INFER_WITH_TENSORRT
      return CreateTensorRTBackend(config);
#else
      throw DisabledBackendError(config.backend);
#endif
  }

  throw std::runtime_error("unknown backend");
}

}  // namespace mw::infer
