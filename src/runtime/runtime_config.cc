#include "mw/infer/runtime/runtime_config.h"

#include <stdexcept>

namespace mw::infer {

bool IsTensorRTEnginePath(const std::filesystem::path& path) {
  try {
    return InferModelFormatFromPath(path) == ModelFormat::kTensorRT;
  } catch (const std::invalid_argument&) {
    return false;
  }
}

bool IsOnnxModelPath(const std::filesystem::path& path) {
  try {
    return InferModelFormatFromPath(path) == ModelFormat::kOnnx;
  } catch (const std::invalid_argument&) {
    return false;
  }
}

void ValidateRuntimeConfig(const RuntimeConfig& config) {
  if (config.model.source.type == ModelSourceType::kPath &&
      config.model.source.path.empty()) {
    throw std::invalid_argument("RuntimeConfig::model path must not be empty");
  }

  if (config.model.source.type == ModelSourceType::kMemory &&
      config.model.source.memory.empty()) {
    throw std::invalid_argument(
        "RuntimeConfig::model memory must not be empty");
  }

  switch (config.backend) {
    case BackendKind::kOnnxCpu:
    case BackendKind::kOnnxCuda:
      if (config.model.format != ModelFormat::kOnnx) {
        throw std::invalid_argument(
            "ONNX Runtime backends require a .onnx model");
      }
      return;
    case BackendKind::kTensorRT:
      if (config.model.format != ModelFormat::kTensorRT) {
        throw std::invalid_argument(
            "TensorRT backend requires a serialized engine file (.engine or "
            ".plan)");
      }
      return;
  }

  throw std::invalid_argument("unknown backend");
}

}  // namespace mw::infer
