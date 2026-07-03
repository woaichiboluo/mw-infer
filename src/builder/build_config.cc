#include "mw/infer/builder/build_config.h"

#include <stdexcept>

namespace mw::infer::builder {

void ValidateBuildConfig(const BuildConfig& config) {
  if (config.input_model.format != ModelFormat::kOnnx) {
    throw std::invalid_argument(
        "BuildConfig::input_model must be an ONNX model");
  }

  if (config.input_model.source.type == ModelSourceType::kPath &&
      config.input_model.source.path.empty()) {
    throw std::invalid_argument(
        "BuildConfig::input_model path must not be empty");
  }

  if (config.input_model.source.type == ModelSourceType::kMemory &&
      config.input_model.source.memory.empty()) {
    throw std::invalid_argument(
        "BuildConfig::input_model memory must not be empty");
  }

  if (config.output_model_path.empty()) {
    throw std::invalid_argument(
        "BuildConfig::output_model_path must not be empty");
  }

  if (config.target_format == TargetFormat::kTensorRT &&
      config.output_model_path.extension() != ".engine" &&
      config.output_model_path.extension() != ".plan") {
    throw std::invalid_argument(
        "TensorRT builder output must be a serialized engine file (.engine or "
        ".plan)");
  }
}

}  // namespace mw::infer::builder
