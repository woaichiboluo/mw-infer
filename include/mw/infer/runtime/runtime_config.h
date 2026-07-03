#ifndef MW_INFER_RUNTIME_CONFIG_H_
#define MW_INFER_RUNTIME_CONFIG_H_

#include <filesystem>

#include "mw/infer/common/model.h"

namespace mw::infer {

enum class BackendKind {
  kOnnxCpu,
  kOnnxCuda,
  kTensorRT,
};

struct RuntimeConfig {
  BackendKind backend = BackendKind::kOnnxCpu;
  Model model;
};

void ValidateRuntimeConfig(const RuntimeConfig& config);
bool IsTensorRTEnginePath(const std::filesystem::path& path);
bool IsOnnxModelPath(const std::filesystem::path& path);

}  // namespace mw::infer

#endif  // MW_INFER_RUNTIME_CONFIG_H_
