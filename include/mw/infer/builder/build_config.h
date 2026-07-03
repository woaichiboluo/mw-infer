#ifndef MW_INFER_BUILDER_BUILD_CONFIG_H_
#define MW_INFER_BUILDER_BUILD_CONFIG_H_

#include <filesystem>

#include "mw/infer/common/model.h"

namespace mw::infer::builder {

enum class TargetFormat {
  kTensorRT,
};

struct BuildConfig {
  Model input_model;
  std::filesystem::path output_model_path;
  TargetFormat target_format = TargetFormat::kTensorRT;
  bool enable_fp16 = false;
  bool enable_int8 = false;
};

void ValidateBuildConfig(const BuildConfig& config);

}  // namespace mw::infer::builder

#endif  // MW_INFER_BUILDER_BUILD_CONFIG_H_
