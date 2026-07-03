#ifndef MW_INFER_RUNTIME_MODEL_H_
#define MW_INFER_RUNTIME_MODEL_H_

#include <cstddef>
#include <filesystem>
#include <string>

#include "mw/infer/common/memory.h"

namespace mw::infer {

enum class ModelFormat {
  kOnnx,
  kTensorRT,
};

enum class ModelSourceType {
  kPath,
  kMemory,
};

struct ModelSource {
  ModelSourceType type = ModelSourceType::kPath;
  std::filesystem::path path;
  MemoryView memory;
};

struct Model {
  ModelFormat format = ModelFormat::kOnnx;
  ModelSource source;
  std::string name;
};

ModelSource ModelSourceFromPath(std::filesystem::path path);
ModelSource ModelSourceFromMemory(const void* data, std::size_t size_bytes);
ModelFormat InferModelFormatFromPath(const std::filesystem::path& path);
Model ModelFromPath(std::filesystem::path path);

}  // namespace mw::infer

#endif  // MW_INFER_RUNTIME_MODEL_H_
