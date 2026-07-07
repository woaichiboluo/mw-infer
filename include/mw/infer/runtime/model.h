#ifndef MW_INFER_RUNTIME_MODEL_H_
#define MW_INFER_RUNTIME_MODEL_H_

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>

namespace mw::infer {

enum class ModelFormat {
  kOnnx,
  kTensorRT,
};

enum class ModelSourceKind {
  kPath,
  kMemory,
};

struct ModelSource {
  ModelSourceKind kind = ModelSourceKind::kPath;
  std::filesystem::path path;
  const void* data = nullptr;
  std::size_t bytes = 0;
  std::shared_ptr<const void> owner;
};

struct Model {
  ModelFormat format = ModelFormat::kOnnx;
  ModelSource source;
  std::string name;
};

ModelSource ModelSourceFromPath(std::filesystem::path path);

ModelSource ModelSourceFromMemory(const void* data, std::size_t bytes,
                                  std::shared_ptr<const void> owner = nullptr);

ModelFormat InferModelFormatFromPath(const std::filesystem::path& path);

Model ModelFromPath(std::filesystem::path path);

Model ModelFromMemory(ModelFormat format, const void* data, std::size_t bytes,
                      std::shared_ptr<const void> owner = nullptr,
                      std::string name = {});

}  // namespace mw::infer

#endif  // MW_INFER_RUNTIME_MODEL_H_
