#ifndef MW_INFER_RUNTIME_BACKEND_MODEL_H_
#define MW_INFER_RUNTIME_BACKEND_MODEL_H_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "mw/infer/runtime/tensor/tensor.h"

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

struct TensorShapeRange {
  std::string name;
  std::vector<int64_t> min_shape;
  std::vector<int64_t> opt_shape;
  std::vector<int64_t> max_shape;
};

struct ModelProfile {
  std::string name;
  std::vector<TensorShapeRange> inputs;
};

struct ModelInfo {
  std::vector<TensorInfo> inputs;
  std::vector<TensorInfo> outputs;
  std::vector<ModelProfile> profiles;
};

struct Model {
  ModelFormat format = ModelFormat::kOnnx;
  ModelSource source;
  std::string name;
  ModelInfo info;
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

#endif  // MW_INFER_RUNTIME_BACKEND_MODEL_H_
