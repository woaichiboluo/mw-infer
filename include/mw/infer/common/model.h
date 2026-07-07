#ifndef MW_INFER_COMMON_MODEL_H_
#define MW_INFER_COMMON_MODEL_H_

#include <cctype>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

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

inline ModelSource ModelSourceFromPath(std::filesystem::path path) {
  if (path.empty()) {
    throw std::invalid_argument("Model path is empty");
  }

  ModelSource source;
  source.kind = ModelSourceKind::kPath;
  source.path = std::move(path);
  return source;
}

inline ModelSource ModelSourceFromMemory(
    const void* data, std::size_t bytes,
    std::shared_ptr<const void> owner = nullptr) {
  if (data == nullptr) {
    throw std::invalid_argument("Model memory data is null");
  }
  if (bytes == 0) {
    throw std::invalid_argument("Model memory size is zero");
  }

  ModelSource source;
  source.kind = ModelSourceKind::kMemory;
  source.data = data;
  source.bytes = bytes;
  source.owner = std::move(owner);
  return source;
}

inline ModelFormat InferModelFormatFromPath(const std::filesystem::path& path) {
  std::string extension = path.extension().string();
  for (char& character : extension) {
    character =
        static_cast<char>(std::tolower(static_cast<unsigned char>(character)));
  }

  if (extension == ".onnx") {
    return ModelFormat::kOnnx;
  }
  if (extension == ".engine" || extension == ".plan") {
    return ModelFormat::kTensorRT;
  }

  throw std::invalid_argument("Unsupported model file extension: " + extension);
}

inline Model ModelFromPath(std::filesystem::path path) {
  Model model;
  model.format = InferModelFormatFromPath(path);
  model.name = path.stem().string();
  model.source = ModelSourceFromPath(std::move(path));
  return model;
}

inline Model ModelFromMemory(ModelFormat format, const void* data,
                             std::size_t bytes,
                             std::shared_ptr<const void> owner = nullptr,
                             std::string name = {}) {
  Model model;
  model.format = format;
  model.source = ModelSourceFromMemory(data, bytes, std::move(owner));
  model.name = std::move(name);
  return model;
}

}  // namespace mw::infer

#endif  // MW_INFER_COMMON_MODEL_H_
