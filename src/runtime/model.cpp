#include "mw/infer/runtime/model.h"

#include <cctype>
#include <stdexcept>
#include <utility>

namespace mw::infer {

ModelSource ModelSourceFromPath(std::filesystem::path path) {
  if (path.empty()) {
    throw std::invalid_argument("Model path is empty");
  }

  ModelSource source;
  source.kind = ModelSourceKind::kPath;
  source.path = std::move(path);
  return source;
}

ModelSource ModelSourceFromMemory(const void* data, std::size_t bytes,
                                  std::shared_ptr<const void> owner) {
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

ModelFormat InferModelFormatFromPath(const std::filesystem::path& path) {
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

Model ModelFromPath(std::filesystem::path path) {
  Model model;
  model.format = InferModelFormatFromPath(path);
  model.name = path.stem().string();
  model.source = ModelSourceFromPath(std::move(path));
  return model;
}

Model ModelFromMemory(ModelFormat format, const void* data, std::size_t bytes,
                      std::shared_ptr<const void> owner, std::string name) {
  Model model;
  model.format = format;
  model.source = ModelSourceFromMemory(data, bytes, std::move(owner));
  model.name = std::move(name);
  return model;
}

}  // namespace mw::infer
