#include "mw/infer/common/model.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>
#include <utility>

namespace mw::infer {

namespace {

std::string LowercaseExtension(const std::filesystem::path& path) {
  std::string extension = path.extension().string();
  std::transform(extension.begin(), extension.end(), extension.begin(),
                 [](unsigned char value) {
                   return static_cast<char>(std::tolower(value));
                 });
  return extension;
}

}  // namespace

ModelSource ModelSourceFromPath(std::filesystem::path path) {
  ModelSource source;
  source.type = ModelSourceType::kPath;
  source.path = std::move(path);
  return source;
}

ModelSource ModelSourceFromMemory(const void* data, std::size_t size_bytes) {
  ModelSource source;
  source.type = ModelSourceType::kMemory;
  source.memory = MemoryView(data, size_bytes);
  return source;
}

ModelFormat InferModelFormatFromPath(const std::filesystem::path& path) {
  const std::string extension = LowercaseExtension(path);
  if (extension == ".onnx") {
    return ModelFormat::kOnnx;
  }
  if (extension == ".engine" || extension == ".plan") {
    return ModelFormat::kTensorRT;
  }

  throw std::invalid_argument("unsupported model file extension");
}

Model ModelFromPath(std::filesystem::path path) {
  Model model;
  model.format = InferModelFormatFromPath(path);
  model.source = ModelSourceFromPath(std::move(path));
  return model;
}

}  // namespace mw::infer
