#include "mw/infer/runtime/model.h"

#include <gtest/gtest.h>

#include <array>
#include <memory>
#include <stdexcept>

namespace mw::infer {
namespace {

TEST(ModelTest, CreatesModelFromPath) {
  Model model = ModelFromPath("models/add.ONNX");

  EXPECT_EQ(model.format, ModelFormat::kOnnx);
  EXPECT_EQ(model.name, "add");
  EXPECT_EQ(model.source.kind, ModelSourceKind::kPath);
  EXPECT_EQ(model.source.path, "models/add.ONNX");
}

TEST(ModelTest, CreatesModelFromMemory) {
  const auto owner = std::make_shared<std::array<unsigned char, 3>>(
      std::array<unsigned char, 3>{1, 2, 3});
  Model model = ModelFromMemory(ModelFormat::kTensorRT, owner->data(),
                                owner->size(), owner, "engine");

  EXPECT_EQ(model.format, ModelFormat::kTensorRT);
  EXPECT_EQ(model.name, "engine");
  EXPECT_EQ(model.source.kind, ModelSourceKind::kMemory);
  EXPECT_EQ(model.source.data, owner->data());
  EXPECT_EQ(model.source.bytes, owner->size());
  EXPECT_EQ(model.source.owner, owner);
}

TEST(ModelTest, RejectsInvalidSources) {
  EXPECT_THROW(static_cast<void>(ModelSourceFromPath({})),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(ModelSourceFromMemory(nullptr, 1)),
               std::invalid_argument);

  const int data = 0;
  EXPECT_THROW(static_cast<void>(ModelSourceFromMemory(&data, 0)),
               std::invalid_argument);
}

TEST(ModelTest, RejectsUnsupportedPathExtension) {
  EXPECT_THROW(static_cast<void>(InferModelFormatFromPath("model.pb")),
               std::invalid_argument);
}

}  // namespace
}  // namespace mw::infer
