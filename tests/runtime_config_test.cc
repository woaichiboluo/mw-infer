#include "mw/infer/runtime/runtime_config.h"

#include <gtest/gtest.h>

#include <stdexcept>

namespace mw::infer {
namespace {

TEST(RuntimeConfigTest, AcceptsOnnxModelForOnnxCpu) {
  RuntimeConfig config;
  config.backend = BackendKind::kOnnxCpu;
  config.model = ModelFromPath("classifier.onnx");

  EXPECT_NO_THROW(ValidateRuntimeConfig(config));
}

TEST(RuntimeConfigTest, AcceptsOnnxModelForOnnxGpu) {
  RuntimeConfig config;
  config.backend = BackendKind::kOnnxGpu;
  config.model = ModelFromPath("detector.ONNX");

  EXPECT_NO_THROW(ValidateRuntimeConfig(config));
}

TEST(RuntimeConfigTest, AcceptsTensorRTSerializedEngineExtensions) {
  RuntimeConfig config;
  config.backend = BackendKind::kTensorRT;

  config.model = ModelFromPath("detector.engine");
  EXPECT_EQ(config.model.format, ModelFormat::kTensorRT);
  EXPECT_NO_THROW(ValidateRuntimeConfig(config));

  config.model = ModelFromPath("detector.plan");
  EXPECT_EQ(config.model.format, ModelFormat::kTensorRT);
  EXPECT_NO_THROW(ValidateRuntimeConfig(config));
}

TEST(RuntimeConfigTest, RejectsEmptyModelPath) {
  RuntimeConfig config;

  EXPECT_THROW(ValidateRuntimeConfig(config), std::invalid_argument);
}

TEST(RuntimeConfigTest, RejectsMismatchedModelExtension) {
  RuntimeConfig config;
  config.backend = BackendKind::kTensorRT;
  config.model = ModelFromPath("classifier.onnx");

  EXPECT_THROW(ValidateRuntimeConfig(config), std::invalid_argument);
}

TEST(RuntimeConfigTest, RejectsUnknownModelExtension) {
  EXPECT_FALSE(IsOnnxModelPath("model.bin"));
  EXPECT_FALSE(IsTensorRTEnginePath("model.bin"));
}

TEST(RuntimeConfigTest, AcceptsMemoryBackedOnnxModel) {
  const unsigned char bytes[] = {0x01, 0x02, 0x03};

  RuntimeConfig config;
  config.backend = BackendKind::kOnnxCpu;
  config.model.format = ModelFormat::kOnnx;
  config.model.source = ModelSourceFromMemory(bytes, sizeof(bytes));

  EXPECT_NO_THROW(ValidateRuntimeConfig(config));
}

}  // namespace
}  // namespace mw::infer
