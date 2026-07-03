#include "mw/infer/runtime/runtime_config.h"

#include <gtest/gtest.h>

#include <stdexcept>

#include "mw/infer/common/image.h"
#include "mw/infer/common/tensor.h"

namespace mw::infer {
namespace {

TEST(RuntimeConfigTest, AcceptsOnnxModelForOnnxCpu) {
  RuntimeConfig config;
  config.backend = BackendKind::kOnnxCpu;
  config.model = ModelFromPath("classifier.onnx");

  EXPECT_NO_THROW(ValidateRuntimeConfig(config));
}

TEST(RuntimeConfigTest, AcceptsOnnxModelForOnnxCuda) {
  RuntimeConfig config;
  config.backend = BackendKind::kOnnxCuda;
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

TEST(RuntimeDataModelTest, StoresImageViewMetadata) {
  const unsigned char bytes[] = {0x00, 0x01, 0x02};

  ImageView image;
  image.width = 1;
  image.height = 1;
  image.channels = 3;
  image.format = PixelFormat::kRgb;
  image.data = MemoryView(bytes, sizeof(bytes));
  image.stride_bytes = 3;

  EXPECT_EQ(image.width, 1);
  EXPECT_EQ(image.format, PixelFormat::kRgb);
  EXPECT_FALSE(image.data.empty());
}

TEST(RuntimeDataModelTest, StoresTensorSpecAndDevice) {
  TensorSpec spec;
  spec.name = "input";
  spec.shape = {1, 3, 224, 224};

  Tensor tensor(spec, Device{}, MemoryBuffer{});

  EXPECT_EQ(tensor.spec().name, "input");
  EXPECT_EQ(tensor.shape().size(), 4);
  EXPECT_TRUE(IsCpuDevice(tensor.device()));
}

}  // namespace
}  // namespace mw::infer
