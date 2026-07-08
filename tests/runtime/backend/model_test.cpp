#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "mw/infer/runtime/backend/backend.h"

namespace mw::infer {
namespace {

class FakeBackend final : public IBackend {
 public:
  FakeBackend(Model model, Device execution_device)
      : IBackend(std::move(model), execution_device) {
    TensorInfo input;
    input.name = "input";
    input.data_type = DataType::kFloat32;
    input.shape = {1, 3, 224, 224};
    mutable_model().info.inputs.push_back(std::move(input));

    TensorInfo output;
    output.name = "output";
    output.data_type = DataType::kFloat32;
    output.shape = {1, 1000};
    mutable_model().info.outputs.push_back(std::move(output));

    TensorShapeRange input_range;
    input_range.name = "input";
    input_range.min_shape = {1, 3, 224, 224};
    input_range.opt_shape = {4, 3, 224, 224};
    input_range.max_shape = {8, 3, 224, 224};

    ModelProfile profile;
    profile.name = "profile0";
    profile.inputs.push_back(std::move(input_range));
    mutable_model().info.profiles.push_back(std::move(profile));
  }

  std::vector<Tensor> Infer(const std::vector<Tensor>& inputs) override {
    last_input_count_ = inputs.size();
    last_input_name_ = inputs.empty() ? "" : inputs.front().name();
    return {};
  }

  std::size_t last_input_count() const { return last_input_count_; }
  const std::string& last_input_name() const { return last_input_name_; }

 private:
  std::size_t last_input_count_ = 0;
  std::string last_input_name_;
};

class FakeBackendAdapter final : public BackendAdapter {
 public:
  bool Supports(const Model& model, Device) const override {
    return model.format == ModelFormat::kOnnx;
  }

  BackendPtr Create(Model model, Device execution_device,
                    std::vector<std::string>) const override {
    return std::make_unique<FakeBackend>(std::move(model), execution_device);
  }
};

TEST(ModelTest, CreatesModelFromPath) {
  Model model = ModelFromPath("models/add.ONNX");

  EXPECT_EQ(model.format, ModelFormat::kOnnx);
  EXPECT_EQ(model.name, "add");
  EXPECT_EQ(model.source.kind, ModelSourceKind::kPath);
  EXPECT_EQ(model.source.path, "models/add.ONNX");
  EXPECT_TRUE(model.info.inputs.empty());
  EXPECT_TRUE(model.info.outputs.empty());
  EXPECT_TRUE(model.info.profiles.empty());
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
  EXPECT_TRUE(model.info.inputs.empty());
  EXPECT_TRUE(model.info.outputs.empty());
  EXPECT_TRUE(model.info.profiles.empty());
}

TEST(ModelTest, BackendOwnsActiveModelInfo) {
  Model model = ModelFromMemory(ModelFormat::kOnnx, "data", 4, nullptr, "fake");

  FakeBackend backend(std::move(model), Device{DeviceType::kCuda, 1});
  const ModelInfo& info = backend.model_info();

  EXPECT_EQ(backend.model().name, "fake");
  EXPECT_EQ(backend.execution_device().type, DeviceType::kCuda);
  EXPECT_EQ(backend.execution_device().id, 1);
  EXPECT_EQ(&info, &backend.model().info);
  ASSERT_EQ(info.inputs.size(), 1U);
  EXPECT_EQ(info.inputs[0].name, "input");
  ASSERT_EQ(info.outputs.size(), 1U);
  EXPECT_EQ(info.outputs[0].shape, std::vector<int64_t>({1, 1000}));
  ASSERT_EQ(info.profiles.size(), 1U);
  EXPECT_EQ(info.profiles[0].name, "profile0");
  ASSERT_EQ(info.profiles[0].inputs.size(), 1U);
  EXPECT_EQ(info.profiles[0].inputs[0].name, "input");
  EXPECT_EQ(info.profiles[0].inputs[0].min_shape,
            std::vector<int64_t>({1, 3, 224, 224}));
  EXPECT_EQ(info.profiles[0].inputs[0].opt_shape,
            std::vector<int64_t>({4, 3, 224, 224}));
  EXPECT_EQ(info.profiles[0].inputs[0].max_shape,
            std::vector<int64_t>({8, 3, 224, 224}));
}

TEST(ModelTest, BackendFactoryUsesRegisteredAdapters) {
  std::vector<std::unique_ptr<BackendAdapter>> adapters;
  adapters.push_back(std::make_unique<FakeBackendAdapter>());
  BackendFactory factory(std::move(adapters));

  Model model = ModelFromMemory(ModelFormat::kOnnx, "data", 4, nullptr, "fake");

  const Device execution_device{DeviceType::kCuda, 2};
  ASSERT_TRUE(factory.Supports(model, execution_device));
  BackendPtr backend = factory.Create(std::move(model), execution_device);

  EXPECT_EQ(backend->model().name, "fake");
  EXPECT_EQ(backend->execution_device().type, DeviceType::kCuda);
  EXPECT_EQ(backend->execution_device().id, 2);
  ASSERT_EQ(backend->model_info().inputs.size(), 1U);
  EXPECT_EQ(backend->model_info().inputs[0].name, "input");
}

TEST(ModelTest, SingleTensorInferForwardsToMultiInputInfer) {
  Model model = ModelFromMemory(ModelFormat::kOnnx, "data", 4, nullptr, "fake");
  FakeBackend backend(std::move(model), Device{DeviceType::kCpu, 0});
  IBackend& base = backend;

  float input_value = 1.0F;
  TensorDesc desc;
  desc.info.name = "input";
  desc.info.data_type = DataType::kFloat32;
  desc.info.shape = {1};
  desc.device = Device{DeviceType::kCpu, 0};
  Tensor input = Tensor::FromExternal(desc, &input_value, sizeof(input_value));

  static_cast<void>(base.Infer(input));

  EXPECT_EQ(backend.last_input_count(), 1U);
  EXPECT_EQ(backend.last_input_name(), "input");
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
