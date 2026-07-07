#include "mw/infer/runtime/backend/onnx_runtime_backend.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mw::infer {
namespace {

Model AddModel() {
  static constexpr std::array<uint8_t, 111> kModel = {
      8,   8,   18,  13,  109, 119, 45,  105, 110, 102, 101, 114, 45,  116,
      101, 115, 116, 58,  86,  10,  27,  10,  5,   105, 110, 112, 117, 116,
      10,  5,   105, 110, 112, 117, 116, 18,  6,   111, 117, 116, 112, 117,
      116, 34,  3,   65,  100, 100, 18,  12,  109, 119, 95,  105, 110, 102,
      101, 114, 95,  97,  100, 100, 90,  19,  10,  5,   105, 110, 112, 117,
      116, 18,  10,  10,  8,   8,   1,   18,  4,   10,  2,   8,   3,   98,
      20,  10,  6,   111, 117, 116, 112, 117, 116, 18,  10,  10,  8,   8,
      1,   18,  4,   10,  2,   8,   3,   66,  4,   10,  0,   16,  13};
  return ModelFromMemory(ModelFormat::kOnnx, kModel.data(), kModel.size(),
                         nullptr, "add");
}

Tensor MakeInputTensor(std::vector<float>* buffer) {
  TensorDesc desc;
  desc.data_type = DataType::kFloat32;
  desc.shape = {3};
  desc.device = Device{DeviceType::kCpu, 0};
  return Tensor::FromExternal(desc, buffer->data(),
                              buffer->size() * sizeof(float));
}

void ExpectAddOutput(const Tensor& output) {
  ASSERT_EQ(output.name(), "output");
  ASSERT_EQ(output.data_type(), DataType::kFloat32);
  ASSERT_EQ(output.shape(), std::vector<int64_t>({3}));
  ASSERT_EQ(output.device().type, DeviceType::kCpu);
  ASSERT_EQ(output.bytes(), 3U * sizeof(float));

  const auto* values = static_cast<const float*>(output.data());
  EXPECT_FLOAT_EQ(values[0], 2.0F);
  EXPECT_FLOAT_EQ(values[1], 4.0F);
  EXPECT_FLOAT_EQ(values[2], 6.0F);
}

TEST(OnnxCpuBackendTest, RunsEmbeddedAddModel) {
  BackendPtr backend = CreateOnnxCpuBackend(AddModel());
  const ModelInfo& info = backend->model_info();

  ASSERT_EQ(info.inputs.size(), 1U);
  EXPECT_EQ(info.inputs[0].name, "input");
  EXPECT_EQ(info.inputs[0].data_type, DataType::kFloat32);
  EXPECT_EQ(info.inputs[0].shape, std::vector<int64_t>({3}));
  ASSERT_EQ(info.outputs.size(), 1U);
  EXPECT_EQ(info.outputs[0].name, "output");

  std::vector<float> input = {1.0F, 2.0F, 3.0F};
  std::vector<Tensor> outputs = backend->Infer({MakeInputTensor(&input)});

  ASSERT_EQ(outputs.size(), 1U);
  ExpectAddOutput(outputs[0]);
}

TEST(OnnxGpuBackendTest, RunsWithCpuOutput) {
  if (!IsOnnxGpuBackendAvailable()) {
    GTEST_SKIP() << "ONNX Runtime CUDA provider is unavailable";
  }

  BackendPtr backend = CreateOnnxGpuBackend(AddModel());
  std::vector<float> input = {1.0F, 2.0F, 3.0F};
  std::vector<Tensor> outputs = backend->Infer({MakeInputTensor(&input)});

  ASSERT_EQ(outputs.size(), 1U);
  ExpectAddOutput(outputs[0]);
}

TEST(OnnxGpuBackendTest, CanReturnCudaOutput) {
  if (!IsOnnxGpuBackendAvailable()) {
    GTEST_SKIP() << "ONNX Runtime CUDA provider is unavailable";
  }

  OnnxGpuBackendOptions options;
  options.model = AddModel();
  options.device_id = 0;
  options.output_device = Device{DeviceType::kCuda, 0};
  OnnxGpuBackend backend(std::move(options));

  std::vector<float> input = {1.0F, 2.0F, 3.0F};
  std::vector<Tensor> outputs = backend.Infer({MakeInputTensor(&input)});

  ASSERT_EQ(outputs.size(), 1U);
  EXPECT_EQ(outputs[0].name(), "output");
  EXPECT_EQ(outputs[0].data_type(), DataType::kFloat32);
  EXPECT_EQ(outputs[0].shape(), std::vector<int64_t>({3}));
  EXPECT_EQ(outputs[0].device().type, DeviceType::kCuda);
  EXPECT_EQ(outputs[0].device().id, 0);
  EXPECT_EQ(outputs[0].bytes(), 3U * sizeof(float));
  EXPECT_NE(outputs[0].data(), nullptr);
}

TEST(OnnxGpuBackendTest, ThrowsWhenCudaProviderIsUnavailable) {
  if (IsOnnxGpuBackendAvailable()) {
    GTEST_SKIP() << "ONNX Runtime CUDA provider is available";
  }

  OnnxGpuBackendOptions options;
  options.model = AddModel();
  EXPECT_THROW(static_cast<void>(OnnxGpuBackend(std::move(options))),
               std::runtime_error);
}

}  // namespace
}  // namespace mw::infer
