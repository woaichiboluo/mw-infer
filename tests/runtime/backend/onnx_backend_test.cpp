#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(MW_INFER_HAS_CUDA_RUNTIME)
#include <cuda_runtime_api.h>
#endif

#include "mw/infer/runtime/backend/backend.h"
#include "mw/infer/runtime/tensor/tensor_allocator.h"

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

Model SubModel() {
  // ONNX official backend test: onnx/backend/test/data/node/test_sub/model.onnx
  static constexpr std::array<uint8_t, 125> kModel = {
      8,   7,   18, 12,  98,  97, 99,  107, 101, 110, 100, 45,  116, 101,
      115, 116, 58, 101, 10,  14, 10,  1,   120, 10,  1,   121, 18,  1,
      122, 34,  3,  83,  117, 98, 18,  8,   116, 101, 115, 116, 95,  115,
      117, 98,  90, 23,  10,  1,  120, 18,  18,  10,  16,  8,   1,   18,
      12,  10,  2,  8,   3,   10, 2,   8,   4,   10,  2,   8,   5,   90,
      23,  10,  1,  121, 18,  18, 10,  16,  8,   1,   18,  12,  10,  2,
      8,   3,   10, 2,   8,   4,  10,  2,   8,   5,   98,  23,  10,  1,
      122, 18,  18, 10,  16,  8,  1,   18,  12,  10,  2,   8,   3,   10,
      2,   8,   4,  10,  2,   8,  5,   66,  4,   10,  0,   16,  14};
  return ModelFromMemory(ModelFormat::kOnnx, kModel.data(), kModel.size(),
                         nullptr, "official_test_sub");
}

Model SplitModel() {
  // ONNX official backend test:
  // onnx/backend/test/data/node/test_split_equal_parts_1d_opset13/model.onnx
  static constexpr std::array<uint8_t, 212> kModel = {
      8,   7,   18,  12,  98,  97,  99,  107, 101, 110, 100, 45,  116, 101, 115,
      116, 58,  187, 1,   10,  57,  10,  5,   105, 110, 112, 117, 116, 18,  8,
      111, 117, 116, 112, 117, 116, 95,  49,  18,  8,   111, 117, 116, 112, 117,
      116, 95,  50,  18,  8,   111, 117, 116, 112, 117, 116, 95,  51,  34,  5,
      83,  112, 108, 105, 116, 42,  11,  10,  4,   97,  120, 105, 115, 24,  0,
      160, 1,   2,   18,  33,  116, 101, 115, 116, 95,  115, 112, 108, 105, 116,
      95,  101, 113, 117, 97,  108, 95,  112, 97,  114, 116, 115, 95,  49,  100,
      95,  111, 112, 115, 101, 116, 49,  51,  90,  19,  10,  5,   105, 110, 112,
      117, 116, 18,  10,  10,  8,   8,   1,   18,  4,   10,  2,   8,   6,   98,
      22,  10,  8,   111, 117, 116, 112, 117, 116, 95,  49,  18,  10,  10,  8,
      8,   1,   18,  4,   10,  2,   8,   2,   98,  22,  10,  8,   111, 117, 116,
      112, 117, 116, 95,  50,  18,  10,  10,  8,   8,   1,   18,  4,   10,  2,
      8,   2,   98,  22,  10,  8,   111, 117, 116, 112, 117, 116, 95,  51,  18,
      10,  10,  8,   8,   1,   18,  4,   10,  2,   8,   2,   66,  4,   10,  0,
      16,  13};
  return ModelFromMemory(ModelFormat::kOnnx, kModel.data(), kModel.size(),
                         nullptr, "official_test_split_equal_parts_1d");
}

Model FixedBatchAddModel() {
  static constexpr std::array<uint8_t, 158> kModel = {
      8,   8,   18,  14,  109, 119, 45,  105, 110, 102, 101, 114, 45,  116, 101,
      115, 116, 115, 58,  133, 1,   10,  43,  10,  3,   108, 104, 115, 10,  3,
      114, 104, 115, 18,  3,   115, 117, 109, 26,  21,  98,  97,  116, 99,  104,
      101, 100, 95,  97,  100, 100, 95,  102, 105, 120, 101, 100, 95,  97,  100,
      100, 34,  3,   65,  100, 100, 18,  17,  98,  97,  116, 99,  104, 101, 100,
      95,  97,  100, 100, 95,  102, 105, 120, 101, 100, 90,  21,  10,  3,   108,
      104, 115, 18,  14,  10,  12,  8,   1,   18,  8,   10,  2,   8,   2,   10,
      2,   8,   3,   90,  21,  10,  3,   114, 104, 115, 18,  14,  10,  12,  8,
      1,   18,  8,   10,  2,   8,   2,   10,  2,   8,   3,   98,  21,  10,  3,
      115, 117, 109, 18,  14,  10,  12,  8,   1,   18,  8,   10,  2,   8,   2,
      10,  2,   8,   3,   66,  2,   16,  13};
  return ModelFromMemory(ModelFormat::kOnnx, kModel.data(), kModel.size(),
                         nullptr, "fixed_batch_add");
}

Model DynamicBatchAddModel() {
  static constexpr std::array<uint8_t, 177> kModel = {
      8,   8,   18,  14,  109, 119, 45,  105, 110, 102, 101, 114, 45,  116, 101,
      115, 116, 115, 58,  152, 1,   10,  45,  10,  3,   108, 104, 115, 10,  3,
      114, 104, 115, 18,  3,   115, 117, 109, 26,  23,  98,  97,  116, 99,  104,
      101, 100, 95,  97,  100, 100, 95,  100, 121, 110, 97,  109, 105, 99,  95,
      97,  100, 100, 34,  3,   65,  100, 100, 18,  19,  98,  97,  116, 99,  104,
      101, 100, 95,  97,  100, 100, 95,  100, 121, 110, 97,  109, 105, 99,  90,
      26,  10,  3,   108, 104, 115, 18,  19,  10,  17,  8,   1,   18,  13,  10,
      7,   18,  5,   98,  97,  116, 99,  104, 10,  2,   8,   3,   90,  26,  10,
      3,   114, 104, 115, 18,  19,  10,  17,  8,   1,   18,  13,  10,  7,   18,
      5,   98,  97,  116, 99,  104, 10,  2,   8,   3,   98,  26,  10,  3,   115,
      117, 109, 18,  19,  10,  17,  8,   1,   18,  13,  10,  7,   18,  5,   98,
      97,  116, 99,  104, 10,  2,   8,   3,   66,  2,   16,  13};
  return ModelFromMemory(ModelFormat::kOnnx, kModel.data(), kModel.size(),
                         nullptr, "dynamic_batch_add");
}

Tensor MakeInputTensor(std::vector<float>* buffer) {
  TensorDesc desc;
  desc.info.data_type = DataType::kFloat32;
  desc.info.shape = {3};
  desc.device = Device{DeviceType::kCpu, 0};
  return Tensor::FromExternal(desc, buffer->data(),
                              buffer->size() * sizeof(float));
}

Tensor MakeNamedInputTensor(std::string name, std::vector<int64_t> shape,
                            std::vector<float>* buffer) {
  TensorDesc desc;
  desc.info.name = std::move(name);
  desc.info.data_type = DataType::kFloat32;
  desc.info.shape = std::move(shape);
  desc.device = Device{DeviceType::kCpu, 0};
  return Tensor::FromExternal(desc, buffer->data(),
                              buffer->size() * sizeof(float));
}

Device CudaDevice() { return Device{DeviceType::kCuda, 0}; }

std::vector<float> TensorValues(const Tensor& tensor) {
  if (tensor.data_type() != DataType::kFloat32) {
    throw std::invalid_argument("Expected a float32 tensor");
  }

  std::vector<float> values(tensor.bytes() / sizeof(float));
  if (tensor.device().type == DeviceType::kCpu) {
    std::memcpy(values.data(), tensor.data(), tensor.bytes());
    return values;
  }

#if defined(MW_INFER_HAS_CUDA_RUNTIME)
  const cudaError_t status = cudaMemcpy(values.data(), tensor.data(),
                                        tensor.bytes(), cudaMemcpyDeviceToHost);
  if (status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(status));
  }
  return values;
#else
  throw std::runtime_error("CUDA runtime is unavailable in this build");
#endif
}

void ExpectFloatValues(const Tensor& tensor,
                       std::initializer_list<float> expected) {
  const std::vector<float> values = TensorValues(tensor);
  ASSERT_EQ(values.size(), expected.size());

  std::size_t index = 0;
  for (float expected_value : expected) {
    EXPECT_FLOAT_EQ(values[index], expected_value);
    ++index;
  }
}

void ExpectFloatValues(const Tensor& tensor,
                       const std::vector<float>& expected) {
  const std::vector<float> values = TensorValues(tensor);
  ASSERT_EQ(values.size(), expected.size());
  for (std::size_t index = 0; index < values.size(); ++index) {
    EXPECT_FLOAT_EQ(values[index], expected[index]);
  }
}

std::vector<float> Sequence(std::size_t count, float offset) {
  std::vector<float> values(count);
  for (std::size_t index = 0; index < values.size(); ++index) {
    values[index] = static_cast<float>(index) + offset;
  }
  return values;
}

std::vector<float> Difference(const std::vector<float>& lhs,
                              const std::vector<float>& rhs) {
  std::vector<float> values(lhs.size());
  for (std::size_t index = 0; index < values.size(); ++index) {
    values[index] = lhs[index] - rhs[index];
  }
  return values;
}

std::vector<float> Sum(const std::vector<float>& lhs,
                       const std::vector<float>& rhs) {
  std::vector<float> values(lhs.size());
  for (std::size_t index = 0; index < values.size(); ++index) {
    values[index] = lhs[index] + rhs[index];
  }
  return values;
}

void ExpectBatchedAddOutput(IBackend* backend, int64_t batch,
                            DeviceType expected_device) {
  std::vector<float> lhs = Sequence(static_cast<std::size_t>(batch * 3), 1.0F);
  std::vector<float> rhs =
      Sequence(static_cast<std::size_t>(batch * 3), 100.0F);
  Tensor lhs_tensor = MakeNamedInputTensor("lhs", {batch, 3}, &lhs);
  Tensor rhs_tensor = MakeNamedInputTensor("rhs", {batch, 3}, &rhs);

  std::vector<Tensor> outputs = backend->Infer({rhs_tensor, lhs_tensor});

  ASSERT_EQ(outputs.size(), 1U);
  EXPECT_EQ(outputs[0].name(), "sum");
  EXPECT_EQ(outputs[0].shape(), std::vector<int64_t>({batch, 3}));
  EXPECT_EQ(outputs[0].device().type, expected_device);
  if (expected_device == DeviceType::kCuda) {
    EXPECT_EQ(outputs[0].device().id, 0);
  }
  ExpectFloatValues(outputs[0], Sum(lhs, rhs));
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
  BackendPtr backend = CreateBackend(AddModel());
  const ModelInfo& info = backend->model().info;

  EXPECT_EQ(backend->model().name, "add");
  EXPECT_EQ(&backend->model_info(), &info);
  ASSERT_EQ(info.inputs.size(), 1U);
  EXPECT_EQ(info.inputs[0].name, "input");
  EXPECT_EQ(info.inputs[0].data_type, DataType::kFloat32);
  EXPECT_EQ(info.inputs[0].shape, std::vector<int64_t>({3}));
  ASSERT_EQ(info.outputs.size(), 1U);
  EXPECT_EQ(info.outputs[0].name, "output");
  EXPECT_TRUE(info.profiles.empty());

  std::vector<float> input = {1.0F, 2.0F, 3.0F};
  std::vector<Tensor> outputs = backend->Infer({MakeInputTensor(&input)});

  ASSERT_EQ(outputs.size(), 1U);
  ExpectAddOutput(outputs[0]);
}

TEST(OnnxCpuBackendTest, UsesExplicitAllocatorForOutputs) {
  BackendPtr backend = CreateBackend(AddModel());
  PooledTensorAllocator allocator;
  std::vector<float> input = {1.0F, 2.0F, 3.0F};
  Tensor input_tensor = MakeInputTensor(&input);

  void* first_output_data = nullptr;
  {
    std::vector<Tensor> outputs = backend->Infer(input_tensor, allocator);
    ASSERT_EQ(outputs.size(), 1U);
    first_output_data = outputs.front().data();
    ExpectAddOutput(outputs.front());
  }

  std::vector<Tensor> outputs = backend->Infer(input_tensor, allocator);
  ASSERT_EQ(outputs.size(), 1U);
  EXPECT_EQ(outputs.front().data(), first_output_data);
  ExpectAddOutput(outputs.front());
}

TEST(OnnxCpuBackendTest, BindsMultiInputTensorsByName) {
  BackendPtr backend = CreateBackend(SubModel());
  const ModelInfo& info = backend->model_info();

  ASSERT_EQ(info.inputs.size(), 2U);
  EXPECT_EQ(info.inputs[0].name, "x");
  EXPECT_EQ(info.inputs[0].shape, std::vector<int64_t>({3, 4, 5}));
  EXPECT_EQ(info.inputs[1].name, "y");
  EXPECT_EQ(info.inputs[1].shape, std::vector<int64_t>({3, 4, 5}));
  ASSERT_EQ(info.outputs.size(), 1U);
  EXPECT_EQ(info.outputs[0].name, "z");
  EXPECT_EQ(info.outputs[0].shape, std::vector<int64_t>({3, 4, 5}));

  std::vector<float> x = Sequence(60, 100.0F);
  std::vector<float> y = Sequence(60, 1.0F);
  Tensor x_tensor = MakeNamedInputTensor("x", {3, 4, 5}, &x);
  Tensor y_tensor = MakeNamedInputTensor("y", {3, 4, 5}, &y);

  std::vector<Tensor> outputs = backend->Infer({y_tensor, x_tensor});

  ASSERT_EQ(outputs.size(), 1U);
  EXPECT_EQ(outputs[0].name(), "z");
  EXPECT_EQ(outputs[0].shape(), std::vector<int64_t>({3, 4, 5}));
  EXPECT_EQ(outputs[0].device().type, DeviceType::kCpu);
  ExpectFloatValues(outputs[0], Difference(x, y));
}

TEST(OnnxCpuBackendTest, RejectsNamelessTensorForMultiInputModel) {
  BackendPtr backend = CreateBackend(SubModel());

  std::vector<float> x = Sequence(60, 100.0F);
  std::vector<float> y = Sequence(60, 1.0F);
  Tensor x_tensor = MakeNamedInputTensor("", {3, 4, 5}, &x);
  Tensor y_tensor = MakeNamedInputTensor("y", {3, 4, 5}, &y);

  EXPECT_THROW(static_cast<void>(backend->Infer({x_tensor, y_tensor})),
               std::invalid_argument);
}

TEST(OnnxCpuBackendTest, ReturnsNamedMultiOutputs) {
  BackendPtr backend = CreateBackend(SplitModel());
  const ModelInfo& info = backend->model_info();

  ASSERT_EQ(info.inputs.size(), 1U);
  EXPECT_EQ(info.inputs[0].name, "input");
  EXPECT_EQ(info.inputs[0].shape, std::vector<int64_t>({6}));
  ASSERT_EQ(info.outputs.size(), 3U);
  EXPECT_EQ(info.outputs[0].name, "output_1");
  EXPECT_EQ(info.outputs[1].name, "output_2");
  EXPECT_EQ(info.outputs[2].name, "output_3");

  std::vector<float> input = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
  std::vector<Tensor> outputs =
      backend->Infer({MakeNamedInputTensor("input", {6}, &input)});

  ASSERT_EQ(outputs.size(), 3U);
  EXPECT_EQ(outputs[0].name(), "output_1");
  EXPECT_EQ(outputs[0].shape(), std::vector<int64_t>({2}));
  ExpectFloatValues(outputs[0], {1.0F, 2.0F});
  EXPECT_EQ(outputs[1].name(), "output_2");
  EXPECT_EQ(outputs[1].shape(), std::vector<int64_t>({2}));
  ExpectFloatValues(outputs[1], {3.0F, 4.0F});
  EXPECT_EQ(outputs[2].name(), "output_3");
  EXPECT_EQ(outputs[2].shape(), std::vector<int64_t>({2}));
  ExpectFloatValues(outputs[2], {5.0F, 6.0F});
}

TEST(OnnxCpuBackendTest, BindsRequestedOutputNamesInOrder) {
  BackendPtr backend = CreateBackend(SplitModel(), Device{DeviceType::kCpu, 0},
                                     {"output_3", "output_1"});
  const ModelInfo& info = backend->model_info();

  ASSERT_EQ(info.outputs.size(), 2U);
  EXPECT_EQ(info.outputs[0].name, "output_3");
  EXPECT_EQ(info.outputs[1].name, "output_1");

  std::vector<float> input = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
  std::vector<Tensor> outputs =
      backend->Infer({MakeNamedInputTensor("input", {6}, &input)});

  ASSERT_EQ(outputs.size(), 2U);
  EXPECT_EQ(outputs[0].name(), "output_3");
  ExpectFloatValues(outputs[0], {5.0F, 6.0F});
  EXPECT_EQ(outputs[1].name(), "output_1");
  ExpectFloatValues(outputs[1], {1.0F, 2.0F});
}

TEST(OnnxCpuBackendTest, RejectsUnknownRequestedOutputName) {
  EXPECT_THROW(
      static_cast<void>(CreateBackend(SplitModel(), Device{DeviceType::kCpu, 0},
                                      {"missing_output"})),
      std::invalid_argument);
}

TEST(OnnxCpuBackendTest, RunsFixedMultiBatchModel) {
  BackendPtr backend = CreateBackend(FixedBatchAddModel());
  const ModelInfo& info = backend->model_info();

  ASSERT_EQ(info.inputs.size(), 2U);
  EXPECT_EQ(info.inputs[0].name, "lhs");
  EXPECT_EQ(info.inputs[0].shape, std::vector<int64_t>({2, 3}));
  EXPECT_EQ(info.inputs[1].name, "rhs");
  EXPECT_EQ(info.inputs[1].shape, std::vector<int64_t>({2, 3}));
  ASSERT_EQ(info.outputs.size(), 1U);
  EXPECT_EQ(info.outputs[0].name, "sum");
  EXPECT_EQ(info.outputs[0].shape, std::vector<int64_t>({2, 3}));

  ExpectBatchedAddOutput(backend.get(), 2, DeviceType::kCpu);
}

TEST(OnnxCpuBackendTest, RunsDynamicBatchModelWithBatchOneAndTwo) {
  BackendPtr backend = CreateBackend(DynamicBatchAddModel());
  const ModelInfo& info = backend->model_info();

  ASSERT_EQ(info.inputs.size(), 2U);
  EXPECT_EQ(info.inputs[0].name, "lhs");
  EXPECT_EQ(info.inputs[0].shape, std::vector<int64_t>({-1, 3}));
  EXPECT_EQ(info.inputs[1].name, "rhs");
  EXPECT_EQ(info.inputs[1].shape, std::vector<int64_t>({-1, 3}));
  ASSERT_EQ(info.outputs.size(), 1U);
  EXPECT_EQ(info.outputs[0].name, "sum");
  EXPECT_EQ(info.outputs[0].shape, std::vector<int64_t>({-1, 3}));

  ExpectBatchedAddOutput(backend.get(), 1, DeviceType::kCpu);
  ExpectBatchedAddOutput(backend.get(), 2, DeviceType::kCpu);
}

TEST(OnnxGpuBackendTest, RunsWithCudaOutput) {
  Model model = AddModel();
  if (!BackendFactory().Supports(model, CudaDevice())) {
    GTEST_SKIP() << "ONNX Runtime CUDA provider is unavailable";
  }

  BackendPtr backend = CreateBackend(std::move(model), CudaDevice());

  std::vector<float> input = {1.0F, 2.0F, 3.0F};
  std::vector<Tensor> outputs = backend->Infer({MakeInputTensor(&input)});

  ASSERT_EQ(outputs.size(), 1U);
  EXPECT_EQ(outputs[0].name(), "output");
  EXPECT_EQ(outputs[0].data_type(), DataType::kFloat32);
  EXPECT_EQ(outputs[0].shape(), std::vector<int64_t>({3}));
  EXPECT_EQ(outputs[0].device().type, DeviceType::kCuda);
  EXPECT_EQ(outputs[0].device().id, 0);
  EXPECT_EQ(outputs[0].bytes(), 3U * sizeof(float));
  EXPECT_NE(outputs[0].data(), nullptr);
}

TEST(OnnxGpuBackendTest, BindsMultiInputTensorsByName) {
  Model model = SubModel();
  if (!BackendFactory().Supports(model, CudaDevice())) {
    GTEST_SKIP() << "ONNX Runtime CUDA provider is unavailable";
  }

  BackendPtr backend = CreateBackend(std::move(model), CudaDevice());

  std::vector<float> x = Sequence(60, 100.0F);
  std::vector<float> y = Sequence(60, 1.0F);
  Tensor x_tensor = MakeNamedInputTensor("x", {3, 4, 5}, &x);
  Tensor y_tensor = MakeNamedInputTensor("y", {3, 4, 5}, &y);

  std::vector<Tensor> outputs = backend->Infer({y_tensor, x_tensor});

  ASSERT_EQ(outputs.size(), 1U);
  EXPECT_EQ(outputs[0].name(), "z");
  EXPECT_EQ(outputs[0].shape(), std::vector<int64_t>({3, 4, 5}));
  EXPECT_EQ(outputs[0].device().type, DeviceType::kCuda);
  EXPECT_EQ(outputs[0].device().id, 0);
  ExpectFloatValues(outputs[0], Difference(x, y));
}

TEST(OnnxGpuBackendTest, BindsRequestedMultiOutputNamesInOrder) {
  Model model = SplitModel();
  if (!BackendFactory().Supports(model, CudaDevice())) {
    GTEST_SKIP() << "ONNX Runtime CUDA provider is unavailable";
  }

  BackendPtr backend =
      CreateBackend(std::move(model), CudaDevice(), {"output_2", "output_1"});
  const ModelInfo& info = backend->model_info();

  ASSERT_EQ(info.outputs.size(), 2U);
  EXPECT_EQ(info.outputs[0].name, "output_2");
  EXPECT_EQ(info.outputs[1].name, "output_1");

  std::vector<float> input = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
  std::vector<Tensor> outputs =
      backend->Infer({MakeNamedInputTensor("input", {6}, &input)});

  ASSERT_EQ(outputs.size(), 2U);
  EXPECT_EQ(outputs[0].name(), "output_2");
  EXPECT_EQ(outputs[0].device().type, DeviceType::kCuda);
  ExpectFloatValues(outputs[0], {3.0F, 4.0F});
  EXPECT_EQ(outputs[1].name(), "output_1");
  EXPECT_EQ(outputs[1].device().type, DeviceType::kCuda);
  ExpectFloatValues(outputs[1], {1.0F, 2.0F});
}

TEST(OnnxGpuBackendTest, RunsFixedMultiBatchModel) {
  Model model = FixedBatchAddModel();
  if (!BackendFactory().Supports(model, CudaDevice())) {
    GTEST_SKIP() << "ONNX Runtime CUDA provider is unavailable";
  }

  BackendPtr backend = CreateBackend(std::move(model), CudaDevice());

  const ModelInfo& info = backend->model_info();
  ASSERT_EQ(info.inputs.size(), 2U);
  EXPECT_EQ(info.inputs[0].shape, std::vector<int64_t>({2, 3}));
  ASSERT_EQ(info.outputs.size(), 1U);
  EXPECT_EQ(info.outputs[0].shape, std::vector<int64_t>({2, 3}));

  ExpectBatchedAddOutput(backend.get(), 2, DeviceType::kCuda);
}

TEST(OnnxGpuBackendTest, RunsDynamicBatchModelWithBatchOneAndTwo) {
  Model model = DynamicBatchAddModel();
  if (!BackendFactory().Supports(model, CudaDevice())) {
    GTEST_SKIP() << "ONNX Runtime CUDA provider is unavailable";
  }

  BackendPtr backend = CreateBackend(std::move(model), CudaDevice());

  const ModelInfo& info = backend->model_info();
  ASSERT_EQ(info.inputs.size(), 2U);
  EXPECT_EQ(info.inputs[0].shape, std::vector<int64_t>({-1, 3}));
  ASSERT_EQ(info.outputs.size(), 1U);
  EXPECT_EQ(info.outputs[0].shape, std::vector<int64_t>({-1, 3}));

  ExpectBatchedAddOutput(backend.get(), 1, DeviceType::kCuda);
  ExpectBatchedAddOutput(backend.get(), 2, DeviceType::kCuda);
}

TEST(OnnxGpuBackendTest, ThrowsWhenCudaProviderIsUnavailable) {
  Model model = AddModel();
  if (BackendFactory().Supports(model, CudaDevice())) {
    GTEST_SKIP() << "ONNX Runtime CUDA provider is available";
  }

  EXPECT_THROW(static_cast<void>(CreateBackend(std::move(model), CudaDevice())),
               std::invalid_argument);
}

}  // namespace
}  // namespace mw::infer
