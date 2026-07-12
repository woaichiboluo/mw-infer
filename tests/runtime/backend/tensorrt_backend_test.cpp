#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "mw/infer/runtime/backend/backend.h"
#include "mw/infer/runtime/tensor/tensor_allocator.h"

namespace mw::infer {
namespace {

#if NV_TENSORRT_MAJOR < 10
constexpr std::uint32_t kNetworkDefinitionFlags =
    1U << static_cast<std::uint32_t>(
        nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
#else
constexpr std::uint32_t kNetworkDefinitionFlags = 0U;
#endif

class TestTensorRtLogger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* message) noexcept override {
    if (severity <= Severity::kERROR) {
      std::fprintf(stderr, "TensorRT test error: %s\n", message);
    }
  }
};

template <typename T>
std::unique_ptr<T> CheckTensorRtPtr(T* ptr, const char* operation) {
  if (ptr == nullptr) {
    throw std::runtime_error(std::string(operation) + " failed");
  }
  return std::unique_ptr<T>(ptr);
}

bool HasUsableCudaDevice() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

Device CudaDevice() { return Device{DeviceType::kCuda, 0}; }

void CheckCuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) +
                             " failed: " + cudaGetErrorString(status));
  }
}

void CheckTrue(bool value, const char* operation) {
  if (!value) {
    throw std::runtime_error(std::string(operation) + " failed");
  }
}

void CUDART_CB MarkCompleteAfterDelay(void* opaque) noexcept {
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  static_cast<std::atomic<bool>*>(opaque)->store(true,
                                                 std::memory_order_release);
}

nvinfer1::Dims MakeDims(std::vector<int64_t> shape) {
  nvinfer1::Dims dims{};
  dims.nbDims = static_cast<int32_t>(shape.size());
  for (int32_t index = 0; index < nvinfer1::Dims::MAX_DIMS; ++index) {
    dims.d[index] = 0;
  }
  for (std::size_t index = 0; index < shape.size(); ++index) {
    dims.d[index] = shape[index];
  }
  return dims;
}

std::shared_ptr<std::vector<std::uint8_t>> BuildDynamicAddEngine() {
  CheckCuda(cudaSetDevice(0), "cudaSetDevice");
  auto engine_bytes = std::make_shared<std::vector<std::uint8_t>>();

  TestTensorRtLogger logger;
  auto builder = CheckTensorRtPtr(nvinfer1::createInferBuilder(logger),
                                  "createInferBuilder");
  auto network =
      CheckTensorRtPtr(builder->createNetworkV2(kNetworkDefinitionFlags),
                       "IBuilder::createNetworkV2");
  auto config = CheckTensorRtPtr(builder->createBuilderConfig(),
                                 "IBuilder::createBuilderConfig");

#if NV_TENSORRT_MAJOR > 8 || (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR >= 6)
  config->setBuilderOptimizationLevel(0);
#endif
  nvinfer1::ITensor* lhs =
      network->addInput("lhs", nvinfer1::DataType::kFLOAT, MakeDims({-1, 3}));
  nvinfer1::ITensor* rhs =
      network->addInput("rhs", nvinfer1::DataType::kFLOAT, MakeDims({-1, 3}));
  CheckTrue(lhs != nullptr, "INetworkDefinition::addInput(lhs)");
  CheckTrue(rhs != nullptr, "INetworkDefinition::addInput(rhs)");

  nvinfer1::IElementWiseLayer* sum_layer =
      network->addElementWise(*lhs, *rhs, nvinfer1::ElementWiseOperation::kSUM);
  CheckTrue(sum_layer != nullptr, "INetworkDefinition::addElementWise");
  nvinfer1::ITensor* sum = sum_layer->getOutput(0);
  CheckTrue(sum != nullptr, "IElementWiseLayer::getOutput");
  sum->setName("sum");
  network->markOutput(*sum);

  nvinfer1::IOptimizationProfile* profile =
      builder->createOptimizationProfile();
  CheckTrue(profile != nullptr, "IBuilder::createOptimizationProfile");
  CheckTrue(profile->setDimensions("lhs", nvinfer1::OptProfileSelector::kMIN,
                                   MakeDims({1, 3})),
            "IOptimizationProfile::setDimensions(lhs min)");
  CheckTrue(profile->setDimensions("lhs", nvinfer1::OptProfileSelector::kOPT,
                                   MakeDims({2, 3})),
            "IOptimizationProfile::setDimensions(lhs opt)");
  CheckTrue(profile->setDimensions("lhs", nvinfer1::OptProfileSelector::kMAX,
                                   MakeDims({4, 3})),
            "IOptimizationProfile::setDimensions(lhs max)");
  CheckTrue(profile->setDimensions("rhs", nvinfer1::OptProfileSelector::kMIN,
                                   MakeDims({1, 3})),
            "IOptimizationProfile::setDimensions(rhs min)");
  CheckTrue(profile->setDimensions("rhs", nvinfer1::OptProfileSelector::kOPT,
                                   MakeDims({2, 3})),
            "IOptimizationProfile::setDimensions(rhs opt)");
  CheckTrue(profile->setDimensions("rhs", nvinfer1::OptProfileSelector::kMAX,
                                   MakeDims({4, 3})),
            "IOptimizationProfile::setDimensions(rhs max)");
  CheckTrue(config->addOptimizationProfile(profile) >= 0,
            "IBuilderConfig::addOptimizationProfile");

  auto serialized =
      CheckTensorRtPtr(builder->buildSerializedNetwork(*network, *config),
                       "IBuilder::buildSerializedNetwork");
  const auto* data = static_cast<const std::uint8_t*>(serialized->data());
  engine_bytes->assign(data, data + serialized->size());
  return engine_bytes;
}

std::shared_ptr<std::vector<std::uint8_t>> BuildStaticIdentityEngine() {
  CheckCuda(cudaSetDevice(0), "cudaSetDevice");
  auto engine_bytes = std::make_shared<std::vector<std::uint8_t>>();

  TestTensorRtLogger logger;
  auto builder = CheckTensorRtPtr(nvinfer1::createInferBuilder(logger),
                                  "createInferBuilder");
  auto network =
      CheckTensorRtPtr(builder->createNetworkV2(kNetworkDefinitionFlags),
                       "IBuilder::createNetworkV2");
  auto config = CheckTensorRtPtr(builder->createBuilderConfig(),
                                 "IBuilder::createBuilderConfig");

#if NV_TENSORRT_MAJOR > 8 || (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR >= 6)
  config->setBuilderOptimizationLevel(0);
#endif
  nvinfer1::ITensor* input =
      network->addInput("input", nvinfer1::DataType::kFLOAT, MakeDims({2, 3}));
  CheckTrue(input != nullptr, "INetworkDefinition::addInput(input)");

  nvinfer1::IIdentityLayer* identity = network->addIdentity(*input);
  CheckTrue(identity != nullptr, "INetworkDefinition::addIdentity");
  nvinfer1::ITensor* output = identity->getOutput(0);
  CheckTrue(output != nullptr, "IIdentityLayer::getOutput");
  output->setName("identity");
  network->markOutput(*output);

  auto serialized =
      CheckTensorRtPtr(builder->buildSerializedNetwork(*network, *config),
                       "IBuilder::buildSerializedNetwork");
  const auto* data = static_cast<const std::uint8_t*>(serialized->data());
  engine_bytes->assign(data, data + serialized->size());
  return engine_bytes;
}

std::shared_ptr<std::vector<std::uint8_t>> BuildDynamicAddSubEngine() {
  CheckCuda(cudaSetDevice(0), "cudaSetDevice");
  auto engine_bytes = std::make_shared<std::vector<std::uint8_t>>();

  TestTensorRtLogger logger;
  auto builder = CheckTensorRtPtr(nvinfer1::createInferBuilder(logger),
                                  "createInferBuilder");
  auto network =
      CheckTensorRtPtr(builder->createNetworkV2(kNetworkDefinitionFlags),
                       "IBuilder::createNetworkV2");
  auto config = CheckTensorRtPtr(builder->createBuilderConfig(),
                                 "IBuilder::createBuilderConfig");

#if NV_TENSORRT_MAJOR > 8 || (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR >= 6)
  config->setBuilderOptimizationLevel(0);
#endif
  nvinfer1::ITensor* lhs =
      network->addInput("lhs", nvinfer1::DataType::kFLOAT, MakeDims({-1, 3}));
  nvinfer1::ITensor* rhs =
      network->addInput("rhs", nvinfer1::DataType::kFLOAT, MakeDims({-1, 3}));
  CheckTrue(lhs != nullptr, "INetworkDefinition::addInput(lhs)");
  CheckTrue(rhs != nullptr, "INetworkDefinition::addInput(rhs)");

  nvinfer1::IElementWiseLayer* sum_layer =
      network->addElementWise(*lhs, *rhs, nvinfer1::ElementWiseOperation::kSUM);
  CheckTrue(sum_layer != nullptr, "INetworkDefinition::addElementWise(sum)");
  nvinfer1::ITensor* sum = sum_layer->getOutput(0);
  CheckTrue(sum != nullptr, "IElementWiseLayer::getOutput(sum)");
  sum->setName("sum");
  network->markOutput(*sum);

  nvinfer1::IElementWiseLayer* diff_layer =
      network->addElementWise(*lhs, *rhs, nvinfer1::ElementWiseOperation::kSUB);
  CheckTrue(diff_layer != nullptr, "INetworkDefinition::addElementWise(diff)");
  nvinfer1::ITensor* diff = diff_layer->getOutput(0);
  CheckTrue(diff != nullptr, "IElementWiseLayer::getOutput(diff)");
  diff->setName("diff");
  network->markOutput(*diff);

  nvinfer1::IOptimizationProfile* profile =
      builder->createOptimizationProfile();
  CheckTrue(profile != nullptr, "IBuilder::createOptimizationProfile");
  CheckTrue(profile->setDimensions("lhs", nvinfer1::OptProfileSelector::kMIN,
                                   MakeDims({1, 3})),
            "IOptimizationProfile::setDimensions(lhs min)");
  CheckTrue(profile->setDimensions("lhs", nvinfer1::OptProfileSelector::kOPT,
                                   MakeDims({2, 3})),
            "IOptimizationProfile::setDimensions(lhs opt)");
  CheckTrue(profile->setDimensions("lhs", nvinfer1::OptProfileSelector::kMAX,
                                   MakeDims({4, 3})),
            "IOptimizationProfile::setDimensions(lhs max)");
  CheckTrue(profile->setDimensions("rhs", nvinfer1::OptProfileSelector::kMIN,
                                   MakeDims({1, 3})),
            "IOptimizationProfile::setDimensions(rhs min)");
  CheckTrue(profile->setDimensions("rhs", nvinfer1::OptProfileSelector::kOPT,
                                   MakeDims({2, 3})),
            "IOptimizationProfile::setDimensions(rhs opt)");
  CheckTrue(profile->setDimensions("rhs", nvinfer1::OptProfileSelector::kMAX,
                                   MakeDims({4, 3})),
            "IOptimizationProfile::setDimensions(rhs max)");
  CheckTrue(config->addOptimizationProfile(profile) >= 0,
            "IBuilderConfig::addOptimizationProfile");

  auto serialized =
      CheckTensorRtPtr(builder->buildSerializedNetwork(*network, *config),
                       "IBuilder::buildSerializedNetwork");
  const auto* data = static_cast<const std::uint8_t*>(serialized->data());
  engine_bytes->assign(data, data + serialized->size());
  return engine_bytes;
}

std::shared_ptr<std::vector<std::uint8_t>> DynamicAddEngineBytes() {
  static std::shared_ptr<std::vector<std::uint8_t>> engine =
      BuildDynamicAddEngine();
  return engine;
}

std::shared_ptr<std::vector<std::uint8_t>> StaticIdentityEngineBytes() {
  static std::shared_ptr<std::vector<std::uint8_t>> engine =
      BuildStaticIdentityEngine();
  return engine;
}

std::shared_ptr<std::vector<std::uint8_t>> DynamicAddSubEngineBytes() {
  static std::shared_ptr<std::vector<std::uint8_t>> engine =
      BuildDynamicAddSubEngine();
  return engine;
}

Model DynamicAddEngineModel() {
  std::shared_ptr<std::vector<std::uint8_t>> engine = DynamicAddEngineBytes();
  return ModelFromMemory(ModelFormat::kTensorRT, engine->data(), engine->size(),
                         engine, "dynamic_add_engine");
}

Model StaticIdentityEngineModel() {
  std::shared_ptr<std::vector<std::uint8_t>> engine =
      StaticIdentityEngineBytes();
  return ModelFromMemory(ModelFormat::kTensorRT, engine->data(), engine->size(),
                         engine, "static_identity_engine");
}

Model DynamicAddSubEngineModel() {
  std::shared_ptr<std::vector<std::uint8_t>> engine =
      DynamicAddSubEngineBytes();
  return ModelFromMemory(ModelFormat::kTensorRT, engine->data(), engine->size(),
                         engine, "dynamic_add_sub_engine");
}

Tensor MakeCpuFloatTensor(std::string name, std::vector<int64_t> shape,
                          const std::vector<float>& data) {
  TensorDesc desc;
  desc.info.name = std::move(name);
  desc.info.data_type = DataType::kFloat32;
  desc.info.shape = std::move(shape);
  desc.device = Device{DeviceType::kCpu, 0};
  Tensor tensor = Tensor::Allocate(std::move(desc));
  std::memcpy(tensor.data(), data.data(), tensor.bytes());
  return tensor;
}

Tensor MakeCudaFloatTensor(std::string name, std::vector<int64_t> shape,
                           const std::vector<float>& data) {
  Tensor tensor = MakeCpuFloatTensor(std::move(name), std::move(shape), data)
                      .CopyTo(CudaDevice());
  return tensor;
}

TensorDesc MakeCudaFloatDesc(std::vector<int64_t> shape) {
  TensorDesc desc;
  desc.info.data_type = DataType::kFloat32;
  desc.info.shape = std::move(shape);
  desc.device = CudaDevice();
  return desc;
}

class CountingTensorAllocator final : public TensorAllocator {
 public:
  bool Supports(Device device) const override {
    return allocator_.Supports(device);
  }

  Tensor Allocate(TensorDesc desc) override {
    ++allocation_count_;
    return allocator_.Allocate(std::move(desc));
  }

  std::size_t allocation_count() const { return allocation_count_; }

 private:
  DirectTensorAllocator allocator_;
  std::size_t allocation_count_ = 0;
};

std::vector<float> Sequence(std::size_t count, float offset) {
  std::vector<float> values(count);
  for (std::size_t index = 0; index < count; ++index) {
    values[index] = static_cast<float>(index) + offset;
  }
  return values;
}

std::vector<float> Sum(const std::vector<float>& lhs,
                       const std::vector<float>& rhs) {
  std::vector<float> values(lhs.size());
  for (std::size_t index = 0; index < lhs.size(); ++index) {
    values[index] = lhs[index] + rhs[index];
  }
  return values;
}

std::vector<float> Diff(const std::vector<float>& lhs,
                        const std::vector<float>& rhs) {
  std::vector<float> values(lhs.size());
  for (std::size_t index = 0; index < lhs.size(); ++index) {
    values[index] = lhs[index] - rhs[index];
  }
  return values;
}

std::vector<float> CopyFloatTensor(const Tensor& tensor) {
  return tensor.CopyToHostVector<float>();
}

void ExpectFloatValues(const Tensor& tensor,
                       const std::vector<float>& expected) {
  const std::vector<float> actual = CopyFloatTensor(tensor);
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t index = 0; index < actual.size(); ++index) {
    EXPECT_FLOAT_EQ(actual[index], expected[index]);
  }
}

void ExpectDynamicAddOutput(IBackend* backend, int64_t batch) {
  const std::vector<float> lhs =
      Sequence(static_cast<std::size_t>(batch * 3), 1.0F);
  const std::vector<float> rhs =
      Sequence(static_cast<std::size_t>(batch * 3), 100.0F);
  Tensor lhs_tensor = MakeCudaFloatTensor("lhs", {batch, 3}, lhs);
  Tensor rhs_tensor = MakeCudaFloatTensor("rhs", {batch, 3}, rhs);

  std::vector<Tensor> outputs = backend->Infer({rhs_tensor, lhs_tensor});

  ASSERT_EQ(outputs.size(), 1U);
  EXPECT_EQ(outputs[0].name(), "sum");
  EXPECT_EQ(outputs[0].data_type(), DataType::kFloat32);
  EXPECT_EQ(outputs[0].shape(), std::vector<int64_t>({batch, 3}));
  EXPECT_EQ(outputs[0].device().type, DeviceType::kCuda);
  EXPECT_EQ(outputs[0].device().id, 0);
  ExpectFloatValues(outputs[0], Sum(lhs, rhs));
}

TEST(TensorRtBackendTest, SupportsTensorRtEngineOnCudaOnly) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "CUDA device is unavailable";
  }

  Model model = DynamicAddEngineModel();
  BackendFactory factory;

  EXPECT_TRUE(factory.Supports(model, CudaDevice()));
  EXPECT_FALSE(factory.Supports(model, Device{DeviceType::kCpu, 0}));
}

TEST(TensorRtBackendTest, RejectsInvalidSerializedEngine) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "CUDA device is unavailable";
  }

  const std::array<std::uint8_t, 16> invalid_engine{};
  for (int attempt = 0; attempt < 2; ++attempt) {
    EXPECT_THROW(
        static_cast<void>(CreateBackend(
            ModelFromMemory(ModelFormat::kTensorRT, invalid_engine.data(),
                            invalid_engine.size()),
            CudaDevice())),
        std::runtime_error);
  }
}

TEST(TensorRtBackendTest, RunsStaticSingleInputEngineWithUnnamedTensor) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "CUDA device is unavailable";
  }

  BackendPtr backend = CreateBackend(StaticIdentityEngineModel(), CudaDevice());
  const ModelInfo& info = backend->model_info();

  ASSERT_EQ(info.inputs.size(), 1U);
  EXPECT_EQ(info.inputs[0].name, "input");
  EXPECT_EQ(info.inputs[0].shape, std::vector<int64_t>({2, 3}));
  ASSERT_EQ(info.outputs.size(), 1U);
  EXPECT_EQ(info.outputs[0].name, "identity");
  EXPECT_EQ(info.outputs[0].shape, std::vector<int64_t>({2, 3}));

  const std::vector<float> values = Sequence(6, 10.0F);
  Tensor input = MakeCudaFloatTensor("", {2, 3}, values);

  std::vector<Tensor> outputs = backend->Infer(input);

  ASSERT_EQ(outputs.size(), 1U);
  EXPECT_EQ(outputs[0].name(), "identity");
  EXPECT_EQ(outputs[0].shape(), std::vector<int64_t>({2, 3}));
  EXPECT_EQ(outputs[0].device().type, DeviceType::kCuda);
  ExpectFloatValues(outputs[0], values);
}

TEST(TensorRtBackendTest, RunsWithInjectedExecutionStream) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "CUDA device is unavailable";
  }

  auto stream = std::make_shared<ExecutionStream>(CudaDevice());
  BackendPtr backend =
      CreateBackend(StaticIdentityEngineModel(), CudaDevice(), {}, stream);

  const std::vector<float> values = Sequence(6, 10.0F);
  Tensor input = MakeCudaFloatTensor("", {2, 3}, values);
  std::atomic<bool> preceding_work_completed{false};
  ASSERT_EQ(cudaLaunchHostFunc(stream->cuda_handle(), MarkCompleteAfterDelay,
                               &preceding_work_completed),
            cudaSuccess);
  std::vector<Tensor> outputs = backend->Infer(input);

  EXPECT_FALSE(preceding_work_completed.load(std::memory_order_acquire));
  stream->Synchronize();

  ASSERT_EQ(outputs.size(), 1U);
  EXPECT_TRUE(preceding_work_completed.load(std::memory_order_acquire));
  ExpectFloatValues(outputs[0], values);
}

TEST(TensorRtBackendTest,
     RetainsCudaInputsAndSelectedOutputsUntilInjectedStreamCompletes) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "CUDA device is unavailable";
  }

  auto stream = std::make_shared<ExecutionStream>(CudaDevice());
  BackendPtr backend =
      CreateBackend(StaticIdentityEngineModel(), CudaDevice(), {}, stream);
  auto upstream = std::make_unique<CountingTensorAllocator>();
  CountingTensorAllocator* counting = upstream.get();
  PooledTensorAllocator allocator(std::move(upstream));
  const std::vector<float> values = Sequence(6, 10.0F);
  Tensor input =
      MakeCpuFloatTensor("", {2, 3}, values).CopyTo(CudaDevice(), allocator);
  std::atomic<bool> preceding_work_completed{false};
  ASSERT_EQ(cudaLaunchHostFunc(stream->cuda_handle(), MarkCompleteAfterDelay,
                               &preceding_work_completed),
            cudaSuccess);

  std::vector<Tensor> outputs = backend->Infer(input, allocator);

  EXPECT_FALSE(preceding_work_completed.load(std::memory_order_acquire));
  ASSERT_EQ(outputs.size(), 1U);
  EXPECT_EQ(counting->allocation_count(), 2U);
  input = Tensor{};
  outputs.clear();
  {
    Tensor first = Tensor::Allocate(MakeCudaFloatDesc({2, 3}), allocator);
    Tensor second = Tensor::Allocate(MakeCudaFloatDesc({2, 3}), allocator);
    EXPECT_EQ(counting->allocation_count(), 4U);
  }

  stream->Synchronize();
  Tensor second_input = MakeCudaFloatTensor("", {2, 3}, values);
  std::vector<Tensor> second_outputs =
      backend->Infer(second_input, allocator);
  EXPECT_EQ(counting->allocation_count(), 4U);
  stream->Synchronize();
  ASSERT_EQ(second_outputs.size(), 1U);
  ExpectFloatValues(second_outputs[0], values);
}

TEST(TensorRtBackendTest, WaitsForInjectedStreamBeforeDestroyingSession) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "CUDA device is unavailable";
  }

  auto stream = std::make_shared<ExecutionStream>(CudaDevice());
  BackendPtr backend =
      CreateBackend(DynamicAddEngineModel(), CudaDevice(), {}, stream);
  PooledTensorAllocator allocator;
  const std::vector<float> lhs = Sequence(6, 1.0F);
  const std::vector<float> rhs = Sequence(6, 100.0F);
  Tensor lhs_tensor = MakeCpuFloatTensor("lhs", {2, 3}, lhs);
  Tensor rhs_tensor = MakeCpuFloatTensor("rhs", {2, 3}, rhs);
  std::atomic<bool> preceding_work_completed{false};
  ASSERT_EQ(cudaLaunchHostFunc(stream->cuda_handle(), MarkCompleteAfterDelay,
                               &preceding_work_completed),
            cudaSuccess);

  std::vector<Tensor> outputs =
      backend->Infer({rhs_tensor, lhs_tensor}, allocator);
  ASSERT_FALSE(preceding_work_completed.load(std::memory_order_acquire));

  backend.reset();

  EXPECT_TRUE(preceding_work_completed.load(std::memory_order_acquire));
  ASSERT_EQ(outputs.size(), 1U);
  ExpectFloatValues(outputs[0], Sum(lhs, rhs));
}

TEST(TensorRtBackendTest, KeepsStagedInputsAliveUntilInjectedStreamCompletes) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "CUDA device is unavailable";
  }

  auto stream = std::make_shared<ExecutionStream>(CudaDevice());
  BackendPtr backend =
      CreateBackend(DynamicAddEngineModel(), CudaDevice(), {}, stream);
  auto upstream = std::make_unique<CountingTensorAllocator>();
  CountingTensorAllocator* counting = upstream.get();
  PooledTensorAllocator allocator(std::move(upstream));
  const std::vector<float> lhs = Sequence(6, 1.0F);
  const std::vector<float> rhs = Sequence(6, 100.0F);
  Tensor lhs_tensor = MakeCpuFloatTensor("lhs", {2, 3}, lhs);
  Tensor rhs_tensor = MakeCpuFloatTensor("rhs", {2, 3}, rhs);
  std::atomic<bool> preceding_work_completed{false};
  ASSERT_EQ(cudaLaunchHostFunc(stream->cuda_handle(), MarkCompleteAfterDelay,
                               &preceding_work_completed),
            cudaSuccess);

  std::vector<Tensor> outputs =
      backend->Infer({rhs_tensor, lhs_tensor}, allocator);

  EXPECT_FALSE(preceding_work_completed.load(std::memory_order_acquire));
  ASSERT_EQ(outputs.size(), 1U);
  EXPECT_EQ(counting->allocation_count(), 3U);
  {
    Tensor scratch = Tensor::Allocate(MakeCudaFloatDesc({2, 3}), allocator);
    EXPECT_EQ(counting->allocation_count(), 4U);
  }
  stream->Synchronize();
  ExpectFloatValues(outputs[0], Sum(lhs, rhs));

  std::vector<Tensor> second_outputs =
      backend->Infer({rhs_tensor, lhs_tensor}, allocator);
  EXPECT_EQ(counting->allocation_count(), 4U);
  stream->Synchronize();
  ASSERT_EQ(second_outputs.size(), 1U);
  ExpectFloatValues(second_outputs[0], Sum(lhs, rhs));
}

TEST(TensorRtBackendTest,
     KeepsUnselectedOutputsAliveUntilInjectedStreamCompletes) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "CUDA device is unavailable";
  }

  auto stream = std::make_shared<ExecutionStream>(CudaDevice());
  BackendPtr backend = CreateBackend(DynamicAddSubEngineModel(), CudaDevice(),
                                     {"sum"}, stream);
  auto upstream = std::make_unique<CountingTensorAllocator>();
  CountingTensorAllocator* counting = upstream.get();
  PooledTensorAllocator allocator(std::move(upstream));
  const std::vector<float> lhs = Sequence(6, 1.0F);
  const std::vector<float> rhs = Sequence(6, 100.0F);
  Tensor lhs_tensor = MakeCudaFloatTensor("lhs", {2, 3}, lhs);
  Tensor rhs_tensor = MakeCudaFloatTensor("rhs", {2, 3}, rhs);
  std::atomic<bool> preceding_work_completed{false};
  ASSERT_EQ(cudaLaunchHostFunc(stream->cuda_handle(), MarkCompleteAfterDelay,
                               &preceding_work_completed),
            cudaSuccess);

  std::vector<Tensor> outputs =
      backend->Infer({rhs_tensor, lhs_tensor}, allocator);

  EXPECT_FALSE(preceding_work_completed.load(std::memory_order_acquire));
  ASSERT_EQ(outputs.size(), 1U);
  EXPECT_EQ(outputs[0].name(), "sum");
  EXPECT_EQ(counting->allocation_count(), 2U);
  {
    Tensor scratch = Tensor::Allocate(MakeCudaFloatDesc({2, 3}), allocator);
    EXPECT_EQ(counting->allocation_count(), 3U);
  }
  stream->Synchronize();
  ExpectFloatValues(outputs[0], Sum(lhs, rhs));

  std::vector<Tensor> second_outputs =
      backend->Infer({rhs_tensor, lhs_tensor}, allocator);
  EXPECT_EQ(counting->allocation_count(), 3U);
  stream->Synchronize();
  ASSERT_EQ(second_outputs.size(), 1U);
  EXPECT_EQ(second_outputs[0].name(), "sum");
  ExpectFloatValues(second_outputs[0], Sum(lhs, rhs));
}

TEST(TensorRtBackendTest, RunsMemoryEngineWithCudaInputsAndOutputs) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "CUDA device is unavailable";
  }

  BackendPtr backend = CreateBackend(DynamicAddEngineModel(), CudaDevice());
  const ModelInfo& info = backend->model_info();

  ASSERT_EQ(info.inputs.size(), 2U);
  EXPECT_EQ(info.inputs[0].name, "lhs");
  EXPECT_EQ(info.inputs[0].data_type, DataType::kFloat32);
  EXPECT_EQ(info.inputs[0].shape, std::vector<int64_t>({-1, 3}));
  EXPECT_EQ(info.inputs[1].name, "rhs");
  EXPECT_EQ(info.inputs[1].shape, std::vector<int64_t>({-1, 3}));
  ASSERT_EQ(info.outputs.size(), 1U);
  EXPECT_EQ(info.outputs[0].name, "sum");
  EXPECT_EQ(info.outputs[0].shape, std::vector<int64_t>({-1, 3}));
  ASSERT_EQ(info.profiles.size(), 1U);
  ASSERT_EQ(info.profiles[0].inputs.size(), 2U);
  EXPECT_EQ(info.profiles[0].inputs[0].min_shape, std::vector<int64_t>({1, 3}));
  EXPECT_EQ(info.profiles[0].inputs[0].opt_shape, std::vector<int64_t>({2, 3}));
  EXPECT_EQ(info.profiles[0].inputs[0].max_shape, std::vector<int64_t>({4, 3}));

  ExpectDynamicAddOutput(backend.get(), 1);
  ExpectDynamicAddOutput(backend.get(), 2);
}

TEST(TensorRtBackendTest, ReusesOutputsWithExplicitPooledAllocator) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "CUDA device is unavailable";
  }

  BackendPtr backend = CreateBackend(DynamicAddEngineModel(), CudaDevice());
  PooledTensorAllocator allocator;
  const std::vector<float> lhs = Sequence(6, 1.0F);
  const std::vector<float> rhs = Sequence(6, 100.0F);
  Tensor lhs_tensor = MakeCudaFloatTensor("lhs", {2, 3}, lhs);
  Tensor rhs_tensor = MakeCudaFloatTensor("rhs", {2, 3}, rhs);

  void* first_output_data = nullptr;
  {
    std::vector<Tensor> outputs =
        backend->Infer({lhs_tensor, rhs_tensor}, allocator);
    ASSERT_EQ(outputs.size(), 1U);
    first_output_data = outputs.front().data();
    ExpectFloatValues(outputs.front(), Sum(lhs, rhs));
  }

  std::vector<Tensor> outputs =
      backend->Infer({lhs_tensor, rhs_tensor}, allocator);
  ASSERT_EQ(outputs.size(), 1U);
  EXPECT_EQ(outputs.front().data(), first_output_data);
  ExpectFloatValues(outputs.front(), Sum(lhs, rhs));
}

TEST(TensorRtBackendTest, BindsRequestedMultiOutputsInOrder) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "CUDA device is unavailable";
  }

  BackendPtr backend =
      CreateBackend(DynamicAddSubEngineModel(), CudaDevice(), {"diff", "sum"});
  const ModelInfo& info = backend->model_info();

  ASSERT_EQ(info.outputs.size(), 2U);
  EXPECT_EQ(info.outputs[0].name, "diff");
  EXPECT_EQ(info.outputs[1].name, "sum");

  const std::vector<float> lhs = Sequence(6, 10.0F);
  const std::vector<float> rhs = Sequence(6, 1.0F);
  Tensor lhs_tensor = MakeCudaFloatTensor("lhs", {2, 3}, lhs);
  Tensor rhs_tensor = MakeCudaFloatTensor("rhs", {2, 3}, rhs);

  std::vector<Tensor> outputs = backend->Infer({lhs_tensor, rhs_tensor});

  ASSERT_EQ(outputs.size(), 2U);
  EXPECT_EQ(outputs[0].name(), "diff");
  EXPECT_EQ(outputs[1].name(), "sum");
  ExpectFloatValues(outputs[0], Diff(lhs, rhs));
  ExpectFloatValues(outputs[1], Sum(lhs, rhs));
}

TEST(TensorRtBackendTest, CopiesCpuInputsToExecutionDevice) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "CUDA device is unavailable";
  }

  BackendPtr backend = CreateBackend(DynamicAddEngineModel(), CudaDevice());
  const std::vector<float> lhs = Sequence(6, 1.0F);
  const std::vector<float> rhs = Sequence(6, 100.0F);
  Tensor lhs_tensor = MakeCpuFloatTensor("lhs", {2, 3}, lhs);
  Tensor rhs_tensor = MakeCpuFloatTensor("rhs", {2, 3}, rhs);
  PooledTensorAllocator allocator;

  std::vector<Tensor> outputs =
      backend->Infer({rhs_tensor, lhs_tensor}, allocator);

  ASSERT_EQ(outputs.size(), 1U);
  EXPECT_EQ(outputs[0].device().type, DeviceType::kCuda);
  ExpectFloatValues(outputs[0], Sum(lhs, rhs));
}

TEST(TensorRtBackendTest, LoadsEngineFromPlanAndEnginePaths) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "CUDA device is unavailable";
  }

  std::shared_ptr<std::vector<std::uint8_t>> engine = DynamicAddEngineBytes();
  for (const char* extension : {".plan", ".engine"}) {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        (std::string("mw_infer_dynamic_add") + extension);
    {
      std::ofstream output(path, std::ios::binary);
      ASSERT_TRUE(output);
      output.write(reinterpret_cast<const char*>(engine->data()),
                   static_cast<std::streamsize>(engine->size()));
    }

    BackendPtr backend = CreateBackend(ModelFromPath(path), CudaDevice());
    ExpectDynamicAddOutput(backend.get(), 2);

    std::error_code error;
    std::filesystem::remove(path, error);
  }
}

TEST(TensorRtBackendTest, RejectsInvalidInputs) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "CUDA device is unavailable";
  }

  BackendPtr backend = CreateBackend(DynamicAddEngineModel(), CudaDevice());
  Tensor lhs = MakeCudaFloatTensor("lhs", {2, 3}, Sequence(6, 1.0F));
  Tensor rhs_bad_shape =
      MakeCudaFloatTensor("rhs", {5, 3}, Sequence(15, 100.0F));
  Tensor rhs = MakeCudaFloatTensor("rhs", {2, 3}, Sequence(6, 100.0F));
  Tensor duplicate = MakeCudaFloatTensor("lhs", {2, 3}, Sequence(6, 100.0F));
  Tensor unknown = MakeCudaFloatTensor("unknown", {2, 3}, Sequence(6, 100.0F));
  Tensor nameless = MakeCudaFloatTensor("", {2, 3}, Sequence(6, 100.0F));
  Tensor rhs_bad_rank =
      MakeCudaFloatTensor("rhs", {2, 3, 1}, Sequence(6, 100.0F));

  EXPECT_THROW(static_cast<void>(backend->Infer({lhs, rhs_bad_shape})),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(backend->Infer({lhs, duplicate})),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(backend->Infer({lhs, unknown})),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(backend->Infer({lhs, nameless})),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(backend->Infer({lhs, rhs_bad_rank})),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(backend->Infer({lhs})), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(CreateBackend(DynamicAddEngineModel(),
                                               CudaDevice(), {"missing"})),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(CreateBackend(DynamicAddEngineModel(),
                                               CudaDevice(), {"sum", "sum"})),
               std::invalid_argument);
}

}  // namespace
}  // namespace mw::infer
