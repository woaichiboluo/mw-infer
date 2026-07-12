#include <gtest/gtest.h>

#if defined(MW_INFER_HAS_CUDA_RUNTIME)
#include <cuda_runtime_api.h>
#endif

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "mw/infer/runtime/pipeline/pipeline.h"

namespace mw::infer {

struct PipelineTestImage {
  int value = 0;
};

struct LifetimeTestImage {
  std::shared_ptr<int> owner;
};

template <>
struct RawImageConverter<PipelineTestImage> {
  static RawImage Convert(PipelineTestImage image) {
    ImageDesc desc;
    desc.size = ImageSize{1, 1};
    desc.pixel_format = PixelFormat::kGray;
    desc.data_type = DataType::kUInt8;
    desc.channels = 1;
    desc.memory_kind = ImageMemoryKind::kHost;
    return RawImage::FromHandle(std::move(desc), ImageHandleKind::kNone,
                                std::move(image));
  }
};

template <>
struct RawImageConverter<LifetimeTestImage> {
  static RawImage Convert(LifetimeTestImage image) {
    ImageDesc desc;
    desc.size = ImageSize{1, 1};
    desc.pixel_format = PixelFormat::kGray;
    desc.data_type = DataType::kUInt8;
    desc.channels = 1;
    desc.memory_kind = ImageMemoryKind::kHost;
    return RawImage::FromHandle(std::move(desc), ImageHandleKind::kNone,
                                std::move(image));
  }
};

namespace {

class InspectPipeline final : public Pipeline<int> {
 public:
  InspectPipeline() : Pipeline(Device{DeviceType::kCpu, 0}) {}

  const std::vector<void*>& allocation_addresses() const {
    return allocation_addresses_;
  }

  bool uses_cpu_stream() { return stream().is_cpu(); }

 protected:
  BatchResult Process(const RawImageBatch& images) override {
    TensorDesc desc;
    desc.info.data_type = DataType::kInt32;
    desc.info.shape = {static_cast<int64_t>(images.size()), 1};
    desc.device = device();
    Tensor scratch = Tensor::Allocate(std::move(desc), allocator());
    allocation_addresses_.push_back(scratch.data());

    BatchResult results;
    results.reserve(images.size());
    for (const RawImage& image : images.images()) {
      const auto* value = static_cast<const PipelineTestImage*>(image.handle());
      results.push_back(value->value);
    }
    return results;
  }

 private:
  std::vector<void*> allocation_addresses_;
};

class InvalidResultCountPipeline final : public Pipeline<int> {
 public:
  InvalidResultCountPipeline() : Pipeline(Device{DeviceType::kCpu, 0}) {}

 protected:
  BatchResult Process(const RawImageBatch&) override { return {}; }
};

struct BackendObservation {
  std::vector<std::shared_ptr<ExecutionStream>> streams;
  std::vector<std::vector<std::string>> output_names;
  std::vector<void*> output_addresses;
  int infer_calls = 0;
};

class FakeBackend final : public IBackend {
 public:
  FakeBackend(Model model, Device device,
              std::shared_ptr<BackendObservation> observation)
      : IBackend(std::move(model), device),
        observation_(std::move(observation)) {}

  std::vector<Tensor> Infer(const std::vector<Tensor>& inputs) override {
    return Infer(inputs, TensorAllocator::Default());
  }

  std::vector<Tensor> Infer(const std::vector<Tensor>& inputs,
                            TensorAllocator& allocator) override {
    if (inputs.size() != 1) {
      throw std::invalid_argument("Fake backend expects one input");
    }
    Tensor output = Tensor::Allocate(inputs.front().desc(), allocator);
    observation_->output_addresses.push_back(output.data());
    for (std::size_t index = 0; index < output.element_count(); ++index) {
      output.data<std::int32_t>()[index] =
          inputs.front().data<std::int32_t>()[index] + 1;
    }
    ++observation_->infer_calls;
    return {std::move(output)};
  }

 private:
  std::shared_ptr<BackendObservation> observation_;
};

class FakeBackendAdapter final : public BackendAdapter {
 public:
  explicit FakeBackendAdapter(std::shared_ptr<BackendObservation> observation)
      : observation_(std::move(observation)) {}

  bool Supports(const Model&, Device execution_device) const override {
    return execution_device.type == DeviceType::kCpu;
  }

  BackendPtr Create(Model model, Device execution_device,
                    std::vector<std::string> output_names) const override {
    observation_->output_names.push_back(std::move(output_names));
    return std::make_unique<FakeBackend>(std::move(model), execution_device,
                                         observation_);
  }

  BackendPtr Create(
      Model model, Device execution_device,
      std::vector<std::string> output_names,
      std::shared_ptr<ExecutionStream> execution_stream) const override {
    observation_->streams.push_back(execution_stream);
    return Create(std::move(model), execution_device, std::move(output_names));
  }

 private:
  std::shared_ptr<BackendObservation> observation_;
};

BackendFactory MakeFakeBackendFactory(
    const std::shared_ptr<BackendObservation>& observation) {
  std::vector<std::unique_ptr<BackendAdapter>> adapters;
  adapters.push_back(std::make_unique<FakeBackendAdapter>(observation));
  return BackendFactory(std::move(adapters));
}

class MultiModelPipeline final : public Pipeline<int> {
 public:
  explicit MultiModelPipeline(
      const std::shared_ptr<BackendObservation>& observation)
      : Pipeline(Device{DeviceType::kCpu, 0},
                 MakeFakeBackendFactory(observation)) {
    first_model_ = AddModel(Model{}, {"first_output"});
    second_model_ = AddModel(Model{}, {"second_output"});
  }

 protected:
  BatchResult Process(const RawImageBatch& images) override {
    TensorDesc desc;
    desc.info.data_type = DataType::kInt32;
    desc.info.shape = {static_cast<int64_t>(images.size())};
    desc.device = device();
    Tensor input = Tensor::Allocate(std::move(desc), allocator());
    for (std::size_t index = 0; index < images.size(); ++index) {
      const auto* value =
          static_cast<const PipelineTestImage*>(images.image(index).handle());
      input.data<std::int32_t>()[index] = value->value;
    }

    ModelOutputs first_outputs = InferModel(first_model_, input);
    ModelOutputs second_outputs =
        InferModel(second_model_, first_outputs.front());
    const Tensor& output = second_outputs.front();
    return std::vector<int>(
        output.data<std::int32_t>(),
        output.data<std::int32_t>() + output.element_count());
  }

 private:
  ModelId first_model_ = 0;
  ModelId second_model_ = 0;
};

TEST(ExecutionStreamTest, CpuStreamSynchronizesAsNoOp) {
  ExecutionStream stream(Device{DeviceType::kCpu, 0});

  EXPECT_TRUE(stream.is_cpu());
  EXPECT_FALSE(stream.is_cuda());
  EXPECT_NO_THROW(stream.Synchronize());
}

TEST(PipelineTest, ConvertsTemplatedInputAndPreservesBatchOrder) {
  InspectPipeline pipeline;

  const std::vector<int> results =
      pipeline.Infer(std::vector<PipelineTestImage>{{11}, {22}, {33}});

  EXPECT_TRUE(pipeline.uses_cpu_stream());
  EXPECT_EQ(results, std::vector<int>({11, 22, 33}));
}

TEST(PipelineTest, AcceptsRawImageBatchDirectly) {
  InspectPipeline pipeline;
  RawImageBatch images =
      ToRawImageBatch(std::vector<PipelineTestImage>{{7}, {8}});

  const std::vector<int> results = pipeline.Infer(std::move(images));

  EXPECT_EQ(results, std::vector<int>({7, 8}));
}

TEST(PipelineTest, RejectsEmptyInputBatch) {
  InspectPipeline pipeline;

  EXPECT_THROW(static_cast<void>(pipeline.Infer(RawImageBatch{})),
               std::invalid_argument);
}

TEST(PipelineTest, ReusesTensorAllocatorAcrossSequentialRuns) {
  InspectPipeline pipeline;

  static_cast<void>(pipeline.Infer(std::vector<PipelineTestImage>{{1}, {2}}));
  static_cast<void>(pipeline.Infer(std::vector<PipelineTestImage>{{3}, {4}}));

  ASSERT_EQ(pipeline.allocation_addresses().size(), 2U);
  EXPECT_EQ(pipeline.allocation_addresses()[0],
            pipeline.allocation_addresses()[1]);
}

TEST(PipelineTest, RejectsResultCountThatDoesNotMatchBatch) {
  InvalidResultCountPipeline pipeline;

  EXPECT_THROW(
      static_cast<void>(pipeline.Infer(std::vector<PipelineTestImage>{{1}})),
      std::logic_error);
}

TEST(PipelineTest, RunsMultipleModelsOnTheSharedStream) {
  auto observation = std::make_shared<BackendObservation>();
  MultiModelPipeline pipeline(observation);

  const std::vector<int> results =
      pipeline.Infer(std::vector<PipelineTestImage>{{3}, {8}});
  const std::vector<int> second_results =
      pipeline.Infer(std::vector<PipelineTestImage>{{10}, {20}});

  EXPECT_EQ(results, std::vector<int>({5, 10}));
  EXPECT_EQ(second_results, std::vector<int>({12, 22}));
  EXPECT_EQ(observation->infer_calls, 4);
  ASSERT_EQ(observation->streams.size(), 2U);
  EXPECT_EQ(observation->streams[0], observation->streams[1]);
  EXPECT_TRUE(observation->streams[0]->is_cpu());
  EXPECT_EQ(observation->output_names,
            std::vector<std::vector<std::string>>(
                {{"first_output"}, {"second_output"}}));
  ASSERT_EQ(observation->output_addresses.size(), 4U);
  EXPECT_EQ(observation->output_addresses[0], observation->output_addresses[2]);
  EXPECT_EQ(observation->output_addresses[1], observation->output_addresses[3]);
}

TEST(PipelineTest, RejectsCudaInputForCpuPipeline) {
  InspectPipeline pipeline;
  ImageDesc desc;
  desc.size = ImageSize{1, 1};
  desc.pixel_format = PixelFormat::kGray;
  desc.data_type = DataType::kUInt8;
  desc.channels = 1;
  desc.memory_kind = ImageMemoryKind::kCuda;
  desc.device_id = 0;
  std::vector<RawImage> images;
  images.push_back(RawImage::FromHandle(std::move(desc), ImageHandleKind::kNone,
                                        PipelineTestImage{1}));

  EXPECT_THROW(
      static_cast<void>(pipeline.Infer(RawImageBatch(std::move(images)))),
      std::invalid_argument);
}

#if defined(MW_INFER_HAS_CUDA_RUNTIME)

bool HasUsableCudaDevice() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

void CUDART_CB MarkCompleteAfterDelay(void* opaque) noexcept {
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  static_cast<std::atomic<bool>*>(opaque)->store(true,
                                                 std::memory_order_release);
}

struct LifetimeProbe {
  std::weak_ptr<int> owner;
  std::atomic<bool> alive_during_callback{false};
};

void CUDART_CB CheckInputLifetime(void* opaque) noexcept {
  auto* probe = static_cast<LifetimeProbe*>(opaque);
  probe->alive_during_callback.store(!probe->owner.expired(),
                                     std::memory_order_release);
}

class CudaCallbackPipeline final : public Pipeline<int> {
 public:
  CudaCallbackPipeline(std::atomic<bool>& completed, bool throw_after_launch)
      : Pipeline(Device{DeviceType::kCuda, 0}),
        completed_(completed),
        throw_after_launch_(throw_after_launch) {}

 protected:
  BatchResult Process(const RawImageBatch& images) override {
    const cudaError_t status = cudaLaunchHostFunc(
        stream().cuda_handle(), MarkCompleteAfterDelay, &completed_);
    if (status != cudaSuccess) {
      throw std::runtime_error("cudaLaunchHostFunc failed");
    }
    if (throw_after_launch_) {
      throw std::runtime_error("pipeline stage failed");
    }
    return BatchResult(images.size(), 1);
  }

 private:
  std::atomic<bool>& completed_;
  bool throw_after_launch_ = false;
};

class CudaLifetimePipeline final : public Pipeline<int> {
 public:
  explicit CudaLifetimePipeline(LifetimeProbe& probe)
      : Pipeline(Device{DeviceType::kCuda, 0}), probe_(probe) {}

 protected:
  BatchResult Process(const RawImageBatch&) override {
    const cudaError_t status =
        cudaLaunchHostFunc(stream().cuda_handle(), CheckInputLifetime, &probe_);
    if (status != cudaSuccess) {
      throw std::runtime_error("cudaLaunchHostFunc failed");
    }
    throw std::runtime_error("pipeline stage failed");
  }

 private:
  LifetimeProbe& probe_;
};

TEST(ExecutionStreamTest, CudaStreamSynchronizesQueuedWork) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "CUDA device is unavailable";
  }

  std::atomic<bool> completed{false};
  ExecutionStream stream(Device{DeviceType::kCuda, 0});
  ASSERT_EQ(cudaLaunchHostFunc(stream.cuda_handle(), MarkCompleteAfterDelay,
                               &completed),
            cudaSuccess);

  stream.Synchronize();

  EXPECT_TRUE(completed.load(std::memory_order_acquire));
}

TEST(ExecutionStreamTest, UsesRequestedCudaStreamMode) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "CUDA device is unavailable";
  }

  ExecutionStream blocking(Device{DeviceType::kCuda, 0},
                           CudaStreamMode::kBlocking);
  ExecutionStream nonblocking(Device{DeviceType::kCuda, 0},
                              CudaStreamMode::kNonBlocking);
  unsigned int blocking_flags = 0;
  unsigned int nonblocking_flags = 0;

  ASSERT_EQ(cudaStreamGetFlags(blocking.cuda_handle(), &blocking_flags),
            cudaSuccess);
  ASSERT_EQ(cudaStreamGetFlags(nonblocking.cuda_handle(), &nonblocking_flags),
            cudaSuccess);
  EXPECT_EQ(blocking_flags, cudaStreamDefault);
  EXPECT_EQ(nonblocking_flags, cudaStreamNonBlocking);
}

TEST(PipelineTest, SynchronizesCudaStreamBeforeReturning) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "CUDA device is unavailable";
  }

  std::atomic<bool> completed{false};
  CudaCallbackPipeline pipeline(completed, false);

  const std::vector<int> results =
      pipeline.Infer(std::vector<PipelineTestImage>{{1}, {2}});

  EXPECT_TRUE(completed.load(std::memory_order_acquire));
  EXPECT_EQ(results, std::vector<int>({1, 1}));
}

TEST(PipelineTest, SynchronizesCudaStreamBeforePropagatingStageFailure) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "CUDA device is unavailable";
  }

  std::atomic<bool> completed{false};
  CudaCallbackPipeline pipeline(completed, true);

  EXPECT_THROW(
      static_cast<void>(pipeline.Infer(std::vector<PipelineTestImage>{{1}})),
      std::runtime_error);
  EXPECT_TRUE(completed.load(std::memory_order_acquire));
}

TEST(PipelineTest, KeepsInputAliveUntilFailurePathIsSynchronized) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "CUDA device is unavailable";
  }

  LifetimeProbe probe;
  std::shared_ptr<int> owner = std::make_shared<int>(1);
  probe.owner = owner;
  std::vector<LifetimeTestImage> images{{owner}};
  owner.reset();
  CudaLifetimePipeline pipeline(probe);

  EXPECT_THROW(static_cast<void>(pipeline.Infer(std::move(images))),
               std::runtime_error);
  EXPECT_TRUE(probe.alive_during_callback.load(std::memory_order_acquire));
  EXPECT_TRUE(probe.owner.expired());
}

#endif

}  // namespace
}  // namespace mw::infer
