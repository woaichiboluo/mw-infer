#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "mw/infer/runtime/process/channel.h"
#include "mw/infer/runtime/process/normalize.h"

#if defined(MW_INFER_HAS_CUDA_PREPROCESS)
#include <cuda_runtime_api.h>
#endif

namespace mw::infer {
namespace {

Tensor MakeCpuFloatTensor(std::vector<int64_t> shape,
                          const std::vector<float>& data,
                          std::string name = {}) {
  TensorDesc desc;
  desc.info.name = std::move(name);
  desc.info.data_type = DataType::kFloat32;
  desc.info.shape = std::move(shape);
  desc.device = Device{DeviceType::kCpu, 0};
  Tensor tensor = Tensor::Allocate(std::move(desc));
  std::memcpy(tensor.data(), data.data(), tensor.bytes());
  return tensor;
}

std::vector<float> CopyCpuFloats(const Tensor& tensor) {
  EXPECT_EQ(tensor.device().type, DeviceType::kCpu);
  EXPECT_EQ(tensor.data_type(), DataType::kFloat32);
  const auto* data = static_cast<const float*>(tensor.data());
  return std::vector<float>(data, data + tensor.element_count());
}

void ExpectFloatVectorsNear(const std::vector<float>& actual,
                            const std::vector<float>& expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t index = 0; index < actual.size(); ++index) {
    EXPECT_NEAR(actual[index], expected[index], 1.0e-6F);
  }
}

TEST(PreprocessTest, ReordersBchwChannels) {
  Tensor input = MakeCpuFloatTensor({2, 3, 1, 2},
                                    {10.0F, 20.0F, 30.0F, 40.0F, 50.0F, 60.0F,
                                     1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F},
                                    "data");

  Tensor output = ReorderChannels(input, {2, 1, 0}, TensorLayout::kBchw);

  EXPECT_EQ(output.name(), "data");
  EXPECT_EQ(output.shape(), std::vector<int64_t>({2, 3, 1, 2}));
  EXPECT_EQ(CopyCpuFloats(output),
            std::vector<float>({50.0F, 60.0F, 30.0F, 40.0F, 10.0F, 20.0F, 5.0F,
                                6.0F, 3.0F, 4.0F, 1.0F, 2.0F}));
}

TEST(PreprocessTest, ReordersBhwcChannels) {
  Tensor input = MakeCpuFloatTensor(
      {1, 1, 2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}, "data");

  Tensor output = ReorderChannels(input, {2, 1, 0}, TensorLayout::kBhwc);

  EXPECT_EQ(output.name(), "data");
  EXPECT_EQ(output.shape(), std::vector<int64_t>({1, 1, 2, 3}));
  EXPECT_EQ(CopyCpuFloats(output),
            std::vector<float>({3.0F, 2.0F, 1.0F, 6.0F, 5.0F, 4.0F}));
}

TEST(PreprocessTest, NormalizesBchwChannels) {
  Tensor input = MakeCpuFloatTensor(
      {1, 3, 1, 2}, {10.0F, 20.0F, 30.0F, 40.0F, 50.0F, 60.0F}, "data");

  Tensor output = Normalize(input, {1.0F, 2.0F, 3.0F}, {1.0F, 2.0F, 4.0F}, 0.1F,
                            TensorLayout::kBchw);

  EXPECT_EQ(output.name(), "data");
  ExpectFloatVectorsNear(CopyCpuFloats(output),
                         {0.0F, 1.0F, 0.5F, 1.0F, 0.5F, 0.75F});
}

TEST(PreprocessTest, NormalizesBhwcChannels) {
  Tensor input = MakeCpuFloatTensor(
      {1, 1, 2, 3}, {10.0F, 20.0F, 30.0F, 40.0F, 50.0F, 60.0F}, "data");

  Tensor output = Normalize(input, {1.0F, 2.0F, 3.0F}, {1.0F, 2.0F, 4.0F}, 0.1F,
                            TensorLayout::kBhwc);

  EXPECT_EQ(output.name(), "data");
  ExpectFloatVectorsNear(CopyCpuFloats(output),
                         {0.0F, 0.0F, 0.0F, 3.0F, 1.5F, 0.75F});
}

TEST(PreprocessTest, RejectsInvalidInputs) {
  Tensor input = MakeCpuFloatTensor({1, 3, 1, 2},
                                    {10.0F, 20.0F, 30.0F, 40.0F, 50.0F, 60.0F});
  Tensor bad_rank =
      MakeCpuFloatTensor({3, 2}, {10.0F, 20.0F, 30.0F, 40.0F, 50.0F, 60.0F});

  EXPECT_THROW(static_cast<void>(ReorderChannels(Tensor{}, {2, 1, 0})),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(ReorderChannels(bad_rank, {1, 0})),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(ReorderChannels(input, {2, 2, 0})),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(Normalize(input, {0.0F}, {1.0F})),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(
                   Normalize(input, {0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 1.0F})),
               std::invalid_argument);
}

#if defined(MW_INFER_HAS_CUDA_PREPROCESS)

bool HasUsableCudaDevice() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

void CUDART_CB DelayCudaStream(void* user_data) {
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  static_cast<std::atomic<bool>*>(user_data)->store(true,
                                                    std::memory_order_release);
}

std::vector<float> CopyCudaFloatsToHost(const Tensor& tensor) {
  EXPECT_EQ(tensor.device().type, DeviceType::kCuda);
  return CopyCpuFloats(tensor.CopyTo(Device{DeviceType::kCpu, 0}));
}

TEST(PreprocessTest, ReordersCudaBchwChannels) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "CUDA preprocess is unavailable";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  Tensor input =
      MakeCpuFloatTensor({1, 3, 1, 2},
                         {10.0F, 20.0F, 30.0F, 40.0F, 50.0F, 60.0F}, "data")
          .CopyTo(Device{DeviceType::kCuda, 0});

  Tensor output = ReorderChannels(input, {2, 1, 0}, TensorLayout::kBchw);

  EXPECT_EQ(output.name(), "data");
  EXPECT_EQ(output.device().type, DeviceType::kCuda);
  EXPECT_EQ(CopyCudaFloatsToHost(output),
            std::vector<float>({50.0F, 60.0F, 30.0F, 40.0F, 10.0F, 20.0F}));
}

TEST(PreprocessTest, NormalizesCudaBhwcChannels) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "CUDA preprocess is unavailable";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  Tensor input =
      MakeCpuFloatTensor({1, 1, 2, 3},
                         {10.0F, 20.0F, 30.0F, 40.0F, 50.0F, 60.0F}, "data")
          .CopyTo(Device{DeviceType::kCuda, 0});

  Tensor output = Normalize(input, {1.0F, 2.0F, 3.0F}, {1.0F, 2.0F, 4.0F}, 0.1F,
                            TensorLayout::kBhwc);

  EXPECT_EQ(output.name(), "data");
  EXPECT_EQ(output.device().type, DeviceType::kCuda);
  ExpectFloatVectorsNear(CopyCudaFloatsToHost(output),
                          {0.0F, 0.0F, 0.0F, 3.0F, 1.5F, 0.75F});
}

TEST(PreprocessTest, EnqueuesCudaOperationsOnExecutionStream) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "CUDA preprocess is unavailable";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  Tensor input =
      MakeCpuFloatTensor({1, 1, 2, 3},
                         {10.0F, 20.0F, 30.0F, 40.0F, 50.0F, 60.0F}, "data")
          .CopyTo(Device{DeviceType::kCuda, 0});
  ExecutionStream stream(Device{DeviceType::kCuda, 0});
  PooledTensorAllocator allocator;
  {
    Tensor warm_reordered = ReorderChannels(
        input, {2, 1, 0}, stream, TensorLayout::kBhwc, allocator);
    Tensor warm_output = Normalize(
        warm_reordered, {1.0F, 2.0F, 3.0F}, {1.0F, 2.0F, 4.0F}, stream, 0.1F,
        TensorLayout::kBhwc, allocator);
    stream.Synchronize();
  }

  std::atomic<bool> delay_completed{false};
  ASSERT_EQ(cudaLaunchHostFunc(stream.cuda_handle(), DelayCudaStream,
                               &delay_completed),
            cudaSuccess);

  input = ReorderChannels(input, {2, 1, 0}, stream, TensorLayout::kBhwc,
                          allocator);
  input = Normalize(input, {1.0F, 2.0F, 3.0F}, {1.0F, 2.0F, 4.0F}, stream,
                    0.1F, TensorLayout::kBhwc, allocator);
  EXPECT_FALSE(delay_completed.load(std::memory_order_acquire));
  stream.Synchronize();

  EXPECT_TRUE(delay_completed.load(std::memory_order_acquire));
  EXPECT_EQ(input.capacity_bytes(), input.bytes());
  ExpectFloatVectorsNear(CopyCudaFloatsToHost(input),
                         {2.0F, 0.0F, -0.5F, 5.0F, 1.5F, 0.25F});
}

#endif

}  // namespace
}  // namespace mw::infer
