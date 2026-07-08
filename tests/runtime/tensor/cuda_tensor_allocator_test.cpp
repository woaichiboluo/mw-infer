#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

#include "mw/infer/runtime/tensor/tensor_allocator.h"

namespace mw::infer {
namespace {

bool HasUsableCudaDevice() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

TensorDesc MakeCudaDesc() {
  TensorDesc desc;
  desc.info.name = "images";
  desc.info.data_type = DataType::kFloat32;
  desc.info.shape = {1, 3, 2, 2};
  desc.device = Device{DeviceType::kCuda, 0};
  return desc;
}

TensorDesc MakeCudaDescWithShape(std::vector<int64_t> shape) {
  TensorDesc desc = MakeCudaDesc();
  desc.info.shape = std::move(shape);
  return desc;
}

TensorDesc MakeHostDescWithShape(std::vector<int64_t> shape) {
  TensorDesc desc = MakeCudaDescWithShape(std::move(shape));
  desc.device = Device{DeviceType::kCpu, 0};
  return desc;
}

Tensor MakeHostTensor(std::vector<int64_t> shape,
                      const std::vector<float>& values) {
  Tensor tensor = Tensor::Allocate(MakeHostDescWithShape(std::move(shape)));
  std::memcpy(tensor.data(), values.data(), tensor.bytes());
  return tensor;
}

std::vector<float> CopyCpuFloats(const Tensor& tensor) {
  EXPECT_EQ(tensor.device().type, DeviceType::kCpu);
  EXPECT_EQ(tensor.data_type(), DataType::kFloat32);
  const auto* data = static_cast<const float*>(tensor.data());
  return std::vector<float>(data, data + tensor.element_count());
}

TEST(CudaTensorAllocatorTest, AllocatesCudaTensor) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "CUDA tensor allocation is unavailable";
  }

  Tensor tensor = Tensor::Allocate(MakeCudaDesc());

  ASSERT_FALSE(tensor.empty());
  EXPECT_EQ(tensor.name(), "images");
  EXPECT_EQ(tensor.data_type(), DataType::kFloat32);
  EXPECT_EQ(tensor.shape(), std::vector<int64_t>({1, 3, 2, 2}));
  EXPECT_EQ(tensor.device().type, DeviceType::kCuda);
  EXPECT_EQ(tensor.device().id, 0);
  EXPECT_EQ(tensor.bytes(), 48U);
  EXPECT_EQ(tensor.capacity_bytes(), 48U);
  EXPECT_NE(tensor.data(), nullptr);
  EXPECT_EQ(cudaMemset(tensor.data(), 0, tensor.bytes()), cudaSuccess);
}

TEST(CudaTensorAllocatorTest, TensorInterfaceSelectsCudaAdapterByDevice) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "CUDA tensor allocation is unavailable";
  }

  Tensor tensor = Tensor::Allocate(MakeCudaDesc());

  ASSERT_FALSE(tensor.empty());
  EXPECT_EQ(tensor.device().type, DeviceType::kCuda);
  EXPECT_EQ(tensor.device().id, 0);
  EXPECT_EQ(tensor.bytes(), 48U);
  EXPECT_EQ(tensor.capacity_bytes(), 48U);
  EXPECT_NE(tensor.data(), nullptr);
  EXPECT_EQ(cudaMemset(tensor.data(), 0, tensor.bytes()), cudaSuccess);
}

TEST(CudaTensorAllocatorTest, CopiesHostTensorToCudaAndBack) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "CUDA tensor copy is unavailable";
  }

  Tensor host = MakeHostTensor({2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});

  Tensor device = host.CopyTo(Device{DeviceType::kCuda, 0});
  Tensor roundtrip = device.CopyTo(Device{DeviceType::kCpu, 0});

  EXPECT_EQ(device.name(), host.name());
  EXPECT_EQ(device.data_type(), host.data_type());
  EXPECT_EQ(device.shape(), host.shape());
  EXPECT_EQ(device.device().type, DeviceType::kCuda);
  EXPECT_EQ(roundtrip.device().type, DeviceType::kCpu);
  EXPECT_EQ(CopyCpuFloats(roundtrip),
            std::vector<float>({1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}));
}

TEST(CudaTensorAllocatorTest, CopiesCudaTensorToCudaDevice) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "CUDA tensor copy is unavailable";
  }

  Tensor host = MakeHostTensor({2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
  Tensor device = host.CopyTo(Device{DeviceType::kCuda, 0});

  Tensor device_copy = device.CopyTo(Device{DeviceType::kCuda, 0});
  Tensor roundtrip = device_copy.CopyTo(Device{DeviceType::kCpu, 0});

  EXPECT_EQ(device_copy.device().type, DeviceType::kCuda);
  EXPECT_EQ(device_copy.device().id, 0);
  EXPECT_EQ(device_copy.shape(), host.shape());
  EXPECT_NE(device_copy.data(), device.data());
  EXPECT_EQ(CopyCpuFloats(roundtrip),
            std::vector<float>({1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}));
}

TEST(CudaTensorAllocatorTest, TensorBufferReusesAndGrowsLikeVectorReserve) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "CUDA tensor allocation is unavailable";
  }

  TensorBuffer buffer;

  Tensor first = buffer.Ensure(MakeCudaDescWithShape({4, 3, 2, 2}));
  void* first_data = first.data();
  EXPECT_EQ(first.bytes(), 192U);
  EXPECT_EQ(first.capacity_bytes(), 192U);
  EXPECT_EQ(buffer.capacity_bytes(), 192U);

  Tensor smaller = buffer.Ensure(MakeCudaDescWithShape({2, 3, 2, 2}));
  EXPECT_EQ(smaller.data(), first_data);
  EXPECT_EQ(smaller.bytes(), 96U);
  EXPECT_EQ(smaller.capacity_bytes(), 192U);
  EXPECT_EQ(buffer.capacity_bytes(), 192U);

  Tensor larger = buffer.Ensure(MakeCudaDescWithShape({8, 3, 2, 2}));
  EXPECT_NE(larger.data(), nullptr);
  EXPECT_EQ(larger.bytes(), 384U);
  EXPECT_EQ(larger.capacity_bytes(), 384U);
  EXPECT_EQ(buffer.capacity_bytes(), 384U);
  EXPECT_EQ(cudaMemset(larger.data(), 0, larger.bytes()), cudaSuccess);
}

TEST(CudaTensorAllocatorTest,
     TensorBufferKeepsCapacityStableAcrossLargerBatches) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "CUDA tensor allocation is unavailable";
  }

  TensorBuffer buffer;

  Tensor warmup = buffer.Ensure(MakeCudaDescWithShape({32, 3, 2, 2}));
  void* warmup_data = warmup.data();
  EXPECT_EQ(warmup.bytes(), 1536U);
  EXPECT_EQ(warmup.capacity_bytes(), 1536U);

  for (int64_t batch : {1, 2, 4, 8, 16, 32}) {
    Tensor tensor = buffer.Ensure(MakeCudaDescWithShape({batch, 3, 2, 2}));
    EXPECT_EQ(tensor.data(), warmup_data);
    EXPECT_EQ(tensor.bytes(),
              static_cast<std::size_t>(batch * 3 * 2 * 2 * sizeof(float)));
    EXPECT_EQ(tensor.capacity_bytes(), 1536U);
    EXPECT_EQ(buffer.capacity_bytes(), 1536U);
  }

  Tensor grown = buffer.Ensure(MakeCudaDescWithShape({64, 3, 2, 2}));
  void* grown_data = grown.data();
  EXPECT_NE(grown_data, nullptr);
  EXPECT_EQ(grown.bytes(), 3072U);
  EXPECT_EQ(grown.capacity_bytes(), 3072U);
  EXPECT_EQ(buffer.capacity_bytes(), 3072U);

  Tensor last;
  for (int64_t batch : {48, 17, 1, 64}) {
    last = buffer.Ensure(MakeCudaDescWithShape({batch, 3, 2, 2}));
    EXPECT_EQ(last.data(), grown_data);
    EXPECT_EQ(last.bytes(),
              static_cast<std::size_t>(batch * 3 * 2 * 2 * sizeof(float)));
    EXPECT_EQ(last.capacity_bytes(), 3072U);
    EXPECT_EQ(buffer.capacity_bytes(), 3072U);
  }

  EXPECT_EQ(cudaMemset(last.data(), 0, last.bytes()), cudaSuccess);
}

TEST(CudaTensorAllocatorTest,
     TensorBufferDefaultAllocatorSelectsAdapterByDevice) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "CUDA tensor allocation is unavailable";
  }

  TensorBuffer buffer;

  TensorDesc host_desc = MakeCudaDesc();
  host_desc.device = Device{DeviceType::kCpu, 0};
  Tensor host = buffer.Ensure(host_desc);
  EXPECT_EQ(host.device().type, DeviceType::kCpu);

  Tensor cuda = buffer.Ensure(MakeCudaDesc());
  EXPECT_EQ(cuda.device().type, DeviceType::kCuda);
  EXPECT_EQ(cuda.device().id, 0);
  EXPECT_EQ(cuda.bytes(), 48U);
  EXPECT_NE(cuda.data(), host.data());
}

TEST(CudaTensorAllocatorTest, RejectsInvalidCudaAllocationRequests) {
  TensorDesc desc = MakeCudaDesc();
  desc.device = Device{DeviceType::kCpu, 0};
  Tensor host = Tensor::Allocate(desc);
  EXPECT_EQ(host.device().type, DeviceType::kCpu);

  desc = MakeCudaDesc();
  desc.device = Device{DeviceType::kCuda, -1};
  TensorAllocator allocator;
  EXPECT_THROW(static_cast<void>(allocator.Allocate(desc)),
               std::invalid_argument);
}

}  // namespace
}  // namespace mw::infer
