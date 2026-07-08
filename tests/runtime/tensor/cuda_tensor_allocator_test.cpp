#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

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
