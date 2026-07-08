#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mw/infer/runtime/tensor/tensor_allocator.h"

namespace mw::infer {
namespace {

TensorDesc MakeDesc() {
  TensorDesc desc;
  desc.info.name = "images";
  desc.info.data_type = DataType::kFloat32;
  desc.info.shape = {1, 3, 2, 2};
  desc.device = Device{DeviceType::kCpu, 0};
  return desc;
}

TensorDesc MakeDescWithShape(std::vector<int64_t> shape) {
  TensorDesc desc = MakeDesc();
  desc.info.shape = std::move(shape);
  return desc;
}

class CpuOnlyTensorAllocationAdapter final : public TensorAllocationAdapter {
 public:
  bool Supports(Device device) const override {
    return device.type == DeviceType::kCpu;
  }

  Tensor Allocate(TensorDesc) const override {
    throw std::logic_error("CPU-only test adapter should not allocate");
  }
};

TensorAllocator MakeCpuOnlyTestAllocator() {
  std::vector<std::unique_ptr<TensorAllocationAdapter>> adapters;
  adapters.push_back(std::make_unique<CpuOnlyTensorAllocationAdapter>());
  return TensorAllocator(std::move(adapters));
}

TEST(TensorTest, ComputesShapeAndTypeSizes) {
  EXPECT_EQ(DataTypeSize(DataType::kUInt8), 1U);
  EXPECT_EQ(DataTypeSize(DataType::kFloat16), 2U);
  EXPECT_EQ(DataTypeSize(DataType::kFloat32), 4U);
  EXPECT_EQ(DataTypeSize(DataType::kInt64), 8U);
  EXPECT_EQ(DataTypeSize(DataType::kFloat64), 8U);
  EXPECT_EQ(ElementCount({1, 3, 2, 2}), 12U);
  EXPECT_EQ(TensorBytes(MakeDesc()), 48U);
}

TEST(TensorTest, WrapsExternalBuffer) {
  std::vector<float> buffer(12, 1.0F);
  TensorDesc desc = MakeDesc();

  Tensor tensor =
      Tensor::FromExternal(desc, buffer.data(), buffer.size() * sizeof(float));

  ASSERT_FALSE(tensor.empty());
  EXPECT_EQ(tensor.name(), "images");
  EXPECT_EQ(tensor.data_type(), DataType::kFloat32);
  EXPECT_EQ(tensor.shape(), std::vector<int64_t>({1, 3, 2, 2}));
  EXPECT_EQ(tensor.device().type, DeviceType::kCpu);
  EXPECT_EQ(tensor.data(), buffer.data());
  EXPECT_EQ(tensor.bytes(), 48U);
  EXPECT_EQ(tensor.capacity_bytes(), 48U);
  EXPECT_EQ(tensor.element_count(), 12U);
}

TEST(TensorTest, AllocatesHostTensor) {
  Tensor tensor = Tensor::Allocate(MakeDesc());

  ASSERT_FALSE(tensor.empty());
  EXPECT_EQ(tensor.name(), "images");
  EXPECT_EQ(tensor.data_type(), DataType::kFloat32);
  EXPECT_EQ(tensor.shape(), std::vector<int64_t>({1, 3, 2, 2}));
  EXPECT_EQ(tensor.device().type, DeviceType::kCpu);
  EXPECT_EQ(tensor.bytes(), 48U);
  EXPECT_EQ(tensor.capacity_bytes(), 48U);
  EXPECT_NE(tensor.data(), nullptr);

  auto* values = static_cast<float*>(tensor.data());
  values[0] = 1.25F;
  EXPECT_FLOAT_EQ(values[0], 1.25F);
}

TEST(TensorTest, AllocatesThroughTensorInterface) {
  Tensor tensor = Tensor::Allocate(MakeDesc());

  ASSERT_FALSE(tensor.empty());
  EXPECT_EQ(tensor.device().type, DeviceType::kCpu);
  EXPECT_EQ(tensor.bytes(), 48U);
  EXPECT_EQ(tensor.capacity_bytes(), 48U);
  EXPECT_NE(tensor.data(), nullptr);
}

TEST(TensorTest, CreatesViewWithinCapacity) {
  Tensor tensor = Tensor::Allocate(MakeDescWithShape({4, 3, 2, 2}));
  TensorDesc view_desc = MakeDescWithShape({2, 3, 2, 2});
  view_desc.info.name = "view";

  Tensor view = tensor.View(view_desc);

  EXPECT_EQ(view.name(), "view");
  EXPECT_EQ(view.shape(), std::vector<int64_t>({2, 3, 2, 2}));
  EXPECT_EQ(view.bytes(), 96U);
  EXPECT_EQ(view.capacity_bytes(), 192U);
  EXPECT_EQ(view.data(), tensor.data());
}

TEST(TensorTest, CopiesTensorToCpuDevice) {
  Tensor source = Tensor::Allocate(MakeDesc());
  auto* source_values = static_cast<float*>(source.data());
  for (std::size_t index = 0; index < source.element_count(); ++index) {
    source_values[index] = static_cast<float>(index);
  }

  Tensor copy = source.CopyTo(Device{DeviceType::kCpu, 0});

  EXPECT_EQ(copy.name(), source.name());
  EXPECT_EQ(copy.data_type(), source.data_type());
  EXPECT_EQ(copy.shape(), source.shape());
  EXPECT_EQ(copy.device().type, DeviceType::kCpu);
  EXPECT_NE(copy.data(), source.data());
  const auto* copy_values = static_cast<const float*>(copy.data());
  for (std::size_t index = 0; index < copy.element_count(); ++index) {
    EXPECT_FLOAT_EQ(copy_values[index], static_cast<float>(index));
  }

  source_values[0] = 100.0F;
  EXPECT_FLOAT_EQ(copy_values[0], 0.0F);
}

TEST(TensorTest, TensorBufferReusesAndGrowsLikeVectorReserve) {
  TensorBuffer buffer;

  Tensor first = buffer.Ensure(MakeDescWithShape({4, 3, 2, 2}));
  void* first_data = first.data();
  EXPECT_EQ(first.bytes(), 192U);
  EXPECT_EQ(first.capacity_bytes(), 192U);
  EXPECT_EQ(buffer.capacity_bytes(), 192U);

  Tensor smaller = buffer.Ensure(MakeDescWithShape({2, 3, 2, 2}));
  EXPECT_EQ(smaller.data(), first_data);
  EXPECT_EQ(smaller.bytes(), 96U);
  EXPECT_EQ(smaller.capacity_bytes(), 192U);
  EXPECT_EQ(buffer.capacity_bytes(), 192U);

  Tensor larger = buffer.Ensure(MakeDescWithShape({8, 3, 2, 2}));
  EXPECT_NE(larger.data(), nullptr);
  EXPECT_EQ(larger.bytes(), 384U);
  EXPECT_EQ(larger.capacity_bytes(), 384U);
  EXPECT_EQ(buffer.capacity_bytes(), 384U);
}

TEST(TensorTest, TensorBufferKeepsCapacityStableAcrossLargerBatches) {
  TensorBuffer buffer;

  Tensor warmup = buffer.Ensure(MakeDescWithShape({32, 3, 2, 2}));
  void* warmup_data = warmup.data();
  EXPECT_EQ(warmup.bytes(), 1536U);
  EXPECT_EQ(warmup.capacity_bytes(), 1536U);

  for (int64_t batch : {1, 2, 4, 8, 16, 32}) {
    Tensor tensor = buffer.Ensure(MakeDescWithShape({batch, 3, 2, 2}));
    EXPECT_EQ(tensor.data(), warmup_data);
    EXPECT_EQ(tensor.bytes(),
              static_cast<std::size_t>(batch * 3 * 2 * 2 * sizeof(float)));
    EXPECT_EQ(tensor.capacity_bytes(), 1536U);
    EXPECT_EQ(buffer.capacity_bytes(), 1536U);
  }

  Tensor grown = buffer.Ensure(MakeDescWithShape({64, 3, 2, 2}));
  void* grown_data = grown.data();
  EXPECT_NE(grown_data, nullptr);
  EXPECT_EQ(grown.bytes(), 3072U);
  EXPECT_EQ(grown.capacity_bytes(), 3072U);
  EXPECT_EQ(buffer.capacity_bytes(), 3072U);

  for (int64_t batch : {48, 17, 1, 64}) {
    Tensor tensor = buffer.Ensure(MakeDescWithShape({batch, 3, 2, 2}));
    EXPECT_EQ(tensor.data(), grown_data);
    EXPECT_EQ(tensor.bytes(),
              static_cast<std::size_t>(batch * 3 * 2 * 2 * sizeof(float)));
    EXPECT_EQ(tensor.capacity_bytes(), 3072U);
    EXPECT_EQ(buffer.capacity_bytes(), 3072U);
  }
}

TEST(TensorTest, TensorAllocatorRejectsUnsupportedDevice) {
  TensorBuffer buffer(MakeCpuOnlyTestAllocator());
  TensorDesc desc = MakeDesc();
  desc.device = Device{DeviceType::kCuda, 0};

  EXPECT_THROW(static_cast<void>(buffer.Ensure(desc)), std::invalid_argument);
}

TEST(TensorTest, KeepsOwnedBufferAliveWithCustomDeleter) {
  bool deleted = false;
  TensorDesc desc = MakeDesc();
  float* data = new float[12];

  {
    Tensor tensor = Tensor::FromBuffer(desc, data, 12 * sizeof(float),
                                       [&deleted](void* ptr) {
                                         deleted = true;
                                         delete[] static_cast<float*>(ptr);
                                       });

    EXPECT_EQ(tensor.data(), data);
    EXPECT_FALSE(deleted);
  }

  EXPECT_TRUE(deleted);
}

TEST(TensorTest, KeepsExternalOwnerAlive) {
  bool owner_deleted = false;
  std::vector<float> buffer(12, 1.0F);
  std::shared_ptr<void> owner(new int(1), [&owner_deleted](void* ptr) {
    owner_deleted = true;
    delete static_cast<int*>(ptr);
  });

  {
    Tensor tensor = Tensor::FromExternal(MakeDesc(), buffer.data(),
                                         buffer.size() * sizeof(float), owner);

    owner.reset();
    EXPECT_FALSE(owner_deleted);
    EXPECT_EQ(tensor.data(), buffer.data());
  }

  EXPECT_TRUE(owner_deleted);
}

TEST(TensorTest, StoresCudaDeviceMetadataWithoutCudaDependency) {
  std::vector<float> buffer(12, 1.0F);
  TensorDesc desc = MakeDesc();
  desc.device = Device{DeviceType::kCuda, 1};

  Tensor tensor =
      Tensor::FromExternal(desc, buffer.data(), buffer.size() * sizeof(float));

  EXPECT_EQ(tensor.device().type, DeviceType::kCuda);
  EXPECT_EQ(tensor.device().id, 1);
}

TEST(TensorTest, RejectsInvalidBuffers) {
  std::vector<float> buffer(12, 1.0F);
  TensorDesc desc = MakeDesc();

  EXPECT_THROW(static_cast<void>(Tensor::FromExternal(desc, nullptr, 48)),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(Tensor::FromExternal(desc, buffer.data(), 47)),
               std::invalid_argument);

  desc.info.shape = {1, -1, 2, 2};
  EXPECT_THROW(static_cast<void>(Tensor::FromExternal(desc, buffer.data(), 48)),
               std::invalid_argument);

  desc = MakeDesc();
  desc.info.data_type = DataType::kUnknown;
  EXPECT_THROW(static_cast<void>(Tensor::FromExternal(desc, buffer.data(), 48)),
               std::invalid_argument);

  Tensor tensor = Tensor::Allocate(MakeDesc());
  EXPECT_THROW(static_cast<void>(tensor.View(MakeDescWithShape({2, 3, 2, 2}))),
               std::invalid_argument);

  TensorDesc cuda_desc = MakeDesc();
  cuda_desc.device = Device{DeviceType::kCuda, 0};
  EXPECT_THROW(static_cast<void>(tensor.View(cuda_desc)),
               std::invalid_argument);
}

TEST(TensorTest, RejectsEmptyTensorCopy) {
  EXPECT_THROW(static_cast<void>(Tensor{}.CopyTo(Device{DeviceType::kCpu, 0})),
               std::invalid_argument);
}

TEST(TensorTest, RejectsEmptyDeleterForOwnedBuffer) {
  std::vector<float> buffer(12, 1.0F);

  EXPECT_THROW(static_cast<void>(Tensor::FromBuffer(
                   MakeDesc(), buffer.data(), buffer.size() * sizeof(float),
                   std::function<void(void*)>())),
               std::invalid_argument);
}

TEST(TensorTest, RejectsInvalidTensorAllocatorAdapters) {
  EXPECT_THROW(static_cast<void>(TensorAllocator(
                   std::vector<std::unique_ptr<TensorAllocationAdapter>>())),
               std::invalid_argument);

  std::vector<std::unique_ptr<TensorAllocationAdapter>> adapters;
  adapters.push_back(nullptr);
  EXPECT_THROW(static_cast<void>(TensorAllocator(std::move(adapters))),
               std::invalid_argument);
}

}  // namespace
}  // namespace mw::infer
