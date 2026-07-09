#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
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

  Tensor Allocate(TensorDesc) override {
    throw std::logic_error("CPU-only test adapter should not allocate");
  }
};

std::vector<std::unique_ptr<TensorAllocationAdapter>>
MakeCpuOnlyTestAdapters() {
  std::vector<std::unique_ptr<TensorAllocationAdapter>> adapters;
  adapters.push_back(std::make_unique<CpuOnlyTensorAllocationAdapter>());
  return adapters;
}

TEST(TensorTest, ComputesShapeAndTypeSizes) {
  EXPECT_EQ(DataTypeSize(DataType::kUInt8), 1U);
  EXPECT_EQ(DataTypeSize(DataType::kFloat16), 2U);
  EXPECT_EQ(DataTypeSize(DataType::kFloat32), 4U);
  EXPECT_EQ(DataTypeSize(DataType::kInt64), 8U);
  EXPECT_EQ(DataTypeSize(DataType::kFloat64), 8U);
  EXPECT_EQ(ElementCount({}), 1U);
  EXPECT_EQ(ElementCount({2, 0, 3}), 0U);
  EXPECT_EQ(ElementCount({1, 3, 2, 2}), 12U);
  EXPECT_EQ(TensorBytes(MakeDesc()), 48U);
}

TEST(TensorTest, ParsesAndFormatsDevices) {
  Device default_device;
  EXPECT_EQ(default_device.type, DeviceType::kCpu);
  EXPECT_EQ(default_device.id, 0);
  EXPECT_EQ(default_device.ToString(), "cpu");

  Device cpu("cpu");
  EXPECT_EQ(cpu.type, DeviceType::kCpu);
  EXPECT_EQ(cpu.id, 0);
  EXPECT_EQ(cpu.ToString(), "cpu");

  Device cuda("cuda");
  EXPECT_EQ(cuda.type, DeviceType::kCuda);
  EXPECT_EQ(cuda.id, 0);
  EXPECT_EQ(cuda.ToString(), "cuda:0");

  Device cuda_with_id("cuda:2");
  EXPECT_EQ(cuda_with_id.type, DeviceType::kCuda);
  EXPECT_EQ(cuda_with_id.id, 2);
  EXPECT_EQ(cuda_with_id.ToString(), "cuda:2");

  EXPECT_EQ(Device(DeviceType::kCpu, 3).ToString(), "cpu");
  EXPECT_EQ(Device(DeviceType::kCuda, 4).ToString(), "cuda:4");
}

TEST(TensorTest, RejectsInvalidDeviceStrings) {
  EXPECT_THROW(static_cast<void>(Device("gpu")), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(Device("cuda:")), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(Device("cuda:-1")), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(Device("cuda:abc")), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(Device("cpu:0")), std::invalid_argument);
}

TEST(TensorTest, RejectsShapeAndByteSizeOverflow) {
  EXPECT_THROW(
      static_cast<void>(ElementCount({std::numeric_limits<int64_t>::max(), 3})),
      std::invalid_argument);

  TensorDesc desc = MakeDesc();
  desc.info.shape = {
      static_cast<int64_t>(std::numeric_limits<std::size_t>::max() /
                           DataTypeSize(desc.info.data_type)) +
      1};
  EXPECT_THROW(static_cast<void>(TensorBytes(desc)), std::invalid_argument);
}

TEST(TensorTest, SupportsScalarAndZeroSizedShapes) {
  TensorDesc scalar_desc = MakeDescWithShape({});
  Tensor scalar = Tensor::Allocate(scalar_desc);
  EXPECT_FALSE(scalar.empty());
  EXPECT_EQ(scalar.element_count(), 1U);
  EXPECT_EQ(scalar.bytes(), sizeof(float));

  TensorDesc zero_desc = MakeDescWithShape({0, 3});
  Tensor zero = Tensor::Allocate(zero_desc);
  EXPECT_FALSE(zero.empty());
  EXPECT_EQ(zero.element_count(), 0U);
  EXPECT_EQ(zero.bytes(), 0U);

  Tensor external_zero = Tensor::FromExternal(zero_desc, nullptr, 0);
  EXPECT_FALSE(external_zero.empty());
  EXPECT_EQ(external_zero.element_count(), 0U);
  EXPECT_EQ(external_zero.bytes(), 0U);
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

TEST(TensorTest, ProvidesTypedDataAccess) {
  Tensor tensor = Tensor::Allocate(MakeDesc());

  float* values = tensor.data<float>();
  values[0] = 1.25F;

  const Tensor& const_tensor = tensor;
  EXPECT_EQ(const_tensor.data<float>(), values);
  EXPECT_FLOAT_EQ(const_tensor.data<float>()[0], 1.25F);
  EXPECT_THROW(static_cast<void>(tensor.data<int64_t>()),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(Tensor{}.data<float>()),
               std::invalid_argument);
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

TEST(TensorTest, CopiesHostValuesToVector) {
  Tensor tensor = Tensor::Allocate(MakeDesc());
  float* values = tensor.data<float>();
  for (std::size_t index = 0; index < tensor.element_count(); ++index) {
    values[index] = static_cast<float>(index);
  }

  EXPECT_EQ(tensor.CopyToHostVector<float>(),
            std::vector<float>({0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F,
                                8.0F, 9.0F, 10.0F, 11.0F}));
  EXPECT_THROW(static_cast<void>(tensor.CopyToHostVector<int64_t>()),
               std::invalid_argument);

  Tensor zero = Tensor::Allocate(MakeDescWithShape({0, 3}));
  EXPECT_TRUE(zero.CopyToHostVector<float>().empty());
}

TEST(TensorTest, ReadsElementAtMultiDimensionalIndex) {
  Tensor tensor = Tensor::Allocate(MakeDesc());
  float* values = tensor.data<float>();
  for (std::size_t index = 0; index < tensor.element_count(); ++index) {
    values[index] = static_cast<float>(index);
  }

  EXPECT_FLOAT_EQ(tensor.At<float>({0, 0, 0, 0}), 0.0F);
  EXPECT_FLOAT_EQ(tensor.At<float>({0, 1, 1, 0}), 6.0F);
  EXPECT_FLOAT_EQ(tensor.At<float>({0, 2, 1, 1}), 11.0F);

  Tensor scalar = Tensor::Allocate(MakeDescWithShape({}));
  scalar.data<float>()[0] = 42.0F;
  EXPECT_FLOAT_EQ(scalar.At<float>({}), 42.0F);
}

TEST(TensorTest, RejectsInvalidElementIndex) {
  Tensor tensor = Tensor::Allocate(MakeDesc());

  EXPECT_THROW(static_cast<void>(tensor.At<float>({0, 0, 0})),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(tensor.At<float>({0, 3, 0, 0})),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(tensor.At<float>({0, -1, 0, 0})),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(tensor.At<int64_t>({0, 0, 0, 0})),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(Tensor{}.At<float>({})),
               std::invalid_argument);

  Tensor zero = Tensor::Allocate(MakeDescWithShape({0, 3}));
  EXPECT_THROW(static_cast<void>(zero.At<float>({0, 0})),
               std::invalid_argument);
}

TEST(TensorTest, PooledTensorAllocatorReusesReleasedBlocks) {
  PooledTensorAllocator allocator;

  void* first_data = nullptr;
  {
    Tensor first = Tensor::Allocate(MakeDescWithShape({4, 3, 2, 2}), allocator);
    first_data = first.data();
    EXPECT_EQ(first.bytes(), 192U);
    EXPECT_EQ(first.capacity_bytes(), 192U);
  }

  Tensor smaller = Tensor::Allocate(MakeDescWithShape({2, 3, 2, 2}), allocator);
  EXPECT_EQ(smaller.data(), first_data);
  EXPECT_EQ(smaller.bytes(), 96U);
  EXPECT_EQ(smaller.capacity_bytes(), 192U);

  Tensor active = Tensor::Allocate(MakeDescWithShape({2, 3, 2, 2}), allocator);
  EXPECT_NE(active.data(), smaller.data());
}

TEST(TensorTest, PooledTensorAllocatorGrowsWhenReleasedBlockIsTooSmall) {
  PooledTensorAllocator allocator;

  void* small_data = nullptr;
  {
    Tensor small = Tensor::Allocate(MakeDescWithShape({4, 3, 2, 2}), allocator);
    small_data = small.data();
    EXPECT_EQ(small.capacity_bytes(), 192U);
  }

  Tensor larger = Tensor::Allocate(MakeDescWithShape({8, 3, 2, 2}), allocator);
  EXPECT_NE(larger.data(), nullptr);
  EXPECT_NE(larger.data(), small_data);
  EXPECT_EQ(larger.bytes(), 384U);
  EXPECT_EQ(larger.capacity_bytes(), 384U);
}

TEST(TensorTest, DirectTensorAllocatorRejectsUnsupportedDevice) {
  DirectTensorAllocator allocator(MakeCpuOnlyTestAdapters());
  TensorDesc desc = MakeDesc();
  desc.device = Device{DeviceType::kCuda, 0};

  EXPECT_THROW(static_cast<void>(allocator.Allocate(desc)),
               std::invalid_argument);
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
  EXPECT_THROW(static_cast<void>(DirectTensorAllocator(
                   std::vector<std::unique_ptr<TensorAllocationAdapter>>())),
               std::invalid_argument);

  std::vector<std::unique_ptr<TensorAllocationAdapter>> adapters;
  adapters.push_back(nullptr);
  EXPECT_THROW(static_cast<void>(DirectTensorAllocator(std::move(adapters))),
               std::invalid_argument);
}

}  // namespace
}  // namespace mw::infer
