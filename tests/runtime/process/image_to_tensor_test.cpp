#include "mw/infer/runtime/process/image_to_tensor.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mw::infer {
namespace {

ImageDesc MakeImageDesc(ImageMemoryKind memory_kind) {
  ImageDesc desc;
  desc.size = ImageSize{20, 10};
  desc.pixel_format = PixelFormat::kBgr;
  desc.data_type = DataType::kUInt8;
  desc.channels = 3;
  desc.memory_kind = memory_kind;
  desc.planes = {ImagePlaneDesc{60, 3}};
  return desc;
}

RawImage MakeRawImage(ImageMemoryKind memory_kind) {
  return RawImage::FromHandle(MakeImageDesc(memory_kind),
                              ImageHandleKind::kNone, 1);
}

RawImage MakeRawImage(ImageSize size) {
  ImageDesc desc = MakeImageDesc(ImageMemoryKind::kHost);
  desc.size = size;
  return RawImage::FromHandle(std::move(desc), ImageHandleKind::kNone, 1);
}

TensorInfo MakeInput(std::vector<int64_t> shape = {-1, 3, -1, -1},
                     DataType data_type = DataType::kFloat32) {
  TensorInfo input;
  input.name = "input";
  input.data_type = data_type;
  input.shape = std::move(shape);
  return input;
}

class HostCpuImageToTensorAdapter final : public ImageToTensorAdapter {
 public:
  bool Supports(const RawImageBatch& images, Device target_device,
                const TensorInfo& input, TensorLayout layout) const override {
    return images.memory_kind() == ImageMemoryKind::kHost &&
           target_device.type == DeviceType::kCpu &&
           (layout == TensorLayout::kBchw || layout == TensorLayout::kBhwc) &&
           (input.data_type == DataType::kFloat32 ||
            input.data_type == DataType::kUInt8);
  }

  void Convert(const RawImageBatch&, Tensor* output,
               TensorLayout) const override {
    if (output == nullptr || output->empty()) {
      throw std::invalid_argument("Output tensor is empty");
    }
    if (output->data_type() == DataType::kUInt8) {
      static_cast<uint8_t*>(output->data())[0] = 255;
      return;
    }
    static_cast<float*>(output->data())[0] = 1.25F;
  }
};

ImageToTensorConverter MakeConverter() {
  std::vector<std::unique_ptr<ImageToTensorAdapter>> adapters;
  adapters.push_back(std::make_unique<HostCpuImageToTensorAdapter>());
  return ImageToTensorConverter(std::move(adapters));
}

TEST(ImageToTensorConverterTest, ResolvesDynamicBchwTensorFromInputInfo) {
  ImageToTensorConverter converter = MakeConverter();
  RawImageBatch images({MakeRawImage(ImageMemoryKind::kHost)});
  const TensorInfo input = MakeInput();

  ASSERT_TRUE(converter.Supports(images, Device{DeviceType::kCpu, 0}, input));

  Tensor tensor = converter.Convert(images, Device{DeviceType::kCpu, 0}, input);

  EXPECT_EQ(tensor.name(), "input");
  EXPECT_EQ(tensor.data_type(), DataType::kFloat32);
  EXPECT_EQ(tensor.shape(), std::vector<int64_t>({1, 3, 10, 20}));
  EXPECT_EQ(tensor.device().type, DeviceType::kCpu);
  EXPECT_FLOAT_EQ(static_cast<float*>(tensor.data())[0], 1.25F);
}

TEST(ImageToTensorConverterTest, ResolvesDynamicBhwcTensorFromInputInfo) {
  ImageToTensorConverter converter = MakeConverter();
  RawImageBatch images({MakeRawImage(ImageMemoryKind::kHost)});

  Tensor tensor =
      converter.Convert(images, Device{DeviceType::kCpu, 0},
                        MakeInput({-1, -1, -1, 3}), TensorLayout::kBhwc);

  EXPECT_EQ(tensor.shape(), std::vector<int64_t>({1, 10, 20, 3}));
}

TEST(ImageToTensorConverterTest, UsesModelInputDataType) {
  ImageToTensorConverter converter = MakeConverter();
  RawImageBatch images({MakeRawImage(ImageMemoryKind::kHost)});
  ASSERT_EQ(images.image(0).data_type(), DataType::kUInt8);

  Tensor tensor =
      converter.Convert(images, Device{DeviceType::kCpu, 0},
                        MakeInput({-1, 3, -1, -1}, DataType::kUInt8));

  EXPECT_EQ(tensor.data_type(), DataType::kUInt8);
  EXPECT_EQ(static_cast<uint8_t*>(tensor.data())[0], 255);
}

TEST(ImageToTensorConverterTest, ReusesTensorBufferWhenProvided) {
  ImageToTensorConverter converter = MakeConverter();
  RawImageBatch images({MakeRawImage(ImageMemoryKind::kHost)});
  TensorBuffer buffer;
  const TensorInfo input = MakeInput();

  Tensor first =
      converter.Convert(images, Device{DeviceType::kCpu, 0}, input, buffer);
  void* first_data = first.data();

  Tensor second =
      converter.Convert(images, Device{DeviceType::kCpu, 0}, input, buffer);

  EXPECT_EQ(second.data(), first_data);
  EXPECT_EQ(second.capacity_bytes(), first.capacity_bytes());
  EXPECT_EQ(buffer.capacity_bytes(), first.capacity_bytes());
  EXPECT_FLOAT_EQ(static_cast<float*>(second.data())[0], 1.25F);
}

TEST(ImageToTensorConverterTest, RejectsInvalidConversionRequests) {
  ImageToTensorConverter converter = MakeConverter();
  RawImageBatch empty_images;
  RawImageBatch images({MakeRawImage(ImageMemoryKind::kHost)});
  const TensorInfo input = MakeInput();

  EXPECT_FALSE(
      converter.Supports(empty_images, Device{DeviceType::kCpu, 0}, input));
  EXPECT_THROW(static_cast<void>(converter.Convert(
                   empty_images, Device{DeviceType::kCpu, 0}, input)),
               std::invalid_argument);

  EXPECT_FALSE(
      converter.Supports(images, Device{DeviceType::kCuda, -1}, input));
  EXPECT_THROW(static_cast<void>(converter.Convert(
                   images, Device{DeviceType::kCuda, -1}, input)),
               std::invalid_argument);

  EXPECT_FALSE(converter.Supports(images, Device{DeviceType::kCpu, 0}, input,
                                  static_cast<TensorLayout>(999)));
  EXPECT_THROW(static_cast<void>(
                   converter.Convert(images, Device{DeviceType::kCpu, 0}, input,
                                     static_cast<TensorLayout>(999))),
               std::invalid_argument);

  EXPECT_FALSE(converter.Supports(images, Device{DeviceType::kCpu, 0},
                                  MakeInput({1, 4, 10, 20})));
  EXPECT_THROW(
      static_cast<void>(converter.Convert(images, Device{DeviceType::kCpu, 0},
                                          MakeInput({1, 4, 10, 20}))),
      std::invalid_argument);

  RawImageBatch mixed_sizes(
      {MakeRawImage(ImageSize{20, 10}), MakeRawImage(ImageSize{10, 10})});
  EXPECT_FALSE(
      converter.Supports(mixed_sizes, Device{DeviceType::kCpu, 0}, input));
  EXPECT_THROW(static_cast<void>(converter.Convert(
                   mixed_sizes, Device{DeviceType::kCpu, 0}, input)),
               std::invalid_argument);
}

}  // namespace
}  // namespace mw::infer
