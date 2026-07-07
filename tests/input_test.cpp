#include "mw/infer/runtime/input.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <type_traits>

namespace mw::infer {
namespace {

struct TestImageHandle {
  std::string name;
};

static_assert(std::is_same_v<decltype(ImageDesc::size), ImageSize>);

TEST(RawImageTest, StoresDescriptionAndTypedHandle) {
  ImageDesc desc;
  desc.size = ImageSize{20, 10};
  desc.pixel_format = PixelFormat::kRgb;
  desc.data_type = DataType::kUInt8;
  desc.channels = 3;
  desc.memory_kind = ImageMemoryKind::kHost;
  desc.planes = {ImagePlaneDesc{60, 3}};

  RawImage image = RawImage::FromHandle(desc, ImageHandleKind::kNone,
                                        TestImageHandle{std::string("frame")});

  ASSERT_FALSE(image.empty());
  EXPECT_EQ(image.size().width, 20);
  EXPECT_EQ(image.size().height, 10);
  EXPECT_EQ(image.pixel_format(), PixelFormat::kRgb);
  EXPECT_EQ(image.data_type(), DataType::kUInt8);
  EXPECT_EQ(image.channels(), 3);
  EXPECT_EQ(image.memory_kind(), ImageMemoryKind::kHost);
  ASSERT_EQ(image.desc().planes.size(), 1U);
  EXPECT_EQ(image.desc().planes[0].row_stride_bytes, 60U);
  EXPECT_EQ(image.desc().planes[0].pixel_stride_bytes, 3U);
  EXPECT_EQ(image.handle_kind(), ImageHandleKind::kNone);

  const auto* handle = static_cast<const TestImageHandle*>(image.handle());
  ASSERT_NE(handle, nullptr);
  EXPECT_EQ(handle->name, "frame");
}

TEST(RawImageBatchTest, AcceptsUniformMemoryKind) {
  ImageDesc desc;
  desc.size = ImageSize{20, 10};
  desc.pixel_format = PixelFormat::kRgb;
  desc.data_type = DataType::kUInt8;
  desc.channels = 3;
  desc.memory_kind = ImageMemoryKind::kHost;

  RawImageBatch batch(
      {RawImage::FromHandle(desc, ImageHandleKind::kNone, TestImageHandle{"a"}),
       RawImage::FromHandle(desc, ImageHandleKind::kNone,
                            TestImageHandle{"b"})});

  ASSERT_EQ(batch.size(), 2U);
  EXPECT_EQ(batch.memory_kind(), ImageMemoryKind::kHost);
  const auto* handle =
      static_cast<const TestImageHandle*>(batch.image(1).handle());
  EXPECT_EQ(handle->name, "b");
}

TEST(RawImageBatchTest, WrapsRawImageBatchWithTemplateEntry) {
  ImageDesc desc;
  desc.size = ImageSize{20, 10};
  desc.pixel_format = PixelFormat::kRgb;
  desc.data_type = DataType::kUInt8;
  desc.channels = 3;
  desc.memory_kind = ImageMemoryKind::kHost;

  RawImageBatch batch = ToRawImageBatch(std::vector<RawImage>{
      RawImage::FromHandle(desc, ImageHandleKind::kNone, TestImageHandle{"a"}),
      RawImage::FromHandle(desc, ImageHandleKind::kNone,
                           TestImageHandle{"b"})});

  ASSERT_EQ(batch.size(), 2U);
  EXPECT_EQ(batch.memory_kind(), ImageMemoryKind::kHost);
}

TEST(RawImageBatchTest, RejectsEmptyImageOrMixedMemoryKind) {
  ImageDesc host_desc;
  host_desc.memory_kind = ImageMemoryKind::kHost;
  ImageDesc cuda_desc;
  cuda_desc.memory_kind = ImageMemoryKind::kCuda;

  EXPECT_THROW(static_cast<void>(RawImageBatch({RawImage()})),
               std::invalid_argument);

  EXPECT_THROW(static_cast<void>(RawImageBatch(
                   {RawImage::FromHandle(host_desc, ImageHandleKind::kNone,
                                         TestImageHandle{"host"}),
                    RawImage::FromHandle(cuda_desc, ImageHandleKind::kNone,
                                         TestImageHandle{"cuda"})})),
               std::invalid_argument);
}

TEST(RawImageBatchTest, RejectsMixedPixelFormats) {
  ImageDesc rgb_desc;
  rgb_desc.pixel_format = PixelFormat::kRgb;
  ImageDesc bgr_desc;
  bgr_desc.pixel_format = PixelFormat::kBgr;

  EXPECT_THROW(static_cast<void>(RawImageBatch(
                   {RawImage::FromHandle(rgb_desc, ImageHandleKind::kNone,
                                         TestImageHandle{"rgb"}),
                    RawImage::FromHandle(bgr_desc, ImageHandleKind::kNone,
                                         TestImageHandle{"bgr"})})),
               std::invalid_argument);
}

TEST(RawImageBatchTest, RejectsMixedDataTypes) {
  ImageDesc uint8_desc;
  uint8_desc.data_type = DataType::kUInt8;
  ImageDesc float32_desc;
  float32_desc.data_type = DataType::kFloat32;

  EXPECT_THROW(static_cast<void>(RawImageBatch(
                   {RawImage::FromHandle(uint8_desc, ImageHandleKind::kNone,
                                         TestImageHandle{"uint8"}),
                    RawImage::FromHandle(float32_desc, ImageHandleKind::kNone,
                                         TestImageHandle{"float32"})})),
               std::invalid_argument);
}

TEST(RawImageBatchTest, RejectsMixedChannelCounts) {
  ImageDesc one_channel_desc;
  one_channel_desc.channels = 1;
  ImageDesc two_channel_desc;
  two_channel_desc.channels = 2;

  EXPECT_THROW(
      static_cast<void>(RawImageBatch(
          {RawImage::FromHandle(one_channel_desc, ImageHandleKind::kNone,
                                TestImageHandle{"one"}),
           RawImage::FromHandle(two_channel_desc, ImageHandleKind::kNone,
                                TestImageHandle{"two"})})),
      std::invalid_argument);
}

}  // namespace
}  // namespace mw::infer
