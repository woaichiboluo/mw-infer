#include "mw/infer/runtime/process/geometry.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace mw::infer {
namespace {

struct TestImageHandle {
  std::string name;
};

RawImage MakeTestImage(std::string name, ImageSize size) {
  ImageDesc desc;
  desc.size = size;
  desc.pixel_format = PixelFormat::kBgr;
  desc.data_type = DataType::kUInt8;
  desc.channels = 3;
  desc.memory_kind = ImageMemoryKind::kHost;
  desc.planes = {
      ImagePlaneDesc{static_cast<std::size_t>(size.width * desc.channels),
                     static_cast<std::size_t>(desc.channels)}};
  return RawImage::FromHandle(std::move(desc), ImageHandleKind::kNone,
                              TestImageHandle{std::move(name)});
}

TEST(GeometryResultTest, WrapsRawImageBatch) {
  RawImageBatch raw_images({MakeTestImage("a", ImageSize{20, 10}),
                            MakeTestImage("b", ImageSize{30, 15})});

  GeometryResult result(raw_images);

  ASSERT_EQ(result.size(), 2U);
  EXPECT_EQ(result.images().image(0).size().width, 20);
  EXPECT_TRUE(result.trace(0).empty());
  EXPECT_EQ(result.images().size(), 2U);
}

TEST(GeometryTraceTest, RestoresPointThroughResizeAndCrop) {
  GeometryTrace trace;
  trace.AddCrop(ImageSize{100, 50}, Rect{10, 5, 40, 20});
  trace.AddResize(ImageSize{40, 20}, ImageSize{80, 40});

  const Point2f restored = trace.RestorePoint(Point2f{20.0F, 10.0F});

  EXPECT_FLOAT_EQ(restored.x, 20.0F);
  EXPECT_FLOAT_EQ(restored.y, 10.0F);
}

TEST(GeometryTraceTest, RestoresRectThroughLetterBox) {
  GeometryTrace trace;
  trace.AddLetterBox(ImageSize{20, 10}, ImageSize{40, 40}, ImageSize{40, 20},
                     Padding{0, 10, 0, 10});

  const Rect2f restored = trace.RestoreRect(Rect2f{10.0F, 15.0F, 20.0F, 10.0F});

  EXPECT_FLOAT_EQ(restored.x, 5.0F);
  EXPECT_FLOAT_EQ(restored.y, 2.5F);
  EXPECT_FLOAT_EQ(restored.width, 10.0F);
  EXPECT_FLOAT_EQ(restored.height, 5.0F);
}

TEST(GeometryTraceTest, RestoresPolygonThroughPad) {
  GeometryTrace trace;
  trace.AddPad(ImageSize{20, 10}, Padding{2, 3, 4, 5});

  const std::vector<Point2f> restored =
      trace.RestorePolygon({Point2f{2.0F, 3.0F}, Point2f{12.0F, 8.0F}});

  ASSERT_EQ(restored.size(), 2U);
  EXPECT_FLOAT_EQ(restored[0].x, 0.0F);
  EXPECT_FLOAT_EQ(restored[0].y, 0.0F);
  EXPECT_FLOAT_EQ(restored[1].x, 10.0F);
  EXPECT_FLOAT_EQ(restored[1].y, 5.0F);
}

TEST(GeometryTraceTest, DumpsOneStepPerLine) {
  GeometryTrace empty_trace;
  EXPECT_TRUE(empty_trace.Dump().empty());

  GeometryTrace trace;
  trace.AddResize(ImageSize{20, 10}, ImageSize{40, 20});
  trace.AddPad(ImageSize{40, 20}, Padding{1, 2, 3, 4});
  trace.AddCrop(ImageSize{44, 26}, Rect{5, 6, 10, 8});
  trace.AddLetterBox(ImageSize{10, 8}, ImageSize{20, 20}, ImageSize{20, 16},
                     Padding{0, 2, 0, 2});

  const std::string dump = trace.Dump();

  EXPECT_EQ(dump,
            "#0 Resize before=20x10 after=40x20 scale_x=2 scale_y=2\n"
            "#1 Pad before=40x20 after=44x26 "
            "padding={left=1,top=2,right=3,bottom=4}\n"
            "#2 Crop before=44x26 after=10x8 "
            "rect={x=5,y=6,width=10,height=8}\n"
            "#3 LetterBox before=10x8 after=20x20 resized=20x16 scale_x=2 "
            "scale_y=2 padding={left=0,top=2,right=0,bottom=2}");
}

TEST(GeometryTraceTest, RejectsInvalidGeometry) {
  GeometryTrace trace;

  EXPECT_THROW(trace.AddResize(ImageSize{0, 10}, ImageSize{20, 20}),
               std::invalid_argument);
  EXPECT_THROW(trace.AddPad(ImageSize{20, 10}, Padding{-1, 0, 0, 0}),
               std::invalid_argument);
  EXPECT_THROW(trace.AddCrop(ImageSize{20, 10}, Rect{15, 0, 10, 10}),
               std::invalid_argument);
}

}  // namespace
}  // namespace mw::infer
