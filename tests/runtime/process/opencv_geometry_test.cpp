#include <gtest/gtest.h>

#include <opencv2/core.hpp>
#include <utility>
#include <vector>

#include "mw/infer/runtime/input/opencv_input.h"
#include "mw/infer/runtime/process/geometry.h"

namespace mw::infer {
namespace {

TEST(OpenCvGeometryTest, DefaultConstructedTransformerHasOpenCvAdapter) {
  RawImage image(cv::Mat(10, 20, CV_8UC3));
  GeometryTransformer transformer;

  EXPECT_TRUE(transformer.Supports(image));
}

TEST(OpenCvGeometryTest, ResizesRawImageBatchAndRestoresPoint) {
  GeometryTransformer transformer;
  RawImageBatch raw_images(std::vector<cv::Mat>{cv::Mat(10, 20, CV_8UC3)});

  GeometryResult original(raw_images);
  EXPECT_EQ(original.original_size(0).width, 20);
  EXPECT_EQ(original.original_size(0).height, 10);
  EXPECT_EQ(original.transformed_size(0).width, 20);
  EXPECT_EQ(original.transformed_size(0).height, 10);

  GeometryResult resized = transformer.Resize(raw_images, ImageSize{40, 20},
                                              Interpolation::kNearest);

  ASSERT_EQ(resized.size(), 1U);
  const cv::Mat& output = GetOpenCvMat(resized.images().image(0));
  EXPECT_EQ(output.cols, 40);
  EXPECT_EQ(output.rows, 20);
  EXPECT_EQ(output.type(), CV_8UC3);
  EXPECT_EQ(resized.original_size(0).width, 20);
  EXPECT_EQ(resized.original_size(0).height, 10);
  EXPECT_EQ(resized.transformed_size(0).width, 40);
  EXPECT_EQ(resized.transformed_size(0).height, 20);

  const GeometryTrace& trace = resized.trace(0);
  ASSERT_EQ(trace.size(), 1U);
  EXPECT_EQ(trace.step(0).kind, GeometryStepKind::kResize);
  EXPECT_FLOAT_EQ(trace.step(0).resize.scale_x, 2.0F);
  EXPECT_FLOAT_EQ(trace.step(0).resize.scale_y, 2.0F);

  const Point2f restored = trace.RestorePoint(Point2f{20.0F, 10.0F});
  EXPECT_FLOAT_EQ(restored.x, 10.0F);
  EXPECT_FLOAT_EQ(restored.y, 5.0F);
}

TEST(OpenCvGeometryTest, ComputesResizeShortSideSize) {
  ImageSize wide = ResizeShortSideSize(ImageSize{20, 10}, 5);
  EXPECT_EQ(wide.width, 10);
  EXPECT_EQ(wide.height, 5);

  ImageSize tall = ResizeShortSideSize(ImageSize{10, 20}, 5);
  EXPECT_EQ(tall.width, 5);
  EXPECT_EQ(tall.height, 10);

  ImageSize square = ResizeShortSideSize(ImageSize{10, 10}, 7);
  EXPECT_EQ(square.width, 7);
  EXPECT_EQ(square.height, 7);
}

TEST(OpenCvGeometryTest, RejectsInvalidResizeShortSideSize) {
  EXPECT_THROW(static_cast<void>(ResizeShortSideSize(ImageSize{0, 10}, 5)),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(ResizeShortSideSize(ImageSize{10, 10}, 0)),
               std::invalid_argument);
}

TEST(OpenCvGeometryTest, ResizesShortSideAndRecordsResizeStep) {
  GeometryTransformer transformer;
  RawImageBatch raw_images(std::vector<cv::Mat>{cv::Mat(10, 20, CV_8UC3)});

  GeometryResult resized =
      transformer.ResizeShortSide(raw_images, 5, Interpolation::kNearest);

  const cv::Mat& output = GetOpenCvMat(resized.images().image(0));
  EXPECT_EQ(output.cols, 10);
  EXPECT_EQ(output.rows, 5);
  EXPECT_EQ(resized.original_size(0).width, 20);
  EXPECT_EQ(resized.original_size(0).height, 10);
  EXPECT_EQ(resized.transformed_size(0).width, 10);
  EXPECT_EQ(resized.transformed_size(0).height, 5);

  const GeometryTrace& trace = resized.trace(0);
  ASSERT_EQ(trace.size(), 1U);
  const GeometryStep& step = trace.step(0);
  EXPECT_EQ(step.kind, GeometryStepKind::kResize);
  EXPECT_EQ(step.before_size.width, 20);
  EXPECT_EQ(step.before_size.height, 10);
  EXPECT_EQ(step.after_size.width, 10);
  EXPECT_EQ(step.after_size.height, 5);
}

TEST(OpenCvGeometryTest, LetterBoxesImageAndRecordsSingleStep) {
  GeometryTransformer transformer;
  RawImageBatch raw_images(
      std::vector<cv::Mat>{cv::Mat(10, 20, CV_8UC3, cv::Scalar(1, 2, 3))});

  GeometryResult boxed =
      transformer.LetterBox(raw_images, ImageSize{40, 40},
                            Interpolation::kNearest, FillValue{{7, 7, 7, 7}});

  const cv::Mat& output = GetOpenCvMat(boxed.images().image(0));
  EXPECT_EQ(output.cols, 40);
  EXPECT_EQ(output.rows, 40);
  EXPECT_EQ(output.at<cv::Vec3b>(0, 0), cv::Vec3b(7, 7, 7));

  const GeometryTrace& trace = boxed.trace(0);
  ASSERT_EQ(trace.size(), 1U);
  const GeometryStep& step = trace.step(0);
  EXPECT_EQ(step.kind, GeometryStepKind::kLetterBox);
  EXPECT_EQ(step.letterbox.resized_size.width, 40);
  EXPECT_EQ(step.letterbox.resized_size.height, 20);
  EXPECT_EQ(step.letterbox.padding.top, 10);
  EXPECT_EQ(step.letterbox.padding.bottom, 10);

  const Point2f restored = trace.RestorePoint(Point2f{20.0F, 20.0F});
  EXPECT_FLOAT_EQ(restored.x, 10.0F);
  EXPECT_FLOAT_EQ(restored.y, 5.0F);
}

TEST(OpenCvGeometryTest, CropsThenPadsGeometryResult) {
  GeometryTransformer transformer;
  GeometryResult batch(RawImageBatch(
      std::vector<cv::Mat>{cv::Mat(10, 20, CV_8UC3, cv::Scalar(1, 2, 3))}));

  batch = transformer.Crop(std::move(batch), Rect{2, 3, 8, 4});
  batch = transformer.Pad(std::move(batch), Padding{1, 2, 3, 4},
                          FillValue{{9, 9, 9, 9}});

  const cv::Mat& output = GetOpenCvMat(batch.images().image(0));
  EXPECT_EQ(output.cols, 12);
  EXPECT_EQ(output.rows, 10);
  EXPECT_EQ(output.at<cv::Vec3b>(0, 0), cv::Vec3b(9, 9, 9));
  EXPECT_EQ(batch.original_size(0).width, 20);
  EXPECT_EQ(batch.original_size(0).height, 10);
  EXPECT_EQ(batch.transformed_size(0).width, 12);
  EXPECT_EQ(batch.transformed_size(0).height, 10);

  const GeometryTrace& trace = batch.trace(0);
  ASSERT_EQ(trace.size(), 2U);
  EXPECT_EQ(trace.step(0).kind, GeometryStepKind::kCrop);
  EXPECT_EQ(trace.step(1).kind, GeometryStepKind::kPad);

  const Point2f restored = trace.RestorePoint(Point2f{1.0F, 2.0F});
  EXPECT_FLOAT_EQ(restored.x, 2.0F);
  EXPECT_FLOAT_EQ(restored.y, 3.0F);
}

TEST(OpenCvGeometryTest, CenterCropsEachFrame) {
  GeometryTransformer transformer;
  RawImageBatch raw_images(std::vector<cv::Mat>{cv::Mat(10, 20, CV_8UC3)});

  GeometryResult cropped = transformer.CenterCrop(raw_images, ImageSize{8, 6});

  const cv::Mat& output = GetOpenCvMat(cropped.images().image(0));
  EXPECT_EQ(output.cols, 8);
  EXPECT_EQ(output.rows, 6);

  const GeometryStep& step = cropped.trace(0).step(0);
  EXPECT_EQ(step.kind, GeometryStepKind::kCrop);
  EXPECT_EQ(step.crop.crop_rect.x, 6);
  EXPECT_EQ(step.crop.crop_rect.y, 2);
}

TEST(OpenCvGeometryTest, ResizesTwoChannelMatWithoutColorSemantics) {
  GeometryTransformer transformer;
  RawImageBatch raw_images(std::vector<cv::Mat>{cv::Mat(10, 20, CV_32FC2)});

  GeometryResult resized = transformer.Resize(raw_images, ImageSize{5, 5});

  const RawImage& image = resized.images().image(0);
  EXPECT_EQ(image.pixel_format(), PixelFormat::kUnknown);
  EXPECT_EQ(image.data_type(), DataType::kFloat32);
  EXPECT_EQ(image.channels(), 2);

  const cv::Mat& output = GetOpenCvMat(image);
  EXPECT_EQ(output.cols, 5);
  EXPECT_EQ(output.rows, 5);
  EXPECT_EQ(output.type(), CV_32FC2);
}

TEST(OpenCvGeometryTest, ResizesFourSixAndNineChannelMats) {
  GeometryTransformer transformer;
  {
    RawImageBatch raw_images(std::vector<cv::Mat>{cv::Mat(10, 20, CV_8UC4)});

    GeometryResult resized = transformer.Resize(raw_images, ImageSize{5, 5});

    const RawImage& image = resized.images().image(0);
    EXPECT_EQ(image.pixel_format(), PixelFormat::kBgra);
    EXPECT_EQ(image.data_type(), DataType::kUInt8);
    EXPECT_EQ(image.channels(), 4);
    EXPECT_EQ(GetOpenCvMat(image).type(), CV_8UC4);
  }
  {
    RawImageBatch raw_images(
        std::vector<cv::Mat>{cv::Mat(10, 20, CV_MAKETYPE(CV_8U, 6))});

    GeometryResult resized = transformer.Resize(raw_images, ImageSize{5, 5});

    const RawImage& image = resized.images().image(0);
    EXPECT_EQ(image.pixel_format(), PixelFormat::kUnknown);
    EXPECT_EQ(image.data_type(), DataType::kUInt8);
    EXPECT_EQ(image.channels(), 6);
    EXPECT_EQ(GetOpenCvMat(image).type(), CV_MAKETYPE(CV_8U, 6));
  }
  {
    RawImageBatch raw_images(
        std::vector<cv::Mat>{cv::Mat(10, 20, CV_MAKETYPE(CV_32F, 9))});

    GeometryResult resized = transformer.Resize(raw_images, ImageSize{5, 5});

    const RawImage& image = resized.images().image(0);
    EXPECT_EQ(image.pixel_format(), PixelFormat::kUnknown);
    EXPECT_EQ(image.data_type(), DataType::kFloat32);
    EXPECT_EQ(image.channels(), 9);
    EXPECT_EQ(GetOpenCvMat(image).type(), CV_MAKETYPE(CV_32F, 9));
  }
}

TEST(OpenCvGeometryTest, RejectsInvalidCrop) {
  GeometryTransformer transformer;
  RawImageBatch raw_images(std::vector<cv::Mat>{cv::Mat(10, 20, CV_8UC3)});

  EXPECT_THROW(
      static_cast<void>(transformer.Crop(raw_images, Rect{18, 0, 4, 4})),
      std::invalid_argument);
}

}  // namespace
}  // namespace mw::infer
