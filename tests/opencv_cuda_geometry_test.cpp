#include <gtest/gtest.h>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <utility>
#include <vector>

#include "mw/infer/runtime/adapters/opencv_geometry.h"

namespace mw::infer {
namespace {

bool HasUsableCudaDevice() {
  try {
    return cv::cuda::getCudaEnabledDeviceCount() > 0;
  } catch (const cv::Exception&) {
    return false;
  }
}

TEST(OpenCvCudaGeometryTest, ResizesGpuMatBatchAndRestoresPoint) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "OpenCV CUDA device is unavailable";
  }

  RawImageBatch raw_images(
      std::vector<cv::cuda::GpuMat>{cv::cuda::GpuMat(10, 20, CV_8UC3)});

  ImageBatch resized = Resize(raw_images, ImageSize{40, 20}, cv::INTER_NEAREST);

  ASSERT_EQ(resized.size(), 1U);
  const cv::cuda::GpuMat& output = GetOpenCvCudaGpuMat(resized.frame(0).image);
  EXPECT_EQ(output.cols, 40);
  EXPECT_EQ(output.rows, 20);
  EXPECT_EQ(output.type(), CV_8UC3);

  const GeometryTrace& trace = resized.frame(0).geometry_trace;
  ASSERT_EQ(trace.size(), 1U);
  EXPECT_EQ(trace.step(0).kind, GeometryStepKind::kResize);

  const Point2f restored = trace.RestorePoint(Point2f{20.0F, 10.0F});
  EXPECT_FLOAT_EQ(restored.x, 10.0F);
  EXPECT_FLOAT_EQ(restored.y, 5.0F);
}

TEST(OpenCvCudaGeometryTest, LetterBoxesGpuMat) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "OpenCV CUDA device is unavailable";
  }

  RawImageBatch raw_images(
      std::vector<cv::cuda::GpuMat>{cv::cuda::GpuMat(10, 20, CV_8UC3)});

  ImageBatch boxed = LetterBox(raw_images, ImageSize{40, 40},
                               cv::Scalar(7, 7, 7, 7), cv::INTER_NEAREST);

  const cv::cuda::GpuMat& output = GetOpenCvCudaGpuMat(boxed.frame(0).image);
  EXPECT_EQ(output.cols, 40);
  EXPECT_EQ(output.rows, 40);
  EXPECT_EQ(output.type(), CV_8UC3);

  const GeometryStep& step = boxed.frame(0).geometry_trace.step(0);
  EXPECT_EQ(step.kind, GeometryStepKind::kLetterBox);
  EXPECT_EQ(step.letterbox.resized_size.width, 40);
  EXPECT_EQ(step.letterbox.resized_size.height, 20);
  EXPECT_EQ(step.letterbox.padding.top, 10);
}

TEST(OpenCvCudaGeometryTest, CropsThenPadsGpuMatBatch) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "OpenCV CUDA device is unavailable";
  }

  ImageBatch batch(RawImageBatch(
      std::vector<cv::cuda::GpuMat>{cv::cuda::GpuMat(10, 20, CV_8UC3)}));

  batch = Crop(std::move(batch), Rect{2, 3, 8, 4});
  batch = Pad(std::move(batch), Padding{1, 2, 3, 4}, cv::Scalar(9, 9, 9, 9));

  const cv::cuda::GpuMat& output = GetOpenCvCudaGpuMat(batch.frame(0).image);
  EXPECT_EQ(output.cols, 12);
  EXPECT_EQ(output.rows, 10);
  EXPECT_EQ(output.type(), CV_8UC3);

  const GeometryTrace& trace = batch.frame(0).geometry_trace;
  ASSERT_EQ(trace.size(), 2U);
  EXPECT_EQ(trace.step(0).kind, GeometryStepKind::kCrop);
  EXPECT_EQ(trace.step(1).kind, GeometryStepKind::kPad);
}

TEST(OpenCvCudaGeometryTest, ResizesTwoChannelGpuMatWithoutColorSemantics) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "OpenCV CUDA device is unavailable";
  }

  RawImageBatch raw_images(
      std::vector<cv::cuda::GpuMat>{cv::cuda::GpuMat(10, 20, CV_32FC2)});

  ImageBatch resized = Resize(raw_images, ImageSize{5, 5});

  const RawImage& image = resized.frame(0).image;
  EXPECT_EQ(image.pixel_format(), PixelFormat::kUnknown);
  EXPECT_EQ(image.data_type(), DataType::kFloat32);
  EXPECT_EQ(image.channels(), 2);

  const cv::cuda::GpuMat& output = GetOpenCvCudaGpuMat(image);
  EXPECT_EQ(output.cols, 5);
  EXPECT_EQ(output.rows, 5);
  EXPECT_EQ(output.type(), CV_32FC2);
}

TEST(OpenCvCudaGeometryTest, ResizesAndPadsFourChannelGpuMat) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "OpenCV CUDA device is unavailable";
  }

  ImageBatch batch(RawImageBatch(
      std::vector<cv::cuda::GpuMat>{cv::cuda::GpuMat(10, 20, CV_32FC4)}));

  batch = Resize(std::move(batch), ImageSize{5, 5}, cv::INTER_NEAREST);
  batch = Pad(std::move(batch), Padding{1, 2, 3, 4}, cv::Scalar(1, 2, 3, 4));

  const RawImage& image = batch.frame(0).image;
  EXPECT_EQ(image.pixel_format(), PixelFormat::kBgra);
  EXPECT_EQ(image.data_type(), DataType::kFloat32);
  EXPECT_EQ(image.channels(), 4);

  const cv::cuda::GpuMat& output = GetOpenCvCudaGpuMat(image);
  EXPECT_EQ(output.cols, 9);
  EXPECT_EQ(output.rows, 11);
  EXPECT_EQ(output.type(), CV_32FC4);
}

TEST(OpenCvCudaGeometryTest, RejectsSixAndNineChannelGpuMatResize) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "OpenCV CUDA device is unavailable";
  }

  {
    RawImageBatch raw_images(std::vector<cv::cuda::GpuMat>{
        cv::cuda::GpuMat(10, 20, CV_MAKETYPE(CV_8U, 6))});

    EXPECT_THROW(static_cast<void>(Resize(raw_images, ImageSize{5, 5})),
                 std::invalid_argument);
  }
  {
    RawImageBatch raw_images(std::vector<cv::cuda::GpuMat>{
        cv::cuda::GpuMat(10, 20, CV_MAKETYPE(CV_32F, 9))});

    EXPECT_THROW(static_cast<void>(Resize(raw_images, ImageSize{5, 5})),
                 std::invalid_argument);
  }
}

TEST(OpenCvCudaGeometryTest, RejectsSixAndNineChannelGpuMatPad) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "OpenCV CUDA device is unavailable";
  }

  {
    RawImageBatch raw_images(std::vector<cv::cuda::GpuMat>{
        cv::cuda::GpuMat(10, 20, CV_MAKETYPE(CV_8U, 6))});

    EXPECT_THROW(
        static_cast<void>(Pad(raw_images, Padding{1, 1, 1, 1}, cv::Scalar())),
        std::invalid_argument);
  }
  {
    RawImageBatch raw_images(std::vector<cv::cuda::GpuMat>{
        cv::cuda::GpuMat(10, 20, CV_MAKETYPE(CV_32F, 9))});

    EXPECT_THROW(
        static_cast<void>(Pad(raw_images, Padding{1, 1, 1, 1}, cv::Scalar())),
        std::invalid_argument);
  }
}

}  // namespace
}  // namespace mw::infer
