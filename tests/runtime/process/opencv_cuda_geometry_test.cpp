#include <gtest/gtest.h>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <utility>
#include <vector>

#include "mw/infer/runtime/input/opencv_cuda_input.h"
#include "mw/infer/runtime/process/geometry.h"

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

  const int device_id = cv::cuda::getDevice();
  GeometryTransformer transformer;
  RawImageBatch raw_images(
      std::vector<cv::cuda::GpuMat>{cv::cuda::GpuMat(10, 20, CV_8UC3)});

  GeometryResult resized = transformer.Resize(raw_images, ImageSize{40, 20},
                                              Interpolation::kNearest);

  ASSERT_EQ(resized.size(), 1U);
  EXPECT_EQ(resized.images().image(0).device().type, DeviceType::kCuda);
  EXPECT_EQ(resized.images().image(0).device().id, device_id);
  const cv::cuda::GpuMat& output =
      GetOpenCvCudaGpuMat(resized.images().image(0));
  EXPECT_EQ(output.cols, 40);
  EXPECT_EQ(output.rows, 20);
  EXPECT_EQ(output.type(), CV_8UC3);

  const GeometryTrace& trace = resized.trace(0);
  ASSERT_EQ(trace.size(), 1U);
  EXPECT_EQ(trace.step(0).kind, GeometryStepKind::kResize);

  const Point2f restored = trace.RestorePoint(Point2f{20.0F, 10.0F});
  EXPECT_FLOAT_EQ(restored.x, 10.0F);
  EXPECT_FLOAT_EQ(restored.y, 5.0F);
}

TEST(OpenCvCudaGeometryTest, ResizesGpuMatShortSide) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "OpenCV CUDA device is unavailable";
  }

  GeometryTransformer transformer;
  RawImageBatch raw_images(
      std::vector<cv::cuda::GpuMat>{cv::cuda::GpuMat(10, 20, CV_8UC3)});

  GeometryResult resized =
      transformer.ResizeShortSide(raw_images, 5, Interpolation::kNearest);

  const cv::cuda::GpuMat& output =
      GetOpenCvCudaGpuMat(resized.images().image(0));
  EXPECT_EQ(output.cols, 10);
  EXPECT_EQ(output.rows, 5);
  EXPECT_EQ(output.type(), CV_8UC3);

  const GeometryStep& step = resized.trace(0).step(0);
  EXPECT_EQ(step.kind, GeometryStepKind::kResize);
  EXPECT_EQ(step.after_size.width, 10);
  EXPECT_EQ(step.after_size.height, 5);
}

TEST(OpenCvCudaGeometryTest, LetterBoxesGpuMat) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "OpenCV CUDA device is unavailable";
  }

  GeometryTransformer transformer;
  RawImageBatch raw_images(
      std::vector<cv::cuda::GpuMat>{cv::cuda::GpuMat(10, 20, CV_8UC3)});

  GeometryResult boxed =
      transformer.LetterBox(raw_images, ImageSize{40, 40},
                            Interpolation::kNearest, FillValue{{7, 7, 7, 7}});

  const cv::cuda::GpuMat& output = GetOpenCvCudaGpuMat(boxed.images().image(0));
  EXPECT_EQ(output.cols, 40);
  EXPECT_EQ(output.rows, 40);
  EXPECT_EQ(output.type(), CV_8UC3);

  const GeometryStep& step = boxed.trace(0).step(0);
  EXPECT_EQ(step.kind, GeometryStepKind::kLetterBox);
  EXPECT_EQ(step.letterbox.resized_size.width, 40);
  EXPECT_EQ(step.letterbox.resized_size.height, 20);
  EXPECT_EQ(step.letterbox.padding.top, 10);
}

TEST(OpenCvCudaGeometryTest, CropsThenPadsGeometryResult) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "OpenCV CUDA device is unavailable";
  }

  GeometryTransformer transformer;
  GeometryResult batch(RawImageBatch(
      std::vector<cv::cuda::GpuMat>{cv::cuda::GpuMat(10, 20, CV_8UC3)}));

  batch = transformer.Crop(std::move(batch), Rect{2, 3, 8, 4});
  batch = transformer.Pad(std::move(batch), Padding{1, 2, 3, 4},
                          FillValue{{9, 9, 9, 9}});

  const cv::cuda::GpuMat& output = GetOpenCvCudaGpuMat(batch.images().image(0));
  EXPECT_EQ(output.cols, 12);
  EXPECT_EQ(output.rows, 10);
  EXPECT_EQ(output.type(), CV_8UC3);

  const GeometryTrace& trace = batch.trace(0);
  ASSERT_EQ(trace.size(), 2U);
  EXPECT_EQ(trace.step(0).kind, GeometryStepKind::kCrop);
  EXPECT_EQ(trace.step(1).kind, GeometryStepKind::kPad);
}

TEST(OpenCvCudaGeometryTest, ResizesTwoChannelGpuMatWithoutColorSemantics) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "OpenCV CUDA device is unavailable";
  }

  GeometryTransformer transformer;
  RawImageBatch raw_images(
      std::vector<cv::cuda::GpuMat>{cv::cuda::GpuMat(10, 20, CV_32FC2)});

  GeometryResult resized = transformer.Resize(raw_images, ImageSize{5, 5});

  const RawImage& image = resized.images().image(0);
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

  GeometryTransformer transformer;
  GeometryResult batch(RawImageBatch(
      std::vector<cv::cuda::GpuMat>{cv::cuda::GpuMat(10, 20, CV_32FC4)}));

  batch = transformer.Resize(std::move(batch), ImageSize{5, 5},
                             Interpolation::kNearest);
  batch = transformer.Pad(std::move(batch), Padding{1, 2, 3, 4},
                          FillValue{{1, 2, 3, 4}});

  const RawImage& image = batch.images().image(0);
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

  GeometryTransformer transformer;
  {
    RawImageBatch raw_images(std::vector<cv::cuda::GpuMat>{
        cv::cuda::GpuMat(10, 20, CV_MAKETYPE(CV_8U, 6))});

    EXPECT_THROW(
        static_cast<void>(transformer.Resize(raw_images, ImageSize{5, 5})),
        std::invalid_argument);
  }
  {
    RawImageBatch raw_images(std::vector<cv::cuda::GpuMat>{
        cv::cuda::GpuMat(10, 20, CV_MAKETYPE(CV_32F, 9))});

    EXPECT_THROW(
        static_cast<void>(transformer.Resize(raw_images, ImageSize{5, 5})),
        std::invalid_argument);
  }
}

TEST(OpenCvCudaGeometryTest, RejectsSixAndNineChannelGpuMatPad) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "OpenCV CUDA device is unavailable";
  }

  GeometryTransformer transformer;
  {
    RawImageBatch raw_images(std::vector<cv::cuda::GpuMat>{
        cv::cuda::GpuMat(10, 20, CV_MAKETYPE(CV_8U, 6))});

    EXPECT_THROW(static_cast<void>(transformer.Pad(
                     raw_images, Padding{1, 1, 1, 1}, FillValue{})),
                 std::invalid_argument);
  }
  {
    RawImageBatch raw_images(std::vector<cv::cuda::GpuMat>{
        cv::cuda::GpuMat(10, 20, CV_MAKETYPE(CV_32F, 9))});

    EXPECT_THROW(static_cast<void>(transformer.Pad(
                     raw_images, Padding{1, 1, 1, 1}, FillValue{})),
                 std::invalid_argument);
  }
}

}  // namespace
}  // namespace mw::infer
