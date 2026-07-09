#include "mw/infer/runtime/input/opencv_cuda_input.h"

#include <gtest/gtest.h>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <stdexcept>
#include <vector>

namespace mw::infer {
namespace {

bool HasUsableCudaDevice() {
  try {
    return cv::cuda::getCudaEnabledDeviceCount() > 0;
  } catch (const cv::Exception&) {
    return false;
  }
}

RawImage MakeCudaRawImageWithDevice(int device_id) {
  ImageDesc desc;
  desc.size = ImageSize{20, 10};
  desc.pixel_format = PixelFormat::kBgr;
  desc.data_type = DataType::kUInt8;
  desc.channels = 3;
  desc.memory_kind = ImageMemoryKind::kCuda;
  desc.device_id = device_id;
  return RawImage::FromHandle(desc, ImageHandleKind::kOpenCvCudaGpuMat,
                              cv::cuda::GpuMat());
}

TEST(OpenCvCudaInputTest, WrapsGpuMatAsRawImage) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "OpenCV CUDA device is unavailable";
  }

  const int device_id = cv::cuda::getDevice();
  cv::cuda::GpuMat mat(10, 20, CV_8UC3);

  RawImage image = ToRawImage(mat);

  EXPECT_EQ(image.memory_kind(), ImageMemoryKind::kCuda);
  EXPECT_EQ(image.device().type, DeviceType::kCuda);
  EXPECT_EQ(image.device().id, device_id);
  EXPECT_EQ(image.handle_kind(), ImageHandleKind::kOpenCvCudaGpuMat);
  EXPECT_EQ(image.pixel_format(), PixelFormat::kBgr);
  EXPECT_EQ(image.data_type(), DataType::kUInt8);
  EXPECT_EQ(image.channels(), 3);
  EXPECT_EQ(image.size().width, 20);
  EXPECT_EQ(image.size().height, 10);
  ASSERT_EQ(image.desc().planes.size(), 1U);
  EXPECT_EQ(image.desc().planes[0].row_stride_bytes, mat.step);
  EXPECT_EQ(image.desc().planes[0].pixel_stride_bytes, mat.elemSize());

  const cv::cuda::GpuMat& stored = GetOpenCvCudaGpuMat(image);
  EXPECT_EQ(stored.rows, 10);
  EXPECT_EQ(stored.cols, 20);
  EXPECT_EQ(stored.type(), CV_8UC3);
}

TEST(OpenCvCudaInputTest, DetectsGrayGpuMat) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "OpenCV CUDA device is unavailable";
  }

  cv::cuda::GpuMat mat(10, 20, CV_8UC1);

  RawImage image = ToRawImage(mat);

  EXPECT_EQ(image.memory_kind(), ImageMemoryKind::kCuda);
  EXPECT_EQ(image.handle_kind(), ImageHandleKind::kOpenCvCudaGpuMat);
  EXPECT_EQ(image.pixel_format(), PixelFormat::kGray);
  EXPECT_EQ(image.data_type(), DataType::kUInt8);
  EXPECT_EQ(image.channels(), 1);
}

TEST(OpenCvCudaInputTest, ConstructsRawImageFromGpuMat) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "OpenCV CUDA device is unavailable";
  }

  cv::cuda::GpuMat mat(10, 20, CV_8UC3);

  RawImage image(mat);

  EXPECT_EQ(image.memory_kind(), ImageMemoryKind::kCuda);
  EXPECT_EQ(image.handle_kind(), ImageHandleKind::kOpenCvCudaGpuMat);
  EXPECT_EQ(image.pixel_format(), PixelFormat::kBgr);
}

TEST(OpenCvCudaInputTest, WrapsTwoChannelGpuMatAsUnknownFormat) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "OpenCV CUDA device is unavailable";
  }

  cv::cuda::GpuMat mat(10, 20, CV_32FC2);

  RawImage image = ToRawImage(mat);

  EXPECT_EQ(image.memory_kind(), ImageMemoryKind::kCuda);
  EXPECT_EQ(image.handle_kind(), ImageHandleKind::kOpenCvCudaGpuMat);
  EXPECT_EQ(image.pixel_format(), PixelFormat::kUnknown);
  EXPECT_EQ(image.data_type(), DataType::kFloat32);
  EXPECT_EQ(image.channels(), 2);
}

TEST(OpenCvCudaInputTest, WrapsFourSixAndNineChannelGpuMats) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "OpenCV CUDA device is unavailable";
  }

  {
    cv::cuda::GpuMat mat(10, 20, CV_8UC4);

    RawImage image = ToRawImage(mat);

    EXPECT_EQ(image.pixel_format(), PixelFormat::kBgra);
    EXPECT_EQ(image.data_type(), DataType::kUInt8);
    EXPECT_EQ(image.channels(), 4);
  }
  {
    cv::cuda::GpuMat mat(10, 20, CV_MAKETYPE(CV_8U, 6));

    RawImage image = ToRawImage(mat);

    EXPECT_EQ(image.pixel_format(), PixelFormat::kUnknown);
    EXPECT_EQ(image.data_type(), DataType::kUInt8);
    EXPECT_EQ(image.channels(), 6);
  }
  {
    cv::cuda::GpuMat mat(10, 20, CV_MAKETYPE(CV_32F, 9));

    RawImage image = ToRawImage(mat);

    EXPECT_EQ(image.pixel_format(), PixelFormat::kUnknown);
    EXPECT_EQ(image.data_type(), DataType::kFloat32);
    EXPECT_EQ(image.channels(), 9);
  }
}

TEST(OpenCvCudaInputTest, WrapsGpuMatBatch) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "OpenCV CUDA device is unavailable";
  }

  RawImageBatch batch = ToRawImageBatch(std::vector<cv::cuda::GpuMat>{
      cv::cuda::GpuMat(10, 20, CV_8UC3), cv::cuda::GpuMat(11, 21, CV_8UC3)});

  ASSERT_EQ(batch.size(), 2U);
  EXPECT_EQ(batch.memory_kind(), ImageMemoryKind::kCuda);
  EXPECT_EQ(batch.image(0).device().type, DeviceType::kCuda);
  EXPECT_EQ(batch.image(0).pixel_format(), PixelFormat::kBgr);
  EXPECT_EQ(batch.image(0).channels(), 3);

  std::vector<cv::cuda::GpuMat> mats = GetOpenCvCudaGpuMatBatch(batch);
  ASSERT_EQ(mats.size(), 2U);
  EXPECT_EQ(mats[0].rows, 10);
  EXPECT_EQ(mats[1].cols, 21);
}

TEST(OpenCvCudaInputTest, ConstructsRawImageBatchFromGpuMatVector) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "OpenCV CUDA device is unavailable";
  }

  RawImageBatch batch(std::vector<cv::cuda::GpuMat>{
      cv::cuda::GpuMat(10, 20, CV_8UC3), cv::cuda::GpuMat(11, 21, CV_8UC3)});

  ASSERT_EQ(batch.size(), 2U);
  EXPECT_EQ(batch.memory_kind(), ImageMemoryKind::kCuda);
  EXPECT_EQ(batch.image(1).size().width, 21);
}

TEST(OpenCvCudaInputTest, RejectsMixedCudaDevices) {
  EXPECT_THROW(
      static_cast<void>(RawImageBatch(std::vector<RawImage>{
          MakeCudaRawImageWithDevice(0), MakeCudaRawImageWithDevice(1)})),
      std::invalid_argument);
}

TEST(OpenCvCudaInputTest, RejectsEmptyGpuMat) {
  EXPECT_THROW(static_cast<void>(ToRawImage(cv::cuda::GpuMat())),
               std::invalid_argument);
}

TEST(OpenCvCudaInputTest, RejectsNonOpenCvCudaGpuMatCudaImage) {
  ImageDesc desc;
  desc.memory_kind = ImageMemoryKind::kCuda;

  RawImage image = RawImage::FromHandle(desc, ImageHandleKind::kNone, 1);

  EXPECT_THROW(static_cast<void>(GetOpenCvCudaGpuMat(image)),
               std::invalid_argument);
}

}  // namespace
}  // namespace mw::infer
