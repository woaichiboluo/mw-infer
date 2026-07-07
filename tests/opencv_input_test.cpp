#include "mw/infer/runtime/adapters/opencv_input.h"

#include <gtest/gtest.h>

#include <opencv2/core.hpp>
#include <stdexcept>
#include <vector>

namespace mw::infer {
namespace {

TEST(OpenCvInputTest, WrapsMatAsRawImage) {
  cv::Mat mat(10, 20, CV_8UC3, cv::Scalar(1, 2, 3));

  RawImage image = ToRawImage(mat);

  EXPECT_EQ(image.memory_kind(), ImageMemoryKind::kHost);
  EXPECT_EQ(image.handle_kind(), ImageHandleKind::kOpenCvMat);
  EXPECT_EQ(image.pixel_format(), PixelFormat::kBgr);
  EXPECT_EQ(image.data_type(), DataType::kUInt8);
  EXPECT_EQ(image.channels(), 3);
  EXPECT_EQ(image.size().width, 20);
  EXPECT_EQ(image.size().height, 10);
  ASSERT_EQ(image.desc().planes.size(), 1U);
  EXPECT_EQ(image.desc().planes[0].row_stride_bytes, mat.step[0]);
  EXPECT_EQ(image.desc().planes[0].pixel_stride_bytes, mat.elemSize());

  const cv::Mat& stored = GetOpenCvMat(image);
  EXPECT_EQ(stored.rows, 10);
  EXPECT_EQ(stored.cols, 20);
  EXPECT_EQ(stored.type(), CV_8UC3);
}

TEST(OpenCvInputTest, DetectsGrayMat) {
  cv::Mat mat(10, 20, CV_8UC1);

  RawImage image = ToRawImage(mat);

  EXPECT_EQ(image.memory_kind(), ImageMemoryKind::kHost);
  EXPECT_EQ(image.handle_kind(), ImageHandleKind::kOpenCvMat);
  EXPECT_EQ(image.pixel_format(), PixelFormat::kGray);
  EXPECT_EQ(image.data_type(), DataType::kUInt8);
  EXPECT_EQ(image.channels(), 1);
}

TEST(OpenCvInputTest, ConstructsRawImageFromMat) {
  cv::Mat mat(10, 20, CV_8UC3);

  RawImage image(mat);

  EXPECT_EQ(image.memory_kind(), ImageMemoryKind::kHost);
  EXPECT_EQ(image.handle_kind(), ImageHandleKind::kOpenCvMat);
  EXPECT_EQ(image.pixel_format(), PixelFormat::kBgr);
}

TEST(OpenCvInputTest, WrapsTwoChannelMatAsUnknownFormat) {
  cv::Mat mat(10, 20, CV_32FC2);

  RawImage image = ToRawImage(mat);

  EXPECT_EQ(image.memory_kind(), ImageMemoryKind::kHost);
  EXPECT_EQ(image.handle_kind(), ImageHandleKind::kOpenCvMat);
  EXPECT_EQ(image.pixel_format(), PixelFormat::kUnknown);
  EXPECT_EQ(image.data_type(), DataType::kFloat32);
  EXPECT_EQ(image.channels(), 2);
}

TEST(OpenCvInputTest, WrapsFourSixAndNineChannelMats) {
  {
    cv::Mat mat(10, 20, CV_8UC4);

    RawImage image = ToRawImage(mat);

    EXPECT_EQ(image.pixel_format(), PixelFormat::kBgra);
    EXPECT_EQ(image.data_type(), DataType::kUInt8);
    EXPECT_EQ(image.channels(), 4);
  }
  {
    cv::Mat mat(10, 20, CV_MAKETYPE(CV_8U, 6));

    RawImage image = ToRawImage(mat);

    EXPECT_EQ(image.pixel_format(), PixelFormat::kUnknown);
    EXPECT_EQ(image.data_type(), DataType::kUInt8);
    EXPECT_EQ(image.channels(), 6);
  }
  {
    cv::Mat mat(10, 20, CV_MAKETYPE(CV_32F, 9));

    RawImage image = ToRawImage(mat);

    EXPECT_EQ(image.pixel_format(), PixelFormat::kUnknown);
    EXPECT_EQ(image.data_type(), DataType::kFloat32);
    EXPECT_EQ(image.channels(), 9);
  }
}

TEST(OpenCvInputTest, WrapsMatBatch) {
  RawImageBatch batch = ToRawImageBatch(
      std::vector<cv::Mat>{cv::Mat(10, 20, CV_8UC3), cv::Mat(11, 21, CV_8UC3)});

  ASSERT_EQ(batch.size(), 2U);
  EXPECT_EQ(batch.memory_kind(), ImageMemoryKind::kHost);
  EXPECT_EQ(batch.image(0).pixel_format(), PixelFormat::kBgr);
  EXPECT_EQ(batch.image(0).channels(), 3);

  std::vector<cv::Mat> mats = GetOpenCvMatBatch(batch);
  ASSERT_EQ(mats.size(), 2U);
  EXPECT_EQ(mats[0].rows, 10);
  EXPECT_EQ(mats[1].cols, 21);
}

TEST(OpenCvInputTest, ConstructsRawImageBatchFromMatVector) {
  RawImageBatch batch(
      std::vector<cv::Mat>{cv::Mat(10, 20, CV_8UC3), cv::Mat(11, 21, CV_8UC3)});

  ASSERT_EQ(batch.size(), 2U);
  EXPECT_EQ(batch.memory_kind(), ImageMemoryKind::kHost);
  EXPECT_EQ(batch.image(1).size().width, 21);
}

TEST(OpenCvInputTest, RejectsEmptyMat) {
  EXPECT_THROW(static_cast<void>(ToRawImage(cv::Mat())), std::invalid_argument);
}

TEST(OpenCvInputTest, RejectsNonOpenCvMatHostImage) {
  ImageDesc desc;
  desc.memory_kind = ImageMemoryKind::kHost;

  RawImage image = RawImage::FromHandle(desc, ImageHandleKind::kNone, 1);

  EXPECT_THROW(static_cast<void>(GetOpenCvMat(image)), std::invalid_argument);
}

}  // namespace
}  // namespace mw::infer
