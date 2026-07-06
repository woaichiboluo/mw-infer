#include <gtest/gtest.h>

#include <opencv2/core.hpp>

#include "mw/infer/runtime/blocks.h"

namespace mw::infer {
namespace {

TEST(OpenCvBlocksTest, ResizeUpdatesGeometry) {
  cv::Mat image(10, 20, CV_8UC3, cv::Scalar(1, 2, 3));
  auto pipeline = MakePipeline<std::vector<cv::Mat>>().Then(
      "resize", Resize(ImageSize{40, 20}));

  auto run = pipeline.RunWithContext({image});
  const std::vector<cv::Mat>& output = run.output;
  const RunContext& context = run.context;

  ASSERT_EQ(output.size(), 1U);
  EXPECT_EQ(output[0].cols, 40);
  EXPECT_EQ(output[0].rows, 20);
  ASSERT_TRUE(context.HasGeometry());
  ASSERT_NE(context.GeometryAt(0), nullptr);
  const PointF origin =
      context.GeometryAt(0)->MapPointToOriginal(PointF{20.0F, 10.0F});
  EXPECT_FLOAT_EQ(origin.x, 10.0F);
  EXPECT_FLOAT_EQ(origin.y, 5.0F);
}

TEST(OpenCvBlocksTest, LetterBoxUpdatesGeometry) {
  cv::Mat image(10, 20, CV_8UC3, cv::Scalar(1, 2, 3));
  auto pipeline =
      MakePipeline<std::vector<cv::Mat>>().Then("letter_box", LetterBox());

  auto run = pipeline.RunWithContext({image});
  const std::vector<cv::Mat>& output = run.output;
  const RunContext& context = run.context;

  ASSERT_EQ(output.size(), 1U);
  EXPECT_EQ(output[0].cols, 640);
  EXPECT_EQ(output[0].rows, 640);
  EXPECT_EQ(output[0].at<cv::Vec3b>(0, 0), cv::Vec3b(114, 114, 114));
  EXPECT_EQ(output[0].at<cv::Vec3b>(320, 320), cv::Vec3b(1, 2, 3));

  ASSERT_NE(context.GeometryAt(0), nullptr);
  const PointF origin =
      context.GeometryAt(0)->MapPointToOriginal(PointF{0.0F, 160.0F});
  EXPECT_FLOAT_EQ(origin.x, 0.0F);
  EXPECT_FLOAT_EQ(origin.y, 0.0F);
}

TEST(OpenCvBlocksTest, NormalizeKeepsOpenCvMatOutput) {
  cv::Mat image(1, 2, CV_8UC3);
  image.at<cv::Vec3b>(0, 0) = cv::Vec3b(1, 2, 3);
  image.at<cv::Vec3b>(0, 1) = cv::Vec3b(4, 5, 6);

  RunContext context;
  std::vector<cv::Mat> normalized =
      Normalize({0.0F, 0.0F, 0.0F}, {1.0F, 1.0F, 1.0F}, 1.0F, false)
          .Run({image}, context);

  ASSERT_EQ(normalized.size(), 1U);
  EXPECT_EQ(normalized[0].type(), CV_32FC3);
  EXPECT_EQ(normalized[0].rows, 1);
  EXPECT_EQ(normalized[0].cols, 2);
  EXPECT_FLOAT_EQ(normalized[0].at<cv::Vec3f>(0, 0)[0], 1.0F);
  EXPECT_FLOAT_EQ(normalized[0].at<cv::Vec3f>(0, 0)[1], 2.0F);
  EXPECT_FLOAT_EQ(normalized[0].at<cv::Vec3f>(0, 0)[2], 3.0F);
  EXPECT_FLOAT_EQ(normalized[0].at<cv::Vec3f>(0, 1)[0], 4.0F);
  EXPECT_FLOAT_EQ(normalized[0].at<cv::Vec3f>(0, 1)[1], 5.0F);
  EXPECT_FLOAT_EQ(normalized[0].at<cv::Vec3f>(0, 1)[2], 6.0F);
}

}  // namespace
}  // namespace mw::infer
