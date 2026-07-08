#include <gtest/gtest.h>

#include <cstdint>
#include <utility>
#include <vector>

#include "mw/infer/runtime/input/opencv_input.h"
#include "mw/infer/runtime/process/image_to_tensor.h"

namespace mw::infer {
namespace {

TensorInfo MakeInput(std::vector<int64_t> shape,
                     DataType data_type = DataType::kFloat32) {
  TensorInfo input;
  input.name = "images";
  input.data_type = data_type;
  input.shape = std::move(shape);
  return input;
}

cv::Mat MakeTestMat() {
  cv::Mat image(1, 2, CV_8UC3);
  image.at<cv::Vec3b>(0, 0) = cv::Vec3b(1, 2, 3);
  image.at<cv::Vec3b>(0, 1) = cv::Vec3b(4, 5, 6);
  return image;
}

TEST(OpenCvImageToTensorTest, ConvertsHwcMatToBchwTensor) {
  RawImageBatch images(std::vector<cv::Mat>{MakeTestMat()});

  Tensor tensor =
      ToTensor(images, Device{DeviceType::kCpu, 0}, MakeInput({1, 3, 1, 2}));

  ASSERT_EQ(tensor.shape(), std::vector<int64_t>({1, 3, 1, 2}));
  ASSERT_EQ(tensor.data_type(), DataType::kFloat32);

  const auto* values = static_cast<const float*>(tensor.data());
  EXPECT_FLOAT_EQ(values[0], 1.0F);
  EXPECT_FLOAT_EQ(values[1], 4.0F);
  EXPECT_FLOAT_EQ(values[2], 2.0F);
  EXPECT_FLOAT_EQ(values[3], 5.0F);
  EXPECT_FLOAT_EQ(values[4], 3.0F);
  EXPECT_FLOAT_EQ(values[5], 6.0F);
}

TEST(OpenCvImageToTensorTest, ConvertsHwcMatToBhwcTensor) {
  RawImageBatch images(std::vector<cv::Mat>{MakeTestMat()});

  Tensor tensor = ToTensor(images, Device{DeviceType::kCpu, 0},
                           MakeInput({1, 1, 2, 3}), TensorLayout::kBhwc);

  ASSERT_EQ(tensor.shape(), std::vector<int64_t>({1, 1, 2, 3}));
  ASSERT_EQ(tensor.data_type(), DataType::kFloat32);

  const auto* values = static_cast<const float*>(tensor.data());
  EXPECT_FLOAT_EQ(values[0], 1.0F);
  EXPECT_FLOAT_EQ(values[1], 2.0F);
  EXPECT_FLOAT_EQ(values[2], 3.0F);
  EXPECT_FLOAT_EQ(values[3], 4.0F);
  EXPECT_FLOAT_EQ(values[4], 5.0F);
  EXPECT_FLOAT_EQ(values[5], 6.0F);
}

TEST(OpenCvImageToTensorTest, ConvertsToFloat16WhenModelInputRequiresIt) {
  RawImageBatch images(std::vector<cv::Mat>{MakeTestMat()});

  Tensor tensor = ToTensor(images, Device{DeviceType::kCpu, 0},
                           MakeInput({1, 3, 1, 2}, DataType::kFloat16));

  ASSERT_EQ(tensor.data_type(), DataType::kFloat16);
  const auto* values = static_cast<const uint16_t*>(tensor.data());
  EXPECT_EQ(values[0], 0x3C00U);
  EXPECT_EQ(values[1], 0x4400U);
}

TEST(OpenCvImageToTensorTest, ReusesBuffer) {
  RawImageBatch images(std::vector<cv::Mat>{MakeTestMat()});
  TensorBuffer buffer;
  const TensorInfo input = MakeInput({-1, 3, -1, -1});

  Tensor first = ToTensor(images, Device{DeviceType::kCpu, 0}, input, buffer);
  void* first_data = first.data();
  Tensor second = ToTensor(images, Device{DeviceType::kCpu, 0}, input, buffer);

  EXPECT_EQ(second.data(), first_data);
  EXPECT_EQ(buffer.capacity_bytes(), first.capacity_bytes());
}

}  // namespace
}  // namespace mw::infer
