#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <utility>
#include <vector>

#include "mw/infer/runtime/input/opencv_cuda_input.h"
#include "mw/infer/runtime/input/opencv_input.h"
#include "mw/infer/runtime/process/image_to_tensor.h"

namespace mw::infer {
namespace {

bool HasUsableCudaDevice() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

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

cv::Mat MakeFourChannelTestMat() {
  cv::Mat image(1, 2, CV_8UC4);
  image.at<cv::Vec4b>(0, 0) = cv::Vec4b(1, 2, 3, 4);
  image.at<cv::Vec4b>(0, 1) = cv::Vec4b(5, 6, 7, 8);
  return image;
}

std::vector<float> CopyFloatTensorToHost(const Tensor& tensor) {
  std::vector<float> values(tensor.element_count());
  EXPECT_EQ(cudaMemcpy(values.data(), tensor.data(), tensor.bytes(),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  return values;
}

TEST(OpenCvCudaImageToTensorTest, UploadsBgrHostMatToRgbCudaBchwTensor) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "CUDA image-to-tensor is unavailable";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

  RawImageBatch images(std::vector<cv::Mat>{MakeTestMat()});

  Tensor tensor =
      ToTensor(images, Device{DeviceType::kCuda, 0}, MakeInput({1, 3, 1, 2}));

  ASSERT_EQ(tensor.shape(), std::vector<int64_t>({1, 3, 1, 2}));
  ASSERT_EQ(tensor.device().type, DeviceType::kCuda);
  ASSERT_EQ(tensor.data_type(), DataType::kFloat32);

  const std::vector<float> values = CopyFloatTensorToHost(tensor);
  EXPECT_FLOAT_EQ(values[0], 3.0F);
  EXPECT_FLOAT_EQ(values[1], 6.0F);
  EXPECT_FLOAT_EQ(values[2], 2.0F);
  EXPECT_FLOAT_EQ(values[3], 5.0F);
  EXPECT_FLOAT_EQ(values[4], 1.0F);
  EXPECT_FLOAT_EQ(values[5], 4.0F);
}

TEST(OpenCvCudaImageToTensorTest, ConvertsBgrGpuMatToRgbCudaBhwcTensor) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "CUDA image-to-tensor is unavailable";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

  cv::cuda::GpuMat gpu_image;
  gpu_image.upload(MakeTestMat());
  RawImageBatch images(std::vector<cv::cuda::GpuMat>{gpu_image});

  Tensor tensor = ToTensor(images, Device{DeviceType::kCuda, 0},
                           MakeInput({1, 1, 2, 3}), TensorLayout::kBhwc);

  ASSERT_EQ(tensor.shape(), std::vector<int64_t>({1, 1, 2, 3}));
  ASSERT_EQ(tensor.device().type, DeviceType::kCuda);
  ASSERT_EQ(tensor.data_type(), DataType::kFloat32);

  const std::vector<float> values = CopyFloatTensorToHost(tensor);
  EXPECT_FLOAT_EQ(values[0], 3.0F);
  EXPECT_FLOAT_EQ(values[1], 2.0F);
  EXPECT_FLOAT_EQ(values[2], 1.0F);
  EXPECT_FLOAT_EQ(values[3], 6.0F);
  EXPECT_FLOAT_EQ(values[4], 5.0F);
  EXPECT_FLOAT_EQ(values[5], 4.0F);
}

TEST(OpenCvCudaImageToTensorTest, ConvertsBgraGpuMatToRgbaTensor) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "CUDA image-to-tensor is unavailable";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

  cv::cuda::GpuMat gpu_image;
  gpu_image.upload(MakeFourChannelTestMat());
  RawImageBatch images(std::vector<cv::cuda::GpuMat>{gpu_image});

  Tensor tensor = ToTensor(images, Device{DeviceType::kCuda, 0},
                           MakeInput({1, 1, 2, 4}), TensorLayout::kBhwc);

  ASSERT_EQ(tensor.shape(), std::vector<int64_t>({1, 1, 2, 4}));
  ASSERT_EQ(tensor.device().type, DeviceType::kCuda);
  ASSERT_EQ(tensor.data_type(), DataType::kFloat32);

  const std::vector<float> values = CopyFloatTensorToHost(tensor);
  EXPECT_FLOAT_EQ(values[0], 3.0F);
  EXPECT_FLOAT_EQ(values[1], 2.0F);
  EXPECT_FLOAT_EQ(values[2], 1.0F);
  EXPECT_FLOAT_EQ(values[3], 4.0F);
  EXPECT_FLOAT_EQ(values[4], 7.0F);
  EXPECT_FLOAT_EQ(values[5], 6.0F);
  EXPECT_FLOAT_EQ(values[6], 5.0F);
  EXPECT_FLOAT_EQ(values[7], 8.0F);
}

}  // namespace
}  // namespace mw::infer
