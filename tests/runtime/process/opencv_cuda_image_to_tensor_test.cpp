#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "mw/infer/runtime/input/opencv_cuda_input.h"
#include "mw/infer/runtime/input/opencv_input.h"
#include "mw/infer/runtime/process/image_to_tensor.h"
#include "mw/infer/runtime/process/normalize.h"

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

RawImage MakeCudaRawImageWithDevice(int device_id) {
  ImageDesc desc;
  desc.size = ImageSize{2, 1};
  desc.pixel_format = PixelFormat::kBgr;
  desc.data_type = DataType::kUInt8;
  desc.channels = 3;
  desc.memory_kind = ImageMemoryKind::kCuda;
  desc.device_id = device_id;
  return RawImage::FromHandle(desc, ImageHandleKind::kOpenCvCudaGpuMat,
                              cv::cuda::GpuMat());
}

std::vector<float> CopyFloatTensorToHost(const Tensor& tensor) {
  std::vector<float> values(tensor.element_count());
  EXPECT_EQ(cudaMemcpy(values.data(), tensor.data(), tensor.bytes(),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  return values;
}

void CUDART_CB DelayStream(void* user_data) {
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  static_cast<std::atomic<bool>*>(user_data)->store(true,
                                                    std::memory_order_release);
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
  ExecutionStream stream(Device{DeviceType::kCuda, 0});

  Tensor tensor = ToTensor(images, Device{DeviceType::kCuda, 0},
                           MakeInput({1, 1, 2, 3}), stream,
                           TensorLayout::kBhwc);
  stream.Synchronize();

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

TEST(OpenCvCudaImageToTensorTest,
     PreservesExternalStreamOrderThroughNormalize) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "CUDA image-to-tensor is unavailable";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

  cv::cuda::GpuMat gpu_image(1, 2, CV_8UC3);
  gpu_image.setTo(cv::Scalar::all(0));
  RawImageBatch images(std::vector<cv::cuda::GpuMat>{gpu_image});
  ExecutionStream stream(Device{DeviceType::kCuda, 0});
  PooledTensorAllocator allocator;
  {
    Tensor warm_converted = ToTensor(images, Device{DeviceType::kCuda, 0},
                                     MakeInput({1, 3, 1, 2}), stream,
                                     TensorLayout::kBchw, allocator);
    Tensor warm_normalized = Normalize(
        warm_converted, {0.0F, 0.0F, 0.0F}, {1.0F, 1.0F, 1.0F}, stream, 1.0F,
        TensorLayout::kBchw, allocator);
    stream.Synchronize();
  }

  void* source_data = gpu_image.data;
  const std::size_t source_step = gpu_image.step;
  const int source_rows = gpu_image.rows;
  const int source_cols = gpu_image.cols;
  RawImageBatch pending_images = std::move(images);
  gpu_image.release();
  std::atomic<bool> delay_completed{false};
  ASSERT_EQ(cudaLaunchHostFunc(stream.cuda_handle(), DelayStream,
                               &delay_completed),
            cudaSuccess);
  ASSERT_EQ(cudaMemset2DAsync(source_data, source_step, 7,
                              static_cast<std::size_t>(source_cols) * 3,
                              source_rows, stream.cuda_handle()),
            cudaSuccess);

  Tensor converted = ToTensor(pending_images, Device{DeviceType::kCuda, 0},
                              MakeInput({1, 3, 1, 2}), stream,
                              TensorLayout::kBchw, allocator);
  pending_images = RawImageBatch();
  Tensor normalized = Normalize(converted, {0.0F, 0.0F, 0.0F},
                                {1.0F, 1.0F, 1.0F}, stream, 1.0F,
                                TensorLayout::kBchw, allocator);
  EXPECT_FALSE(delay_completed.load(std::memory_order_acquire));
  stream.Synchronize();

  EXPECT_TRUE(delay_completed.load(std::memory_order_acquire));
  EXPECT_EQ(CopyFloatTensorToHost(normalized),
            std::vector<float>({7.0F, 7.0F, 7.0F, 7.0F, 7.0F, 7.0F}));
}

TEST(OpenCvCudaImageToTensorTest, RejectsUnsupportedFloat16Target) {
  RawImageBatch images(std::vector<cv::Mat>{MakeTestMat()});
  ImageToTensorConverter converter;
  const TensorInfo input =
      MakeInput({1, 3, 1, 2}, DataType::kFloat16);

  EXPECT_FALSE(
      converter.Supports(images, Device{DeviceType::kCuda, 0}, input));
  EXPECT_THROW(static_cast<void>(
                   ToTensor(images, Device{DeviceType::kCuda, 0}, input)),
               std::invalid_argument);
}

TEST(OpenCvCudaImageToTensorTest, RejectsGpuMatOnDifferentDevice) {
  RawImageBatch images(std::vector<RawImage>{MakeCudaRawImageWithDevice(1)});
  ImageToTensorConverter converter;

  EXPECT_FALSE(converter.Supports(images, Device{DeviceType::kCuda, 0},
                                  MakeInput({1, 3, 1, 2})));
  EXPECT_THROW(static_cast<void>(ToTensor(images, Device{DeviceType::kCuda, 0},
                                          MakeInput({1, 3, 1, 2}))),
               std::invalid_argument);
}

}  // namespace
}  // namespace mw::infer
