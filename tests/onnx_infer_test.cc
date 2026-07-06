#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <stdexcept>
#include <vector>

#include "mw/infer/common/device.h"
#include "mw/infer/runtime/backend.h"
#include "mw/infer/runtime/blocks.h"

namespace mw::infer {
namespace {

constexpr std::array<std::uint8_t, 127> kIdentityOnnx = {
    8,   7,   18,  11,  77,  119, 73,  110, 102, 101, 114, 84,  101, 115, 116,
    58,  104, 10,  25,  10,  5,   105, 110, 112, 117, 116, 18,  6,   111, 117,
    116, 112, 117, 116, 34,  8,   73,  100, 101, 110, 116, 105, 116, 121, 18,
    8,   105, 100, 101, 110, 116, 105, 116, 121, 90,  31,  10,  5,   105, 110,
    112, 117, 116, 18,  22,  10,  20,  8,   1,   18,  16,  10,  2,   8,   1,
    10,  2,   8,   3,   10,  2,   8,   2,   10,  2,   8,   2,   98,  32,  10,
    6,   111, 117, 116, 112, 117, 116, 18,  22,  10,  20,  8,   1,   18,  16,
    10,  2,   8,   1,   10,  2,   8,   3,   10,  2,   8,   2,   10,  2,   8,
    2,   66,  4,   10,  0,   16,  13};

constexpr std::array<std::uint8_t, 168> kDynamicIdentityOnnx = {
    8,   7,   18,  11,  77,  119, 73,  110, 102, 101, 114, 84,  101, 115,
    116, 58,  144, 1,   10,  25,  10,  5,   105, 110, 112, 117, 116, 18,
    6,   111, 117, 116, 112, 117, 116, 34,  8,   73,  100, 101, 110, 116,
    105, 116, 121, 18,  16,  100, 121, 110, 97,  109, 105, 99,  95,  105,
    100, 101, 110, 116, 105, 116, 121, 90,  47,  10,  5,   105, 110, 112,
    117, 116, 18,  38,  10,  36,  8,   1,   18,  32,  10,  7,   18,  5,
    98,  97,  116, 99,  104, 10,  2,   8,   3,   10,  8,   18,  6,   104,
    101, 105, 103, 104, 116, 10,  7,   18,  5,   119, 105, 100, 116, 104,
    98,  48,  10,  6,   111, 117, 116, 112, 117, 116, 18,  38,  10,  36,
    8,   1,   18,  32,  10,  7,   18,  5,   98,  97,  116, 99,  104, 10,
    2,   8,   3,   10,  8,   18,  6,   104, 101, 105, 103, 104, 116, 10,
    7,   18,  5,   119, 105, 100, 116, 104, 66,  4,   10,  0,   16,  13};

Model MakeMemoryModel(const std::uint8_t* data, std::size_t size) {
  Model model;
  model.format = ModelFormat::kOnnx;
  model.source = ModelSourceFromMemory(data, size);
  return model;
}

cv::Mat MakeImage2x2(float base) {
  cv::Mat image(2, 2, CV_32FC3);
  image.at<cv::Vec3f>(0, 0) = cv::Vec3f(base + 0.0F, base + 1.0F, base + 2.0F);
  image.at<cv::Vec3f>(0, 1) = cv::Vec3f(base + 3.0F, base + 4.0F, base + 5.0F);
  image.at<cv::Vec3f>(1, 0) = cv::Vec3f(base + 6.0F, base + 7.0F, base + 8.0F);
  image.at<cv::Vec3f>(1, 1) =
      cv::Vec3f(base + 9.0F, base + 10.0F, base + 11.0F);
  return image;
}

void ExpectFloatOutput(const InferOutput& output,
                       const std::vector<float>& expected) {
  ASSERT_NE(output.buffer.host, nullptr);
  EXPECT_EQ(output.buffer.size_bytes, expected.size() * sizeof(float));

  const auto* values = static_cast<const float*>(output.buffer.host);
  for (std::size_t index = 0; index < expected.size(); ++index) {
    EXPECT_FLOAT_EQ(values[index], expected[index]);
  }
}

#if MW_INFER_WITH_OPENCV_CUDA

void ExpectGpuFloatOutput(const InferOutput& output,
                          const std::vector<float>& expected) {
  ASSERT_NE(output.buffer.device, nullptr);
  EXPECT_EQ(output.buffer.size_bytes, expected.size() * sizeof(float));

  cv::cuda::GpuMat device_values(1, static_cast<int>(expected.size()), CV_32FC1,
                                 const_cast<void*>(output.buffer.device),
                                 expected.size() * sizeof(float));
  cv::Mat host_values;
  device_values.download(host_values);

  const auto* values = host_values.ptr<float>();
  for (std::size_t index = 0; index < expected.size(); ++index) {
    EXPECT_FLOAT_EQ(values[index], expected[index]);
  }
}

#endif  // MW_INFER_WITH_OPENCV_CUDA

TEST(OnnxInferTest, RunsCpuIdentityModelFromOpenCvMat) {
  RuntimeConfig config;
  config.backend = BackendKind::kOnnxCpu;
  config.model = MakeMemoryModel(kIdentityOnnx.data(), kIdentityOnnx.size());
  config.input_name = "input";
  config.output_names = {"output"};

  cv::Mat image(2, 2, CV_32FC3);
  image.at<cv::Vec3f>(0, 0) = cv::Vec3f(1.0F, 2.0F, 3.0F);
  image.at<cv::Vec3f>(0, 1) = cv::Vec3f(4.0F, 5.0F, 6.0F);
  image.at<cv::Vec3f>(1, 0) = cv::Vec3f(7.0F, 8.0F, 9.0F);
  image.at<cv::Vec3f>(1, 1) = cv::Vec3f(10.0F, 11.0F, 12.0F);

  auto pipeline =
      MakePipeline<std::vector<cv::Mat>>().Then("infer", OnnxInfer(config));
  auto run = pipeline.RunWithContext({image});
  const InferOutputs& outputs = run.output;
  const RunContext& context = run.context;

  ASSERT_TRUE(context.HasGeometry());
  ASSERT_NE(context.GeometryAt(0), nullptr);
  EXPECT_EQ(context.GeometryAt(0)->original_size().width, 2);
  EXPECT_EQ(context.GeometryAt(0)->original_size().height, 2);

  ASSERT_EQ(outputs.batch_size, 1);
  ASSERT_EQ(outputs.outputs.size(), 1U);
  const InferOutput& output = outputs.outputs[0];
  EXPECT_EQ(output.name, "output");
  EXPECT_EQ(output.data_type, InferDataType::kFloat32);
  EXPECT_EQ(output.shape, (Shape{1, 3, 2, 2}));
  ASSERT_NE(output.buffer.host, nullptr);
  EXPECT_EQ(output.buffer.size_bytes, 12U * sizeof(float));

  ExpectFloatOutput(output, {1.0F, 4.0F, 7.0F, 10.0F, 2.0F, 5.0F, 8.0F, 11.0F,
                             3.0F, 6.0F, 9.0F, 12.0F});
}

#if MW_INFER_WITH_OPENCV_CUDA

TEST(OnnxGpuBackendTest, RunsGpuMatBatchWithoutCpuDownload) {
  if (cv::cuda::getCudaEnabledDeviceCount() <= 0) {
    GTEST_SKIP() << "OpenCV CUDA device is unavailable";
  }

  RuntimeConfig config;
  config.backend = BackendKind::kOnnxGpu;
  config.model =
      MakeMemoryModel(kDynamicIdentityOnnx.data(), kDynamicIdentityOnnx.size());
  config.input_name = "input";
  config.output_names = {"output"};

  cv::cuda::GpuMat image_a;
  cv::cuda::GpuMat image_b;
  image_a.upload(MakeImage2x2(1.0F));
  image_b.upload(MakeImage2x2(101.0F));

  std::unique_ptr<IBackend> backend = CreateBackend(config);
  const InferOutputs& outputs =
      backend->InferBatch(std::vector<cv::cuda::GpuMat>{image_a, image_b});

  ASSERT_EQ(outputs.batch_size, 2);
  ASSERT_EQ(outputs.outputs.size(), 1U);
  const InferOutput& output = outputs.outputs[0];
  EXPECT_EQ(output.name, "output");
  EXPECT_EQ(output.data_type, InferDataType::kFloat32);
  EXPECT_EQ(output.device.type(), DeviceType::kCuda);
  EXPECT_EQ(output.shape, (Shape{2, 3, 2, 2}));
  ExpectGpuFloatOutput(
      output, {1.0F,   4.0F,   7.0F,   10.0F,  2.0F,   5.0F,   8.0F,   11.0F,
               3.0F,   6.0F,   9.0F,   12.0F,  101.0F, 104.0F, 107.0F, 110.0F,
               102.0F, 105.0F, 108.0F, 111.0F, 103.0F, 106.0F, 109.0F, 112.0F});
}

#endif  // MW_INFER_WITH_OPENCV_CUDA

TEST(OnnxCpuBackendTest, SupportsDynamicBatchAndDynamicImageSize) {
  RuntimeConfig config;
  config.backend = BackendKind::kOnnxCpu;
  config.model =
      MakeMemoryModel(kDynamicIdentityOnnx.data(), kDynamicIdentityOnnx.size());
  config.input_name = "input";
  config.output_names = {"output"};

  std::unique_ptr<IBackend> backend = CreateBackend(config);
  const InferOutputs& first_outputs = backend->InferBatch(
      std::vector<cv::Mat>{MakeImage2x2(1.0F), MakeImage2x2(101.0F)});

  ASSERT_EQ(first_outputs.batch_size, 2);
  ASSERT_EQ(first_outputs.outputs.size(), 1U);
  const InferOutput& first_output = first_outputs.outputs[0];
  EXPECT_EQ(first_output.name, "output");
  EXPECT_EQ(first_output.data_type, InferDataType::kFloat32);
  EXPECT_EQ(first_output.shape, (Shape{2, 3, 2, 2}));
  ExpectFloatOutput(
      first_output,
      {1.0F,   4.0F,   7.0F,   10.0F,  2.0F,   5.0F,   8.0F,   11.0F,
       3.0F,   6.0F,   9.0F,   12.0F,  101.0F, 104.0F, 107.0F, 110.0F,
       102.0F, 105.0F, 108.0F, 111.0F, 103.0F, 106.0F, 109.0F, 112.0F});

  cv::Mat tall_image(3, 1, CV_32FC3);
  tall_image.at<cv::Vec3f>(0, 0) = cv::Vec3f(1.0F, 2.0F, 3.0F);
  tall_image.at<cv::Vec3f>(1, 0) = cv::Vec3f(4.0F, 5.0F, 6.0F);
  tall_image.at<cv::Vec3f>(2, 0) = cv::Vec3f(7.0F, 8.0F, 9.0F);

  const InferOutputs& second_outputs =
      backend->InferBatch(std::vector<cv::Mat>{tall_image});

  ASSERT_EQ(second_outputs.batch_size, 1);
  ASSERT_EQ(second_outputs.outputs.size(), 1U);
  const InferOutput& second_output = second_outputs.outputs[0];
  EXPECT_EQ(second_output.shape, (Shape{1, 3, 3, 1}));
  ExpectFloatOutput(second_output,
                    {1.0F, 4.0F, 7.0F, 2.0F, 5.0F, 8.0F, 3.0F, 6.0F, 9.0F});
}

TEST(OnnxCpuBackendTest, RejectsMixedImageShapeInOneBatch) {
  RuntimeConfig config;
  config.backend = BackendKind::kOnnxCpu;
  config.model =
      MakeMemoryModel(kDynamicIdentityOnnx.data(), kDynamicIdentityOnnx.size());

  std::unique_ptr<IBackend> backend = CreateBackend(config);
  cv::Mat image_a(2, 2, CV_32FC3);
  cv::Mat image_b(3, 2, CV_32FC3);

  EXPECT_THROW(static_cast<void>(
                   backend->InferBatch(std::vector<cv::Mat>{image_a, image_b})),
               std::invalid_argument);
}

}  // namespace
}  // namespace mw::infer
