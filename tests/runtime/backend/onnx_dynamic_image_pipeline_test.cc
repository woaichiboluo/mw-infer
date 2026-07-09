#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <opencv2/core.hpp>
#include <vector>

#include "mw/infer/runtime/backend/backend.h"
#include "mw/infer/runtime/input/opencv_input.h"
#include "mw/infer/runtime/process/geometry.h"
#include "mw/infer/runtime/process/image_to_tensor.h"

namespace mw::infer {
namespace {

Model DynamicImageIdentityModel() {
  static constexpr std::array<uint8_t, 176> kModel = {
      8,   8,   18,  13,  109, 119, 45,  105, 110, 102, 101, 114, 45,  116, 101,
      115, 116, 58,  150, 1,   10,  25,  10,  5,   105, 110, 112, 117, 116, 18,
      6,   111, 117, 116, 112, 117, 116, 34,  8,   73,  100, 101, 110, 116, 105,
      116, 121, 18,  22,  100, 121, 110, 97,  109, 105, 99,  95,  105, 109, 97,
      103, 101, 95,  105, 100, 101, 110, 116, 105, 116, 121, 90,  47,  10,  5,
      105, 110, 112, 117, 116, 18,  38,  10,  36,  8,   1,   18,  32,  10,  7,
      18,  5,   98,  97,  116, 99,  104, 10,  2,   8,   3,   10,  8,   18,  6,
      104, 101, 105, 103, 104, 116, 10,  7,   18,  5,   119, 105, 100, 116, 104,
      98,  48,  10,  6,   111, 117, 116, 112, 117, 116, 18,  38,  10,  36,  8,
      1,   18,  32,  10,  7,   18,  5,   98,  97,  116, 99,  104, 10,  2,   8,
      3,   10,  8,   18,  6,   104, 101, 105, 103, 104, 116, 10,  7,   18,  5,
      119, 105, 100, 116, 104, 66,  4,   10,  0,   16,  13};
  return ModelFromMemory(ModelFormat::kOnnx, kModel.data(), kModel.size(),
                         nullptr, "dynamic_image_identity");
}

cv::Mat MakeSolidBgrImage(int width, int height, int index) {
  const int blue = 10 + index;
  const int green = 20 + index;
  const int red = 30 + index;
  return cv::Mat(height, width, CV_8UC3, cv::Scalar(blue, green, red));
}

std::vector<cv::Mat> MakeImages(int batch) {
  std::vector<cv::Mat> images;
  images.reserve(static_cast<std::size_t>(batch));
  for (int index = 0; index < batch; ++index) {
    images.push_back(MakeSolidBgrImage(11 + index, 7 + index, index));
  }
  return images;
}

std::vector<float> ExpectedRgbBchwValues(int batch, ImageSize size) {
  std::vector<float> values;
  values.reserve(
      static_cast<std::size_t>(batch * 3 * size.width * size.height));
  const int plane_size = size.width * size.height;
  for (int index = 0; index < batch; ++index) {
    const float red = static_cast<float>(30 + index);
    const float green = static_cast<float>(20 + index);
    const float blue = static_cast<float>(10 + index);
    values.insert(values.end(), plane_size, red);
    values.insert(values.end(), plane_size, green);
    values.insert(values.end(), plane_size, blue);
  }
  return values;
}

struct PipelineCase {
  int batch = 1;
  ImageSize target_size;
};

void ExpectDynamicImagePipeline(BackendPtr& backend, int batch,
                                ImageSize target_size) {
  RawImageBatch raw_images(MakeImages(batch));
  GeometryTransformer transformer;
  GeometryResult resized =
      transformer.Resize(raw_images, target_size, Interpolation::kNearest);
  ASSERT_EQ(resized.size(), static_cast<std::size_t>(batch));
  for (int index = 0; index < batch; ++index) {
    EXPECT_EQ(resized.transformed_size(static_cast<std::size_t>(index)).width,
              target_size.width);
    EXPECT_EQ(resized.transformed_size(static_cast<std::size_t>(index)).height,
              target_size.height);
  }

  const TensorInfo& input_info = backend->model_info().inputs.front();
  Tensor input = ToTensor(resized.images(), Device{DeviceType::kCpu, 0},
                          input_info, TensorLayout::kBchw);
  const std::vector<int64_t> expected_shape = {batch, 3, target_size.height,
                                               target_size.width};
  EXPECT_EQ(input.shape(), expected_shape);

  const std::vector<float> expected_values =
      ExpectedRgbBchwValues(batch, target_size);
  EXPECT_EQ(input.CopyToHostVector<float>(), expected_values);

  std::vector<Tensor> outputs = backend->Infer(input);

  ASSERT_EQ(outputs.size(), 1U);
  EXPECT_EQ(outputs.front().name(), "output");
  EXPECT_EQ(outputs.front().shape(), expected_shape);
  EXPECT_EQ(outputs.front().CopyToHostVector<float>(), expected_values);
}

TEST(OnnxDynamicImagePipelineTest,
     ResizesBatchesAndRunsDynamicBatchHeightWidthModel) {
  BackendPtr backend = CreateBackend(DynamicImageIdentityModel());
  const ModelInfo& info = backend->model_info();

  ASSERT_EQ(info.inputs.size(), 1U);
  EXPECT_EQ(info.inputs.front().name, "input");
  EXPECT_EQ(info.inputs.front().data_type, DataType::kFloat32);
  EXPECT_EQ(info.inputs.front().shape, std::vector<int64_t>({-1, 3, -1, -1}));
  ASSERT_EQ(info.outputs.size(), 1U);
  EXPECT_EQ(info.outputs.front().shape, std::vector<int64_t>({-1, 3, -1, -1}));

  const std::vector<PipelineCase> cases = {
      PipelineCase{1, ImageSize{1, 1}}, PipelineCase{1, ImageSize{5, 7}},
      PipelineCase{2, ImageSize{9, 4}}, PipelineCase{3, ImageSize{4, 9}},
      PipelineCase{4, ImageSize{8, 8}}, PipelineCase{5, ImageSize{13, 6}},
  };
  for (const PipelineCase& test_case : cases) {
    SCOPED_TRACE("batch=" + std::to_string(test_case.batch) +
                 " size=" + std::to_string(test_case.target_size.width) + "x" +
                 std::to_string(test_case.target_size.height));
    ExpectDynamicImagePipeline(backend, test_case.batch, test_case.target_size);
  }
}

}  // namespace
}  // namespace mw::infer
