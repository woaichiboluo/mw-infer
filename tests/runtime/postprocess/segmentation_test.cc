#include "mw/infer/runtime/postprocess/segmentation.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(MW_INFER_HAS_CUDA_POSTPROCESS)
#include <cuda_runtime_api.h>
#endif

namespace mw::infer {
namespace {

std::vector<float> TestLogits() {
  return {
      0.1F, 0.9F, 0.3F, 0.2F,  //
      0.8F, 0.2F, 0.4F, 0.7F,  //
      0.2F, 0.1F, 0.6F, 0.1F,
  };
}

Tensor MakeCpuFloatTensor(std::vector<int64_t> shape,
                          const std::vector<float>& data,
                          std::string name = {}) {
  TensorDesc desc;
  desc.info.name = std::move(name);
  desc.info.data_type = DataType::kFloat32;
  desc.info.shape = std::move(shape);
  desc.device = Device{DeviceType::kCpu, 0};
  Tensor tensor = Tensor::Allocate(std::move(desc));
  if (tensor.bytes() > 0) {
    std::memcpy(tensor.data(), data.data(), tensor.bytes());
  }
  return tensor;
}

Tensor MakeCpuInt64Tensor(std::vector<int64_t> shape,
                          const std::vector<int64_t>& data,
                          std::string name = {}) {
  TensorDesc desc;
  desc.info.name = std::move(name);
  desc.info.data_type = DataType::kInt64;
  desc.info.shape = std::move(shape);
  desc.device = Device{DeviceType::kCpu, 0};
  Tensor tensor = Tensor::Allocate(std::move(desc));
  if (tensor.bytes() > 0) {
    std::memcpy(tensor.data(), data.data(), tensor.bytes());
  }
  return tensor;
}

std::vector<float> CopyFloats(const Tensor& tensor) {
  return tensor.CopyToHostVector<float>();
}

std::vector<int64_t> CopyInt64s(const Tensor& tensor) {
  return tensor.CopyToHostVector<int64_t>();
}

void ExpectNearVector(const std::vector<float>& actual,
                      const std::vector<float>& expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t index = 0; index < actual.size(); ++index) {
    EXPECT_NEAR(actual[index], expected[index], 1.0e-5F);
  }
}

float SelectedProbability(const std::vector<float>& logits) {
  const float max_logit = *std::max_element(logits.begin(), logits.end());
  float sum = 0.0F;
  for (float logit : logits) {
    sum += std::exp(logit - max_logit);
  }
  return 1.0F / sum;
}

void ExpectSegmentationResult(const SemanticSegmentationResult& result,
                              DeviceType device) {
  EXPECT_EQ(result.class_ids.device().type, device);
  EXPECT_EQ(result.scores.device().type, device);
  EXPECT_EQ(result.class_ids.name(), "segmentation_class_ids");
  EXPECT_EQ(result.scores.name(), "segmentation_scores");
  EXPECT_EQ(result.class_ids.shape(), std::vector<int64_t>({1, 2, 2}));
  EXPECT_EQ(result.scores.shape(), std::vector<int64_t>({1, 2, 2}));
  EXPECT_EQ(CopyInt64s(result.class_ids), std::vector<int64_t>({1, 0, 2, 1}));
  ExpectNearVector(CopyFloats(result.scores),
                   {
                       SelectedProbability({0.1F, 0.8F, 0.2F}),
                       SelectedProbability({0.9F, 0.2F, 0.1F}),
                       SelectedProbability({0.3F, 0.4F, 0.6F}),
                       SelectedProbability({0.2F, 0.7F, 0.1F}),
                   });
}

std::vector<GeometryTrace> ResizeTraceBatch(ImageSize before_size,
                                            ImageSize after_size) {
  GeometryTrace trace;
  trace.AddResize(before_size, after_size);
  return {trace};
}

std::vector<GeometryTrace> LetterBoxTraceBatch(ImageSize before_size,
                                               ImageSize after_size,
                                               ImageSize resized_size,
                                               Padding padding) {
  GeometryTrace trace;
  trace.AddLetterBox(before_size, after_size, resized_size, padding);
  return {trace};
}

std::vector<GeometryTrace> CropTraceBatch(ImageSize before_size,
                                          Rect crop_rect) {
  GeometryTrace trace;
  trace.AddCrop(before_size, crop_rect);
  return {trace};
}

TEST(SegmentationPostprocessTest, SelectsBestClassPerPixel) {
  Tensor logits = MakeCpuFloatTensor({1, 3, 2, 2}, TestLogits(), "logits");

  SemanticSegmentationResult result = SemanticSegmentation(logits);

  ExpectSegmentationResult(result, DeviceType::kCpu);
}

TEST(SegmentationPostprocessTest, SupportsBatchDimension) {
  Tensor logits = MakeCpuFloatTensor({2, 2, 1, 2},
                                     {
                                         0.1F,
                                         0.8F,  //
                                         0.9F,
                                         0.2F,  //
                                         0.7F,
                                         0.4F,  //
                                         0.3F,
                                         0.6F,
                                     },
                                     "logits");

  SemanticSegmentationResult result = SemanticSegmentation(logits);

  EXPECT_EQ(result.class_ids.shape(), std::vector<int64_t>({2, 1, 2}));
  EXPECT_EQ(CopyInt64s(result.class_ids), std::vector<int64_t>({1, 0, 0, 1}));
  ExpectNearVector(CopyFloats(result.scores),
                   {
                       SelectedProbability({0.1F, 0.9F}),
                       SelectedProbability({0.8F, 0.2F}),
                       SelectedProbability({0.7F, 0.3F}),
                       SelectedProbability({0.4F, 0.6F}),
                   });
}

TEST(SegmentationPostprocessTest, RestoresResizeBeforeSelectingClass) {
  Tensor logits = MakeCpuFloatTensor({1, 2, 1, 4},
                                     {
                                         0.0F,
                                         0.0F,
                                         10.0F,
                                         10.0F,
                                         10.0F,
                                         10.0F,
                                         0.0F,
                                         0.0F,
                                     },
                                     "logits");

  SemanticSegmentationResult result = SemanticSegmentation(
      logits, ResizeTraceBatch(ImageSize{2, 1}, ImageSize{4, 1}));

  EXPECT_EQ(result.class_ids.shape(), std::vector<int64_t>({1, 1, 2}));
  EXPECT_EQ(CopyInt64s(result.class_ids), std::vector<int64_t>({1, 0}));
  ExpectNearVector(CopyFloats(result.scores),
                   {
                       SelectedProbability({0.0F, 10.0F}),
                       SelectedProbability({10.0F, 0.0F}),
                   });
}

TEST(SegmentationPostprocessTest, RestoresLogitsToOriginalShape) {
  Tensor logits = MakeCpuFloatTensor({1, 2, 1, 2},
                                     {
                                         0.0F,
                                         10.0F,
                                         10.0F,
                                         0.0F,
                                     },
                                     "logits");

  Tensor restored = RestoreSegmentationLogits(
      logits, ResizeTraceBatch(ImageSize{4, 1}, ImageSize{2, 1}));

  EXPECT_EQ(restored.shape(), std::vector<int64_t>({1, 2, 1, 4}));
  ExpectNearVector(CopyFloats(restored), {
                                             0.0F,
                                             2.5F,
                                             7.5F,
                                             10.0F,
                                             10.0F,
                                             7.5F,
                                             2.5F,
                                             0.0F,
                                         });
}

TEST(SegmentationPostprocessTest, RestoresLetterBoxBeforeSelectingClass) {
  std::vector<float> logits(2 * 4 * 4, 0.0F);
  for (int index = 0; index < 16; ++index) {
    logits[index] = 10.0F;
  }
  for (int y = 0; y < 4; ++y) {
    logits[16 + y * 4 + 1] = 20.0F;
  }
  Tensor tensor = MakeCpuFloatTensor({1, 2, 4, 4}, logits, "logits");

  SemanticSegmentationResult result = SemanticSegmentation(
      tensor, LetterBoxTraceBatch(ImageSize{2, 4}, ImageSize{4, 4},
                                  ImageSize{2, 4}, Padding{1, 0, 1, 0}));

  EXPECT_EQ(result.class_ids.shape(), std::vector<int64_t>({1, 4, 2}));
  EXPECT_EQ(CopyInt64s(result.class_ids),
            std::vector<int64_t>({1, 0, 1, 0, 1, 0, 1, 0}));
}

TEST(SegmentationPostprocessTest, RestoredInvalidPixelsHaveZeroScore) {
  Tensor logits = MakeCpuFloatTensor({1, 2, 1, 2},
                                     {
                                         0.0F,
                                         10.0F,
                                         10.0F,
                                         0.0F,
                                     },
                                     "logits");

  SemanticSegmentationResult result = SemanticSegmentation(
      logits, CropTraceBatch(ImageSize{4, 1}, Rect{1, 0, 2, 1}));

  EXPECT_EQ(result.class_ids.shape(), std::vector<int64_t>({1, 1, 4}));
  EXPECT_EQ(CopyInt64s(result.class_ids), std::vector<int64_t>({0, 1, 0, 0}));
  ExpectNearVector(CopyFloats(result.scores),
                   {
                       0.0F,
                       SelectedProbability({0.0F, 10.0F}),
                       SelectedProbability({10.0F, 0.0F}),
                       0.0F,
                   });
}

TEST(SegmentationPostprocessTest, SupportsSingleChannelBinaryLogits) {
  Tensor logits =
      MakeCpuFloatTensor({1, 1, 1, 3}, {-2.0F, 0.0F, 2.0F}, "binary_logits");

  SemanticSegmentationResult result = SemanticSegmentation(logits);

  EXPECT_EQ(result.class_ids.shape(), std::vector<int64_t>({1, 1, 3}));
  EXPECT_EQ(CopyInt64s(result.class_ids), std::vector<int64_t>({0, 1, 1}));
  const float low_probability = 1.0F / (1.0F + std::exp(2.0F));
  const float high_probability = 1.0F / (1.0F + std::exp(-2.0F));
  ExpectNearVector(CopyFloats(result.scores),
                   {1.0F - low_probability, 0.5F, high_probability});
}

TEST(SegmentationPostprocessTest, RejectsInvalidInputs) {
  Tensor logits = MakeCpuFloatTensor({1, 3, 2, 2}, TestLogits(), "logits");
  Tensor bad_rank = MakeCpuFloatTensor({3, 2, 2}, TestLogits(), "bad_rank");
  Tensor bad_shape = MakeCpuFloatTensor({1, 0, 2, 2}, {}, "bad_shape");
  Tensor bad_type =
      MakeCpuInt64Tensor({1, 3, 2, 2}, std::vector<int64_t>(12, 0), "bad_type");
  SemanticSegmentationOptions bad_options;
  bad_options.binary_threshold = 1.1F;

  EXPECT_THROW(static_cast<void>(SemanticSegmentation(Tensor{})),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(SemanticSegmentation(bad_rank)),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(SemanticSegmentation(bad_shape)),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(SemanticSegmentation(bad_type)),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(SemanticSegmentation(logits, bad_options)),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(RestoreSegmentationLogits(logits, {})),
               std::invalid_argument);
  EXPECT_NO_THROW(static_cast<void>(SemanticSegmentation(logits)));
}

TEST(SegmentationPostprocessTest,
     RejectsRestoreBatchWithDifferentOriginalSizes) {
  Tensor logits = MakeCpuFloatTensor({2, 2, 1, 1},
                                     {
                                         0.0F,
                                         1.0F,
                                         1.0F,
                                         0.0F,
                                     },
                                     "logits");
  GeometryTrace first;
  first.AddResize(ImageSize{2, 1}, ImageSize{1, 1});
  GeometryTrace second;
  second.AddResize(ImageSize{3, 1}, ImageSize{1, 1});

  EXPECT_THROW(static_cast<void>(RestoreSegmentationLogits(
                   logits, std::vector<GeometryTrace>{first, second})),
               std::invalid_argument);
}

#if defined(MW_INFER_HAS_CUDA_POSTPROCESS)

bool HasUsableCudaDevice() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

Tensor MakeCudaFloatTensor(std::vector<int64_t> shape,
                           const std::vector<float>& data,
                           std::string name = {}) {
  TensorDesc desc;
  desc.info.name = std::move(name);
  desc.info.data_type = DataType::kFloat32;
  desc.info.shape = std::move(shape);
  desc.device = Device{DeviceType::kCuda, 0};
  Tensor tensor = Tensor::Allocate(std::move(desc));
  if (tensor.bytes() > 0) {
    EXPECT_EQ(cudaMemcpy(tensor.data(), data.data(), tensor.bytes(),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
  }
  return tensor;
}

TEST(SegmentationPostprocessTest, DispatchesCudaTensor) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "CUDA postprocess is unavailable";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  Tensor logits = MakeCudaFloatTensor({1, 3, 2, 2}, TestLogits(), "logits");

  SemanticSegmentationResult result = SemanticSegmentation(logits);

  ExpectSegmentationResult(result, DeviceType::kCuda);
}

TEST(SegmentationPostprocessTest, RestoresCudaTensorBeforeSelectingClass) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "CUDA postprocess is unavailable";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  Tensor logits = MakeCudaFloatTensor({1, 2, 1, 4},
                                      {
                                          0.0F,
                                          0.0F,
                                          10.0F,
                                          10.0F,
                                          10.0F,
                                          10.0F,
                                          0.0F,
                                          0.0F,
                                      },
                                      "logits");

  SemanticSegmentationResult result = SemanticSegmentation(
      logits, ResizeTraceBatch(ImageSize{2, 1}, ImageSize{4, 1}));

  EXPECT_EQ(result.class_ids.device().type, DeviceType::kCuda);
  EXPECT_EQ(result.class_ids.shape(), std::vector<int64_t>({1, 1, 2}));
  EXPECT_EQ(CopyInt64s(result.class_ids), std::vector<int64_t>({1, 0}));
  ExpectNearVector(CopyFloats(result.scores),
                   {
                       SelectedProbability({0.0F, 10.0F}),
                       SelectedProbability({10.0F, 0.0F}),
                   });
}

#endif

}  // namespace
}  // namespace mw::infer
