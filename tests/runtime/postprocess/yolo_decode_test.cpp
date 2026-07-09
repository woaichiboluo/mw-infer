#include "mw/infer/runtime/postprocess/yolo_decode.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mw/infer/runtime/postprocess/nms.h"

#if defined(MW_INFER_HAS_CUDA_POSTPROCESS)
#include <cuda_runtime_api.h>
#endif

namespace mw::infer {
namespace {

std::vector<float> YoloV8ChannelFirstPredictions() {
  return {
      5.0F,  6.0F,  20.0F, 30.0F,  //
      5.0F,  6.0F,  20.0F, 30.0F,  //
      10.0F, 10.0F, 8.0F,  4.0F,   //
      10.0F, 10.0F, 8.0F,  4.0F,   //
      0.9F,  0.8F,  0.1F,  0.2F,   //
      0.1F,  0.2F,  0.95F, 0.24F,
  };
}

std::vector<float> YoloV8CandidateFirstPredictions() {
  return {
      5.0F,  5.0F,  10.0F, 10.0F, 0.9F, 0.1F,   //
      6.0F,  6.0F,  10.0F, 10.0F, 0.8F, 0.2F,   //
      20.0F, 20.0F, 8.0F,  8.0F,  0.1F, 0.95F,  //
      30.0F, 30.0F, 4.0F,  4.0F,  0.2F, 0.24F,
  };
}

std::vector<float> YoloV5CandidateFirstPredictions() {
  return {
      5.0F,  5.0F,  10.0F, 10.0F, 0.5F, 0.8F, 0.1F,  //
      6.0F,  6.0F,  10.0F, 10.0F, 0.9F, 0.3F, 0.6F,  //
      20.0F, 20.0F, 8.0F,  8.0F,  0.2F, 0.9F, 0.1F,
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

std::vector<float> CopyFloats(const Tensor& tensor) {
  return tensor.CopyToHostVector<float>();
}

std::vector<int64_t> CopyInt64s(const Tensor& tensor) {
  return tensor.CopyToHostVector<int64_t>();
}

YoloDecodeOptions MakeOptions(YoloVersion version, float threshold = 0.25F) {
  YoloDecodeOptions options;
  options.version = version;
  options.score_threshold = threshold;
  options.class_offset = 100.0F;
  return options;
}

void ExpectYoloV8DecodedResult(const YoloDecodeResult& result,
                               DeviceType device) {
  EXPECT_EQ(result.boxes.device().type, device);
  EXPECT_EQ(result.nms_boxes.device().type, device);
  EXPECT_EQ(result.scores.device().type, device);
  EXPECT_EQ(result.class_ids.device().type, device);
  EXPECT_EQ(result.boxes.name(), "yolo_boxes");
  EXPECT_EQ(result.nms_boxes.name(), "yolo_nms_boxes");
  EXPECT_EQ(result.scores.name(), "yolo_scores");
  EXPECT_EQ(result.class_ids.name(), "yolo_class_ids");
  EXPECT_EQ(result.boxes.shape(), std::vector<int64_t>({3, 4}));
  EXPECT_EQ(result.nms_boxes.shape(), std::vector<int64_t>({3, 4}));
  EXPECT_EQ(result.scores.shape(), std::vector<int64_t>({3}));
  EXPECT_EQ(result.class_ids.shape(), std::vector<int64_t>({3}));
  EXPECT_EQ(CopyFloats(result.boxes),
            std::vector<float>({0.0F, 0.0F, 10.0F, 10.0F, 1.0F, 1.0F, 11.0F,
                                11.0F, 16.0F, 16.0F, 24.0F, 24.0F}));
  EXPECT_EQ(CopyFloats(result.nms_boxes),
            std::vector<float>({0.0F, 0.0F, 10.0F, 10.0F, 1.0F, 1.0F, 11.0F,
                                11.0F, 116.0F, 116.0F, 124.0F, 124.0F}));
  EXPECT_EQ(CopyFloats(result.scores), std::vector<float>({0.9F, 0.8F, 0.95F}));
  EXPECT_EQ(CopyInt64s(result.class_ids), std::vector<int64_t>({0, 0, 1}));
}

TEST(YoloDecodeTest, DecodesYoloV8ChannelFirstPredictionsForNms) {
  Tensor predictions = MakeCpuFloatTensor(
      {1, 6, 4}, YoloV8ChannelFirstPredictions(), "predictions");

  YoloDecodeResult result =
      YoloDecode(predictions, MakeOptions(YoloVersion::kYoloV8));

  ExpectYoloV8DecodedResult(result, DeviceType::kCpu);
  Tensor keep = Nms(result.nms_boxes, result.scores, 0.5F);
  EXPECT_EQ(CopyInt64s(keep), std::vector<int64_t>({2, 0}));
}

TEST(YoloDecodeTest, DecodesYoloV11LikeYoloV8) {
  Tensor predictions = MakeCpuFloatTensor(
      {1, 4, 6}, YoloV8CandidateFirstPredictions(), "predictions");

  YoloDecodeResult result =
      YoloDecode(predictions, MakeOptions(YoloVersion::kYoloV11));

  ExpectYoloV8DecodedResult(result, DeviceType::kCpu);
}

TEST(YoloDecodeTest, DecodesYoloV5ObjectnessScores) {
  Tensor predictions = MakeCpuFloatTensor(
      {1, 3, 7}, YoloV5CandidateFirstPredictions(), "predictions");

  YoloDecodeResult result =
      YoloDecode(predictions, MakeOptions(YoloVersion::kYoloV5));

  EXPECT_EQ(result.boxes.shape(), std::vector<int64_t>({2, 4}));
  EXPECT_EQ(
      CopyFloats(result.boxes),
      std::vector<float>({0.0F, 0.0F, 10.0F, 10.0F, 1.0F, 1.0F, 11.0F, 11.0F}));
  EXPECT_EQ(CopyFloats(result.nms_boxes),
            std::vector<float>(
                {0.0F, 0.0F, 10.0F, 10.0F, 101.0F, 101.0F, 111.0F, 111.0F}));
  EXPECT_EQ(CopyFloats(result.scores), std::vector<float>({0.4F, 0.54F}));
  EXPECT_EQ(CopyInt64s(result.class_ids), std::vector<int64_t>({0, 1}));
}

TEST(YoloDecodeTest, AllowsEmptyDecodedResults) {
  Tensor predictions = MakeCpuFloatTensor(
      {1, 6, 4}, YoloV8ChannelFirstPredictions(), "predictions");

  YoloDecodeResult result =
      YoloDecode(predictions, MakeOptions(YoloVersion::kYoloV8, 1.0F));

  EXPECT_EQ(result.boxes.shape(), std::vector<int64_t>({0, 4}));
  EXPECT_EQ(result.nms_boxes.shape(), std::vector<int64_t>({0, 4}));
  EXPECT_EQ(result.scores.shape(), std::vector<int64_t>({0}));
  EXPECT_EQ(result.class_ids.shape(), std::vector<int64_t>({0}));
}

TEST(YoloDecodeTest, RejectsInvalidInputs) {
  Tensor predictions = MakeCpuFloatTensor(
      {1, 6, 4}, YoloV8ChannelFirstPredictions(), "predictions");
  Tensor bad_rank = MakeCpuFloatTensor({6, 4}, YoloV8ChannelFirstPredictions());
  Tensor bad_batch =
      MakeCpuFloatTensor({2, 6, 4}, std::vector<float>(48, 0.0F));
  Tensor bad_yolo_v5_channels =
      MakeCpuFloatTensor({1, 5, 4}, std::vector<float>(20, 0.0F));

  EXPECT_THROW(static_cast<void>(YoloDecode(Tensor{})), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(YoloDecode(bad_rank)), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(YoloDecode(bad_batch)), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(YoloDecode(bad_yolo_v5_channels,
                                            MakeOptions(YoloVersion::kYoloV5))),
               std::invalid_argument);

  YoloDecodeOptions invalid_options = MakeOptions(YoloVersion::kYoloV8);
  invalid_options.class_offset = -1.0F;
  EXPECT_THROW(static_cast<void>(YoloDecode(predictions, invalid_options)),
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

TEST(YoloDecodeTest, DecodesCudaYoloV8PredictionsForNms) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "CUDA postprocess is unavailable";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  Tensor predictions = MakeCudaFloatTensor(
      {1, 6, 4}, YoloV8ChannelFirstPredictions(), "predictions");

  YoloDecodeResult result =
      YoloDecode(predictions, MakeOptions(YoloVersion::kYoloV8));

  ExpectYoloV8DecodedResult(result, DeviceType::kCuda);
  Tensor keep = Nms(result.nms_boxes, result.scores, 0.5F);
  EXPECT_EQ(keep.device().type, DeviceType::kCuda);
  EXPECT_EQ(CopyInt64s(keep), std::vector<int64_t>({2, 0}));
}

TEST(YoloDecodeTest, DecodesCudaYoloV5Predictions) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "CUDA postprocess is unavailable";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  Tensor predictions = MakeCudaFloatTensor(
      {1, 3, 7}, YoloV5CandidateFirstPredictions(), "predictions");

  YoloDecodeResult result =
      YoloDecode(predictions, MakeOptions(YoloVersion::kYoloV5));

  EXPECT_EQ(result.boxes.device().type, DeviceType::kCuda);
  EXPECT_EQ(result.boxes.shape(), std::vector<int64_t>({2, 4}));
  EXPECT_EQ(CopyFloats(result.scores), std::vector<float>({0.4F, 0.54F}));
  EXPECT_EQ(CopyInt64s(result.class_ids), std::vector<int64_t>({0, 1}));
}

#endif

}  // namespace
}  // namespace mw::infer
