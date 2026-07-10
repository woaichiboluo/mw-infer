#include "mw/infer/runtime/postprocess/yolo_seg_decode.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#if defined(MW_INFER_HAS_CUDA_POSTPROCESS)
#include <cuda_runtime_api.h>
#endif

namespace mw::infer {
namespace {

Tensor MakeCpuFloatTensor(std::vector<int64_t> shape,
                          const std::vector<float>& values,
                          std::string name = {}) {
  TensorDesc desc;
  desc.info.name = std::move(name);
  desc.info.data_type = DataType::kFloat32;
  desc.info.shape = std::move(shape);
  desc.device = Device{DeviceType::kCpu, 0};
  Tensor tensor = Tensor::Allocate(std::move(desc));
  EXPECT_EQ(tensor.element_count(), values.size());
  if (!values.empty()) {
    std::memcpy(tensor.data(), values.data(), tensor.bytes());
  }
  return tensor;
}

std::vector<float> RawYoloV8Predictions() {
  return {
      2.0F, 2.2F,  3.0F,   // center x
      2.0F, 2.2F,  3.0F,   // center y
      4.0F, 4.0F,  2.0F,   // width
      4.0F, 4.0F,  2.0F,   // height
      0.9F, 0.8F,  0.1F,   // class 0
      0.1F, 0.2F,  0.95F,  // class 1
      4.0F, -4.0F, 4.0F,   // mask coefficient 0
      0.0F, 0.0F,  0.0F,   // mask coefficient 1
  };
}

std::vector<float> RawYoloV5Predictions() {
  return {
      2.0F, 2.0F, 4.0F, 4.0F, 0.5F, 0.9F, 0.1F,  1.5F, 0.0F,
      2.2F, 2.2F, 4.0F, 4.0F, 0.9F, 0.8F, 0.2F,  1.5F, 0.0F,
      3.0F, 3.0F, 2.0F, 2.0F, 0.8F, 0.1F, 0.95F, 2.0F, 0.0F,
  };
}

std::vector<float> OneBatchPrototypes() {
  return {
      1.0F, 1.0F, 1.0F, 1.0F,  // channel 0
      0.0F, 0.0F, 0.0F, 0.0F,  // channel 1
  };
}

std::vector<float> SelectedTwoBatchPredictions() {
  return {
      0.0F, 0.0F, 4.0F,  4.0F, 0.9F,  1.0F, 4.0F,  0.0F, 0.5F, 0.5F, 3.5F, 3.5F,
      0.8F, 1.0F, -4.0F, 0.0F, 0.0F,  0.0F, 0.0F,  0.0F, 0.0F, 0.0F, 0.0F, 0.0F,

      2.0F, 2.0F, 4.0F,  4.0F, 0.95F, 0.0F, -4.0F, 0.0F, 0.0F, 0.0F, 4.0F, 4.0F,
      0.7F, 0.0F, -4.0F, 0.0F, 0.0F,  0.0F, 0.0F,  0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
  };
}

std::vector<float> TwoBatchPrototypes() {
  return {
      1.0F,  1.0F,  1.0F,  1.0F,   // batch 0, channel 0
      0.0F,  0.0F,  0.0F,  0.0F,   // batch 0, channel 1
      -1.0F, -1.0F, -1.0F, -1.0F,  // batch 1, channel 0
      0.0F,  0.0F,  0.0F,  0.0F,   // batch 1, channel 1
  };
}

std::vector<float> FirstPrototypeOnes(int64_t mask_count, int64_t height,
                                      int64_t width) {
  std::vector<float> values(
      static_cast<std::size_t>(mask_count * height * width), 0.0F);
  std::fill_n(values.begin(), static_cast<std::size_t>(height * width), 1.0F);
  return values;
}

void SetChannelFirst(std::vector<float>& values, int64_t candidate_count,
                     int64_t channel, int64_t candidate, float value) {
  values[static_cast<std::size_t>(channel * candidate_count + candidate)] =
      value;
}

void SetCandidateFirst(std::vector<float>& values, int64_t channel_count,
                       int64_t candidate, int64_t channel, float value) {
  values[static_cast<std::size_t>(candidate * channel_count + channel)] = value;
}

std::vector<float> LargeCoordinateRawPredictions() {
  std::vector<float> predictions(16, 0.0F);
  SetCandidateFirst(predictions, 8, 0, 0, 4101.0F);
  SetCandidateFirst(predictions, 8, 0, 1, 5.0F);
  SetCandidateFirst(predictions, 8, 0, 2, 10.0F);
  SetCandidateFirst(predictions, 8, 0, 3, 10.0F);
  SetCandidateFirst(predictions, 8, 0, 4, 0.9F);
  SetCandidateFirst(predictions, 8, 0, 6, 4.0F);
  SetCandidateFirst(predictions, 8, 1, 0, 5.0F);
  SetCandidateFirst(predictions, 8, 1, 1, 5.0F);
  SetCandidateFirst(predictions, 8, 1, 2, 10.0F);
  SetCandidateFirst(predictions, 8, 1, 3, 10.0F);
  SetCandidateFirst(predictions, 8, 1, 5, 0.8F);
  SetCandidateFirst(predictions, 8, 1, 6, 4.0F);
  return predictions;
}

std::vector<float> CustomThresholdSelectedPrediction(float coefficient) {
  return {1.0F, 1.0F, 4.0F, 4.0F, 0.9F, 0.0F, coefficient, 0.0F};
}

YoloSegDecodeOptions RawOptions(YoloVersion version) {
  YoloSegDecodeOptions options;
  options.version = version;
  options.prediction_layout = YoloSegPredictionLayout::kRaw;
  options.score_threshold = 0.25F;
  options.iou_threshold = 0.5F;
  options.mask_threshold = 0.5F;
  return options;
}

YoloSegDecodeOptions SelectedOptions() {
  YoloSegDecodeOptions options;
  options.version = YoloVersion::kYoloV26;
  options.prediction_layout = YoloSegPredictionLayout::kSelected;
  options.score_threshold = 0.25F;
  options.mask_threshold = 0.5F;
  return options;
}

std::vector<float> CopyFloats(const Tensor& tensor) {
  return tensor.CopyToHostVector<float>();
}

std::vector<int64_t> CopyInt64s(const Tensor& tensor) {
  return tensor.CopyToHostVector<int64_t>();
}

std::vector<std::uint8_t> CopyBytes(const Tensor& tensor) {
  return tensor.CopyToHostVector<std::uint8_t>();
}

std::vector<int64_t> MaskPixelCounts(const Tensor& masks) {
  const std::vector<std::uint8_t> values = CopyBytes(masks);
  const int64_t count = masks.shape()[0];
  const int64_t plane = masks.shape()[1] * masks.shape()[2];
  std::vector<int64_t> result(static_cast<std::size_t>(count), 0);
  for (int64_t mask = 0; mask < count; ++mask) {
    for (int64_t pixel = 0; pixel < plane; ++pixel) {
      result[static_cast<std::size_t>(mask)] +=
          values[static_cast<std::size_t>(mask * plane + pixel)];
    }
  }
  return result;
}

void ExpectRawV8StyleResult(const YoloSegDecodeResult& result,
                            DeviceType device) {
  EXPECT_EQ(result.boxes.device().type, device);
  EXPECT_EQ(result.scores.device().type, device);
  EXPECT_EQ(result.class_ids.device().type, device);
  EXPECT_EQ(result.batch_ids.device().type, device);
  EXPECT_EQ(result.masks.device().type, device);
  EXPECT_EQ(result.boxes.shape(), std::vector<int64_t>({2, 4}));
  EXPECT_EQ(result.scores.shape(), std::vector<int64_t>({2}));
  EXPECT_EQ(result.class_ids.shape(), std::vector<int64_t>({2}));
  EXPECT_EQ(result.batch_ids.shape(), std::vector<int64_t>({2}));
  EXPECT_EQ(result.masks.shape(), std::vector<int64_t>({2, 4, 4}));
  EXPECT_EQ(
      CopyFloats(result.boxes),
      std::vector<float>({2.0F, 2.0F, 4.0F, 4.0F, 0.0F, 0.0F, 4.0F, 4.0F}));
  EXPECT_EQ(CopyFloats(result.scores), std::vector<float>({0.95F, 0.9F}));
  EXPECT_EQ(CopyInt64s(result.class_ids), std::vector<int64_t>({1, 0}));
  EXPECT_EQ(CopyInt64s(result.batch_ids), std::vector<int64_t>({0, 0}));
  EXPECT_EQ(MaskPixelCounts(result.masks), std::vector<int64_t>({4, 16}));
}

YoloSegDecodeResult DecodeRawV8Style(YoloVersion version,
                                     TensorAllocator& allocator) {
  Tensor predictions =
      MakeCpuFloatTensor({1, 8, 3}, RawYoloV8Predictions(), "predictions");
  Tensor prototypes =
      MakeCpuFloatTensor({1, 2, 2, 2}, OneBatchPrototypes(), "prototypes");
  return YoloSegDecode(predictions, prototypes, ImageSize{4, 4},
                       RawOptions(version), allocator);
}

TEST(YoloSegDecodeTest, DecodesYoloV8RawSegmentationPredictions) {
  YoloSegDecodeResult result =
      DecodeRawV8Style(YoloVersion::kYoloV8, TensorAllocator::Default());
  ExpectRawV8StyleResult(result, DeviceType::kCpu);
}

TEST(YoloSegDecodeTest, DecodesYoloV11RawPredictions) {
  YoloSegDecodeResult result =
      DecodeRawV8Style(YoloVersion::kYoloV11, TensorAllocator::Default());
  ExpectRawV8StyleResult(result, DeviceType::kCpu);
}

TEST(YoloSegDecodeTest, DecodesYoloV26OneToManyRawPredictions) {
  YoloSegDecodeResult result =
      DecodeRawV8Style(YoloVersion::kYoloV26, TensorAllocator::Default());
  ExpectRawV8StyleResult(result, DeviceType::kCpu);
}

TEST(YoloSegDecodeTest, DecodesYoloV5ObjectnessAndMaskCoefficients) {
  Tensor predictions =
      MakeCpuFloatTensor({1, 3, 9}, RawYoloV5Predictions(), "predictions");
  Tensor prototypes =
      MakeCpuFloatTensor({1, 2, 2, 2}, OneBatchPrototypes(), "prototypes");
  YoloSegDecodeOptions options = RawOptions(YoloVersion::kYoloV5);
  options.mask_threshold = 0.8F;

  YoloSegDecodeResult result =
      YoloSegDecode(predictions, prototypes, ImageSize{4, 4}, options);

  EXPECT_EQ(result.boxes.shape(), std::vector<int64_t>({2, 4}));
  EXPECT_FLOAT_EQ(CopyFloats(result.scores)[0], 0.76F);
  EXPECT_FLOAT_EQ(CopyFloats(result.scores)[1], 0.72F);
  EXPECT_EQ(CopyInt64s(result.class_ids), std::vector<int64_t>({1, 0}));
  EXPECT_EQ(MaskPixelCounts(result.masks), std::vector<int64_t>({1, 0}));
}

TEST(YoloSegDecodeTest, ResolvesOfficialYoloV8LayoutAtSmallDynamicSize) {
  constexpr int64_t kCandidateCount = 84;
  constexpr int64_t kChannelCount = 116;
  std::vector<float> predictions(
      static_cast<std::size_t>(kCandidateCount * kChannelCount), 0.0F);
  SetChannelFirst(predictions, kCandidateCount, 0, 0, 32.0F);
  SetChannelFirst(predictions, kCandidateCount, 1, 0, 32.0F);
  SetChannelFirst(predictions, kCandidateCount, 2, 0, 32.0F);
  SetChannelFirst(predictions, kCandidateCount, 3, 0, 32.0F);
  SetChannelFirst(predictions, kCandidateCount, 4 + 16, 0, 0.9F);
  SetChannelFirst(predictions, kCandidateCount, 4 + 80, 0, 4.0F);
  Tensor prediction_tensor = MakeCpuFloatTensor(
      {1, kChannelCount, kCandidateCount}, predictions, "predictions");
  Tensor prototype_tensor = MakeCpuFloatTensor(
      {1, 32, 16, 16}, FirstPrototypeOnes(32, 16, 16), "prototypes");

  YoloSegDecodeResult result =
      YoloSegDecode(prediction_tensor, prototype_tensor, ImageSize{64, 64},
                    RawOptions(YoloVersion::kYoloV8));

  EXPECT_EQ(result.boxes.shape(), std::vector<int64_t>({1, 4}));
  EXPECT_EQ(CopyInt64s(result.class_ids), std::vector<int64_t>({16}));
  EXPECT_GT(MaskPixelCounts(result.masks)[0], 0);
}

TEST(YoloSegDecodeTest, ResolvesOfficialYoloV5LayoutAtSmallDynamicSize) {
  constexpr int64_t kCandidateCount = 63;
  constexpr int64_t kChannelCount = 117;
  std::vector<float> predictions(
      static_cast<std::size_t>(kCandidateCount * kChannelCount), 0.0F);
  SetCandidateFirst(predictions, kChannelCount, 0, 0, 16.0F);
  SetCandidateFirst(predictions, kChannelCount, 0, 1, 16.0F);
  SetCandidateFirst(predictions, kChannelCount, 0, 2, 16.0F);
  SetCandidateFirst(predictions, kChannelCount, 0, 3, 16.0F);
  SetCandidateFirst(predictions, kChannelCount, 0, 4, 0.9F);
  SetCandidateFirst(predictions, kChannelCount, 0, 5 + 16, 0.9F);
  SetCandidateFirst(predictions, kChannelCount, 0, 5 + 80, 4.0F);
  Tensor prediction_tensor = MakeCpuFloatTensor(
      {1, kCandidateCount, kChannelCount}, predictions, "predictions");
  Tensor prototype_tensor = MakeCpuFloatTensor(
      {1, 32, 8, 8}, FirstPrototypeOnes(32, 8, 8), "prototypes");

  YoloSegDecodeResult result =
      YoloSegDecode(prediction_tensor, prototype_tensor, ImageSize{32, 32},
                    RawOptions(YoloVersion::kYoloV5));

  EXPECT_EQ(result.boxes.shape(), std::vector<int64_t>({1, 4}));
  EXPECT_FLOAT_EQ(CopyFloats(result.scores)[0], 0.81F);
  EXPECT_EQ(CopyInt64s(result.class_ids), std::vector<int64_t>({16}));
  EXPECT_GT(MaskPixelCounts(result.masks)[0], 0);
}

TEST(YoloSegDecodeTest, HonorsExplicitCandidateFirstRawLayout) {
  constexpr int64_t kCandidateCount = 84;
  constexpr int64_t kChannelCount = 116;
  std::vector<float> predictions(
      static_cast<std::size_t>(kCandidateCount * kChannelCount), 0.0F);
  SetCandidateFirst(predictions, kChannelCount, 0, 0, 32.0F);
  SetCandidateFirst(predictions, kChannelCount, 0, 1, 32.0F);
  SetCandidateFirst(predictions, kChannelCount, 0, 2, 32.0F);
  SetCandidateFirst(predictions, kChannelCount, 0, 3, 32.0F);
  SetCandidateFirst(predictions, kChannelCount, 0, 4 + 16, 0.9F);
  SetCandidateFirst(predictions, kChannelCount, 0, 4 + 80, 4.0F);
  Tensor prediction_tensor = MakeCpuFloatTensor(
      {1, kCandidateCount, kChannelCount}, predictions, "predictions");
  Tensor prototype_tensor = MakeCpuFloatTensor(
      {1, 32, 16, 16}, FirstPrototypeOnes(32, 16, 16), "prototypes");
  YoloSegDecodeOptions options = RawOptions(YoloVersion::kYoloV8);
  options.tensor_layout = YoloSegTensorLayout::kCandidateFirst;

  YoloSegDecodeResult result = YoloSegDecode(
      prediction_tensor, prototype_tensor, ImageSize{64, 64}, options);

  EXPECT_EQ(result.boxes.shape(), std::vector<int64_t>({1, 4}));
  EXPECT_EQ(CopyInt64s(result.class_ids), std::vector<int64_t>({16}));
}

TEST(YoloSegDecodeTest, ResolvesAmbiguousSelectedLayoutAsCandidateFirst) {
  constexpr int64_t kSelectedChannels = 38;
  std::vector<float> predictions(
      static_cast<std::size_t>(kSelectedChannels * kSelectedChannels), 0.0F);
  SetCandidateFirst(predictions, kSelectedChannels, 0, 0, 0.0F);
  SetCandidateFirst(predictions, kSelectedChannels, 0, 1, 0.0F);
  SetCandidateFirst(predictions, kSelectedChannels, 0, 2, 4.0F);
  SetCandidateFirst(predictions, kSelectedChannels, 0, 3, 4.0F);
  SetCandidateFirst(predictions, kSelectedChannels, 0, 4, 0.9F);
  SetCandidateFirst(predictions, kSelectedChannels, 0, 5, 1.0F);
  SetCandidateFirst(predictions, kSelectedChannels, 0, 6, 4.0F);
  Tensor prediction_tensor = MakeCpuFloatTensor(
      {1, kSelectedChannels, kSelectedChannels}, predictions, "predictions");
  Tensor prototype_tensor = MakeCpuFloatTensor(
      {1, 32, 1, 1}, FirstPrototypeOnes(32, 1, 1), "prototypes");

  YoloSegDecodeResult result = YoloSegDecode(
      prediction_tensor, prototype_tensor, ImageSize{4, 4}, SelectedOptions());

  EXPECT_EQ(result.boxes.shape(), std::vector<int64_t>({1, 4}));
  EXPECT_EQ(CopyInt64s(result.class_ids), std::vector<int64_t>({1}));
}

TEST(YoloSegDecodeTest, DerivesNmsGroupingFromActualBoxCoordinates) {
  Tensor prediction_tensor = MakeCpuFloatTensor(
      {1, 2, 8}, LargeCoordinateRawPredictions(), "predictions");
  Tensor prototype_tensor = MakeCpuFloatTensor(
      {1, 2, 1, 1}, FirstPrototypeOnes(2, 1, 1), "prototypes");

  YoloSegDecodeResult result =
      YoloSegDecode(prediction_tensor, prototype_tensor, ImageSize{10, 10},
                    RawOptions(YoloVersion::kYoloV8));

  EXPECT_EQ(result.boxes.shape(), std::vector<int64_t>({2, 4}));
  EXPECT_EQ(CopyInt64s(result.class_ids), std::vector<int64_t>({0, 1}));
}

TEST(YoloSegDecodeTest, InterpolatesProbabilitiesForCustomMaskThreshold) {
  Tensor prototype_tensor =
      MakeCpuFloatTensor({1, 2, 2, 2}, OneBatchPrototypes(), "prototypes");
  YoloSegDecodeOptions options = SelectedOptions();
  options.mask_threshold = 0.25F;
  Tensor negative_prediction = MakeCpuFloatTensor(
      {1, 1, 8}, CustomThresholdSelectedPrediction(-4.0F), "predictions");
  YoloSegDecodeResult low_threshold_result = YoloSegDecode(
      negative_prediction, prototype_tensor, ImageSize{4, 4}, options);
  EXPECT_EQ(MaskPixelCounts(low_threshold_result.masks),
            std::vector<int64_t>({0}));

  options.mask_threshold = 0.8F;
  Tensor positive_prediction = MakeCpuFloatTensor(
      {1, 1, 8}, CustomThresholdSelectedPrediction(4.0F), "predictions");
  YoloSegDecodeResult high_threshold_result = YoloSegDecode(
      positive_prediction, prototype_tensor, ImageSize{4, 4}, options);
  EXPECT_EQ(MaskPixelCounts(high_threshold_result.masks),
            std::vector<int64_t>({1}));
}

TEST(YoloSegDecodeTest, DecodesYoloV26SelectedPredictionsWithoutNms) {
  Tensor predictions = MakeCpuFloatTensor(
      {2, 3, 8}, SelectedTwoBatchPredictions(), "predictions");
  Tensor prototypes =
      MakeCpuFloatTensor({2, 2, 2, 2}, TwoBatchPrototypes(), "prototypes");
  YoloSegDecodeOptions options = SelectedOptions();
  options.max_detections = 1;

  YoloSegDecodeResult result =
      YoloSegDecode(predictions, prototypes, ImageSize{4, 4}, options);

  EXPECT_EQ(result.boxes.shape(), std::vector<int64_t>({2, 4}));
  EXPECT_EQ(CopyFloats(result.scores), std::vector<float>({0.9F, 0.95F}));
  EXPECT_EQ(CopyInt64s(result.class_ids), std::vector<int64_t>({1, 0}));
  EXPECT_EQ(CopyInt64s(result.batch_ids), std::vector<int64_t>({0, 1}));
  EXPECT_EQ(MaskPixelCounts(result.masks), std::vector<int64_t>({16, 4}));
}

TEST(YoloSegDecodeTest, SelectedLayoutDoesNotRepeatNms) {
  Tensor predictions = MakeCpuFloatTensor(
      {2, 3, 8}, SelectedTwoBatchPredictions(), "predictions");
  Tensor prototypes =
      MakeCpuFloatTensor({2, 2, 2, 2}, TwoBatchPrototypes(), "prototypes");
  YoloSegDecodeOptions options = SelectedOptions();
  options.max_detections = 0;

  YoloSegDecodeResult result =
      YoloSegDecode(predictions, prototypes, ImageSize{4, 4}, options);

  EXPECT_EQ(result.boxes.shape(), std::vector<int64_t>({4, 4}));
  EXPECT_EQ(CopyInt64s(result.batch_ids), std::vector<int64_t>({0, 0, 1, 1}));
}

TEST(YoloSegDecodeTest, AppliesMaxDetectionsPerBatchAfterRawNms) {
  std::vector<float> predictions = RawYoloV8Predictions();
  const std::vector<float> second_batch_predictions = RawYoloV8Predictions();
  predictions.insert(predictions.end(), second_batch_predictions.begin(),
                     second_batch_predictions.end());
  std::vector<float> prototypes = OneBatchPrototypes();
  const std::vector<float> second_batch_prototypes = OneBatchPrototypes();
  prototypes.insert(prototypes.end(), second_batch_prototypes.begin(),
                    second_batch_prototypes.end());
  Tensor prediction_tensor =
      MakeCpuFloatTensor({2, 8, 3}, predictions, "predictions");
  Tensor prototype_tensor =
      MakeCpuFloatTensor({2, 2, 2, 2}, prototypes, "prototypes");
  YoloSegDecodeOptions options = RawOptions(YoloVersion::kYoloV8);
  options.max_detections = 1;

  YoloSegDecodeResult result = YoloSegDecode(
      prediction_tensor, prototype_tensor, ImageSize{4, 4}, options);

  EXPECT_EQ(result.scores.shape(), std::vector<int64_t>({2}));
  EXPECT_EQ(CopyInt64s(result.batch_ids), std::vector<int64_t>({0, 1}));
  EXPECT_EQ(CopyInt64s(result.class_ids), std::vector<int64_t>({1, 1}));
}

TEST(YoloSegDecodeTest, ReturnsInitializedEmptyResult) {
  std::vector<float> predictions = RawYoloV8Predictions();
  Tensor prediction_tensor =
      MakeCpuFloatTensor({1, 8, 3}, predictions, "predictions");
  Tensor prototype_tensor =
      MakeCpuFloatTensor({1, 2, 2, 2}, OneBatchPrototypes(), "prototypes");
  YoloSegDecodeOptions options = RawOptions(YoloVersion::kYoloV8);
  options.score_threshold = 1.0F;

  YoloSegDecodeResult result = YoloSegDecode(
      prediction_tensor, prototype_tensor, ImageSize{4, 4}, options);

  EXPECT_EQ(result.boxes.shape(), std::vector<int64_t>({0, 4}));
  EXPECT_EQ(result.scores.shape(), std::vector<int64_t>({0}));
  EXPECT_EQ(result.class_ids.shape(), std::vector<int64_t>({0}));
  EXPECT_EQ(result.batch_ids.shape(), std::vector<int64_t>({0}));
  EXPECT_EQ(result.masks.shape(), std::vector<int64_t>({0, 4, 4}));
}

TEST(YoloSegDecodeTest, SupportsPooledOutputAllocator) {
  PooledTensorAllocator pool;
  YoloSegDecodeResult result = DecodeRawV8Style(YoloVersion::kYoloV8, pool);
  ExpectRawV8StyleResult(result, DeviceType::kCpu);
}

TEST(YoloSegDecodeTest, RejectsInvalidInputsAndOptions) {
  Tensor predictions =
      MakeCpuFloatTensor({1, 8, 3}, RawYoloV8Predictions(), "predictions");
  Tensor prototypes =
      MakeCpuFloatTensor({1, 2, 2, 2}, OneBatchPrototypes(), "prototypes");
  Tensor bad_batch_prototypes =
      MakeCpuFloatTensor({2, 2, 2, 2}, TwoBatchPrototypes(), "prototypes");
  Tensor bad_selected = MakeCpuFloatTensor(
      {1, 3, 7}, std::vector<float>(21, 0.0F), "predictions");

  EXPECT_THROW(
      static_cast<void>(YoloSegDecode(Tensor{}, prototypes, ImageSize{4, 4})),
      std::invalid_argument);
  EXPECT_THROW(
      static_cast<void>(YoloSegDecode(predictions, Tensor{}, ImageSize{4, 4})),
      std::invalid_argument);
  EXPECT_THROW(static_cast<void>(YoloSegDecode(
                   predictions, bad_batch_prototypes, ImageSize{4, 4})),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(
                   YoloSegDecode(predictions, prototypes, ImageSize{0, 4})),
               std::invalid_argument);

  YoloSegDecodeOptions invalid_options = RawOptions(YoloVersion::kYoloV8);
  invalid_options.mask_threshold = -0.1F;
  EXPECT_THROW(static_cast<void>(YoloSegDecode(
                   predictions, prototypes, ImageSize{4, 4}, invalid_options)),
               std::invalid_argument);

  invalid_options = SelectedOptions();
  EXPECT_THROW(static_cast<void>(YoloSegDecode(
                   bad_selected, prototypes, ImageSize{4, 4}, invalid_options)),
               std::invalid_argument);
}

#if defined(MW_INFER_HAS_CUDA_POSTPROCESS)

bool HasUsableCudaDevice() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

Tensor ToCuda(const Tensor& tensor) {
  return tensor.CopyTo(Device{DeviceType::kCuda, 0});
}

void ExpectSameResult(const YoloSegDecodeResult& expected,
                      const YoloSegDecodeResult& actual) {
  EXPECT_EQ(CopyFloats(actual.boxes), CopyFloats(expected.boxes));
  EXPECT_EQ(CopyFloats(actual.scores), CopyFloats(expected.scores));
  EXPECT_EQ(CopyInt64s(actual.class_ids), CopyInt64s(expected.class_ids));
  EXPECT_EQ(CopyInt64s(actual.batch_ids), CopyInt64s(expected.batch_ids));
  EXPECT_EQ(CopyBytes(actual.masks), CopyBytes(expected.masks));
}

TEST(YoloSegDecodeTest, DecodesCudaYoloV8RawPredictions) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "CUDA postprocess is unavailable";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  Tensor cpu_predictions =
      MakeCpuFloatTensor({1, 8, 3}, RawYoloV8Predictions(), "predictions");
  Tensor cpu_prototypes =
      MakeCpuFloatTensor({1, 2, 2, 2}, OneBatchPrototypes(), "prototypes");
  YoloSegDecodeOptions options = RawOptions(YoloVersion::kYoloV8);
  YoloSegDecodeResult expected =
      YoloSegDecode(cpu_predictions, cpu_prototypes, ImageSize{4, 4}, options);

  YoloSegDecodeResult actual =
      YoloSegDecode(ToCuda(cpu_predictions), ToCuda(cpu_prototypes),
                    ImageSize{4, 4}, options);

  EXPECT_EQ(actual.masks.device().type, DeviceType::kCuda);
  ExpectSameResult(expected, actual);
}

TEST(YoloSegDecodeTest, DecodesCudaYoloV5ObjectnessAndMaskCoefficients) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "CUDA postprocess is unavailable";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  Tensor cpu_predictions =
      MakeCpuFloatTensor({1, 3, 9}, RawYoloV5Predictions(), "predictions");
  Tensor cpu_prototypes =
      MakeCpuFloatTensor({1, 2, 2, 2}, OneBatchPrototypes(), "prototypes");
  YoloSegDecodeOptions options = RawOptions(YoloVersion::kYoloV5);
  options.mask_threshold = 0.8F;
  YoloSegDecodeResult expected =
      YoloSegDecode(cpu_predictions, cpu_prototypes, ImageSize{4, 4}, options);

  YoloSegDecodeResult actual =
      YoloSegDecode(ToCuda(cpu_predictions), ToCuda(cpu_prototypes),
                    ImageSize{4, 4}, options);

  EXPECT_EQ(actual.masks.device().type, DeviceType::kCuda);
  ExpectSameResult(expected, actual);
}

TEST(YoloSegDecodeTest, DerivesCudaNmsGroupingFromActualBoxCoordinates) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "CUDA postprocess is unavailable";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  Tensor cpu_predictions = MakeCpuFloatTensor(
      {1, 2, 8}, LargeCoordinateRawPredictions(), "predictions");
  Tensor cpu_prototypes = MakeCpuFloatTensor(
      {1, 2, 1, 1}, FirstPrototypeOnes(2, 1, 1), "prototypes");
  YoloSegDecodeOptions options = RawOptions(YoloVersion::kYoloV8);
  YoloSegDecodeResult expected = YoloSegDecode(cpu_predictions, cpu_prototypes,
                                               ImageSize{10, 10}, options);

  YoloSegDecodeResult actual =
      YoloSegDecode(ToCuda(cpu_predictions), ToCuda(cpu_prototypes),
                    ImageSize{10, 10}, options);

  ExpectSameResult(expected, actual);
}

TEST(YoloSegDecodeTest, InterpolatesCudaProbabilitiesForCustomMaskThreshold) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "CUDA postprocess is unavailable";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  Tensor cpu_prototypes =
      MakeCpuFloatTensor({1, 2, 2, 2}, OneBatchPrototypes(), "prototypes");
  for (const auto [coefficient, threshold] :
       std::vector<std::pair<float, float>>({{-4.0F, 0.25F}, {4.0F, 0.8F}})) {
    SCOPED_TRACE(threshold);
    Tensor cpu_predictions = MakeCpuFloatTensor(
        {1, 1, 8}, CustomThresholdSelectedPrediction(coefficient),
        "predictions");
    YoloSegDecodeOptions options = SelectedOptions();
    options.mask_threshold = threshold;
    YoloSegDecodeResult expected = YoloSegDecode(
        cpu_predictions, cpu_prototypes, ImageSize{4, 4}, options);

    YoloSegDecodeResult actual =
        YoloSegDecode(ToCuda(cpu_predictions), ToCuda(cpu_prototypes),
                      ImageSize{4, 4}, options);

    ExpectSameResult(expected, actual);
  }
}

TEST(YoloSegDecodeTest, DecodesCudaYoloV26SelectedPredictions) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "CUDA postprocess is unavailable";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  Tensor cpu_predictions = MakeCpuFloatTensor(
      {2, 3, 8}, SelectedTwoBatchPredictions(), "predictions");
  Tensor cpu_prototypes =
      MakeCpuFloatTensor({2, 2, 2, 2}, TwoBatchPrototypes(), "prototypes");
  YoloSegDecodeOptions options = SelectedOptions();
  options.max_detections = 1;
  YoloSegDecodeResult expected =
      YoloSegDecode(cpu_predictions, cpu_prototypes, ImageSize{4, 4}, options);

  YoloSegDecodeResult actual =
      YoloSegDecode(ToCuda(cpu_predictions), ToCuda(cpu_prototypes),
                    ImageSize{4, 4}, options);

  EXPECT_EQ(actual.masks.device().type, DeviceType::kCuda);
  ExpectSameResult(expected, actual);
}

#endif

}  // namespace
}  // namespace mw::infer
