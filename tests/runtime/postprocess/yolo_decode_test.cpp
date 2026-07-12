#include "mw/infer/runtime/postprocess/yolo_decode.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mw/infer/runtime/execution_stream.h"

#if defined(MW_INFER_HAS_CUDA_POSTPROCESS)
#include <cuda_runtime_api.h>

#include <atomic>
#include <chrono>
#include <thread>
#endif

namespace mw::infer {
namespace {

std::vector<float> YoloV8ChannelFirstPredictions() {
  return {
      5.0F,  6.0F,  20.0F, 30.0F,  // cx
      5.0F,  6.0F,  20.0F, 30.0F,  // cy
      10.0F, 10.0F, 8.0F,  4.0F,   // width
      10.0F, 10.0F, 8.0F,  4.0F,   // height
      0.9F,  0.8F,  0.1F,  0.2F,   // class 0
      0.1F,  0.2F,  0.95F, 0.24F,  // class 1
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

std::vector<float> YoloV8TwoBatchChannelFirstPredictions() {
  return {
      5.0F,  6.0F,  5.0F,  6.0F,  10.0F, 10.0F,
      10.0F, 10.0F, 0.9F,  0.8F,  0.1F,  0.2F,
      5.0F,  6.0F,  5.0F,  6.0F,  10.0F, 10.0F,
      10.0F, 10.0F, 0.85F, 0.7F,  0.15F, 0.3F,
  };
}

std::vector<float> NonFiniteCandidateFirstPredictions() {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float infinity = std::numeric_limits<float>::infinity();
  const float maximum = std::numeric_limits<float>::max();
  return {
      5.0F,     5.0F, 10.0F,   10.0F, 0.9F,     0.1F,  // valid
      5.0F,     5.0F, 10.0F,   10.0F, nan,      nan,   // scores
      infinity, 5.0F, 10.0F,   10.0F, 0.8F,     0.1F,  // center
      maximum,  5.0F, maximum, 10.0F, 0.8F,     0.1F,  // xyxy
      5.0F,     5.0F, nan,     10.0F, 0.8F,     0.1F,  // size
      5.0F,     5.0F, 10.0F,   10.0F, infinity, 0.1F,  // score
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

void ExpectFloatsNear(const std::vector<float>& actual,
                      const std::vector<float>& expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t index = 0; index < actual.size(); ++index) {
    EXPECT_NEAR(actual[index], expected[index], 1.0e-6F);
  }
}

YoloDecodeOptions MakeOptions(YoloVersion version) {
  YoloDecodeOptions options;
  options.version = version;
  return options;
}

void ExpectYoloV8Result(const YoloDecodeResult& result, DeviceType device) {
  EXPECT_EQ(result.boxes.device().type, device);
  EXPECT_EQ(result.scores.device().type, device);
  EXPECT_EQ(result.boxes.name(), "yolo_boxes");
  EXPECT_EQ(result.scores.name(), "yolo_scores");
  EXPECT_EQ(result.boxes.shape(), std::vector<int64_t>({1, 4, 4}));
  EXPECT_EQ(result.scores.shape(), std::vector<int64_t>({1, 4, 2}));
  EXPECT_EQ(CopyFloats(result.boxes),
            std::vector<float>({0.0F, 0.0F, 10.0F, 10.0F,
                                1.0F, 1.0F, 11.0F, 11.0F,
                                16.0F, 16.0F, 24.0F, 24.0F,
                                28.0F, 28.0F, 32.0F, 32.0F}));
  EXPECT_EQ(CopyFloats(result.scores),
            std::vector<float>({0.9F, 0.1F, 0.8F, 0.2F,
                                0.1F, 0.95F, 0.2F, 0.24F}));
}

TEST(YoloDecodeTest, DecodesDenseYoloV8ChannelFirstPredictions) {
  Tensor predictions = MakeCpuFloatTensor(
      {1, 6, 4}, YoloV8ChannelFirstPredictions(), "predictions");

  YoloDecodeResult result =
      YoloDecode(predictions, MakeOptions(YoloVersion::kYoloV8));

  ExpectYoloV8Result(result, DeviceType::kCpu);
}

TEST(YoloDecodeTest, AcceptsMatchingCpuExecutionStream) {
  Tensor predictions = MakeCpuFloatTensor(
      {1, 6, 4}, YoloV8ChannelFirstPredictions(), "predictions");
  ExecutionStream stream(Device{DeviceType::kCpu, 0});

  YoloDecodeResult result = YoloDecode(
      predictions, MakeOptions(YoloVersion::kYoloV8),
      TensorAllocator::Default(), &stream);

  ExpectYoloV8Result(result, DeviceType::kCpu);
}

TEST(YoloDecodeTest, DecodesYoloV11AndV26LikeYoloV8) {
  Tensor predictions = MakeCpuFloatTensor(
      {1, 4, 6}, YoloV8CandidateFirstPredictions(), "predictions");

  ExpectYoloV8Result(
      YoloDecode(predictions, MakeOptions(YoloVersion::kYoloV11)),
      DeviceType::kCpu);
  ExpectYoloV8Result(
      YoloDecode(predictions, MakeOptions(YoloVersion::kYoloV26)),
      DeviceType::kCpu);
}

TEST(YoloDecodeTest, MultipliesEveryYoloV5ClassByObjectness) {
  Tensor predictions = MakeCpuFloatTensor(
      {1, 3, 7}, YoloV5CandidateFirstPredictions(), "predictions");

  YoloDecodeResult result =
      YoloDecode(predictions, MakeOptions(YoloVersion::kYoloV5));

  EXPECT_EQ(result.boxes.shape(), std::vector<int64_t>({1, 3, 4}));
  EXPECT_EQ(result.scores.shape(), std::vector<int64_t>({1, 3, 2}));
  ExpectFloatsNear(CopyFloats(result.scores),
                   {0.4F, 0.05F, 0.27F, 0.54F, 0.18F, 0.02F});
}

TEST(YoloDecodeTest, ResolvesOfficialLayoutsWhenBothAxesArePlausible) {
  constexpr int64_t kV8Candidates = 116;
  constexpr int64_t kV8Channels = 84;
  std::vector<float> v8_predictions(
      static_cast<std::size_t>(kV8Channels * kV8Candidates), 0.0F);
  YoloDecodeResult v8_result = YoloDecode(
      MakeCpuFloatTensor({1, kV8Channels, kV8Candidates}, v8_predictions),
      MakeOptions(YoloVersion::kYoloV8));
  EXPECT_EQ(v8_result.boxes.shape(), std::vector<int64_t>({1, 116, 4}));
  EXPECT_EQ(v8_result.scores.shape(), std::vector<int64_t>({1, 116, 80}));

  std::vector<float> v5_predictions(6U * 7U, 0.0F);
  YoloDecodeResult v5_result = YoloDecode(
      MakeCpuFloatTensor({1, 6, 7}, v5_predictions),
      MakeOptions(YoloVersion::kYoloV5));
  EXPECT_EQ(v5_result.boxes.shape(), std::vector<int64_t>({1, 6, 4}));
  EXPECT_EQ(v5_result.scores.shape(), std::vector<int64_t>({1, 6, 2}));
}

TEST(YoloDecodeTest, ExplicitLayoutOverridesOfficialAutoLayout) {
  constexpr int64_t kCandidates = 116;
  constexpr int64_t kChannels = 84;
  std::vector<float> predictions(
      static_cast<std::size_t>(kChannels * kCandidates), 0.0F);
  YoloDecodeOptions options = MakeOptions(YoloVersion::kYoloV8);
  options.tensor_layout = YoloTensorLayout::kCandidateFirst;

  YoloDecodeResult result = YoloDecode(
      MakeCpuFloatTensor({1, kCandidates, kChannels}, predictions), options);

  EXPECT_EQ(result.boxes.shape(), std::vector<int64_t>({1, 116, 4}));
  EXPECT_EQ(result.scores.shape(), std::vector<int64_t>({1, 116, 80}));
}

TEST(YoloDecodeTest, PreservesNonFiniteBoxesAndScoresForBatchNms) {
  YoloDecodeOptions options = MakeOptions(YoloVersion::kYoloV8);
  options.tensor_layout = YoloTensorLayout::kCandidateFirst;
  Tensor predictions = MakeCpuFloatTensor(
      {1, 6, 6}, NonFiniteCandidateFirstPredictions(), "predictions");

  YoloDecodeResult result = YoloDecode(predictions, options);
  const std::vector<float> boxes = CopyFloats(result.boxes);
  const std::vector<float> scores = CopyFloats(result.scores);

  ASSERT_EQ(boxes.size(), 24U);
  ASSERT_EQ(scores.size(), 12U);
  EXPECT_FLOAT_EQ(boxes[0], 0.0F);
  EXPECT_TRUE(std::isinf(boxes[8]));
  EXPECT_TRUE(std::isinf(boxes[14]));
  EXPECT_TRUE(std::isnan(boxes[16]));
  EXPECT_TRUE(std::isnan(scores[2]));
  EXPECT_TRUE(std::isinf(scores[10]));
}

TEST(YoloDecodeTest, PreservesBatchAndCandidateDimensions) {
  Tensor predictions = MakeCpuFloatTensor(
      {2, 6, 2}, YoloV8TwoBatchChannelFirstPredictions(), "predictions");

  YoloDecodeResult result =
      YoloDecode(predictions, MakeOptions(YoloVersion::kYoloV8));

  EXPECT_EQ(result.boxes.shape(), std::vector<int64_t>({2, 2, 4}));
  EXPECT_EQ(result.scores.shape(), std::vector<int64_t>({2, 2, 2}));
  EXPECT_EQ(CopyFloats(result.scores),
            std::vector<float>({0.9F, 0.1F, 0.8F, 0.2F,
                                0.85F, 0.15F, 0.7F, 0.3F}));
}

TEST(YoloDecodeTest, AllowsZeroBatchOrCandidates) {
  YoloDecodeResult zero_candidates = YoloDecode(
      MakeCpuFloatTensor({1, 6, 0}, {}),
      MakeOptions(YoloVersion::kYoloV8));
  EXPECT_EQ(zero_candidates.boxes.shape(),
            std::vector<int64_t>({1, 0, 4}));
  EXPECT_EQ(zero_candidates.scores.shape(),
            std::vector<int64_t>({1, 0, 2}));

  YoloDecodeResult zero_batch = YoloDecode(
      MakeCpuFloatTensor({0, 6, 4}, {}),
      MakeOptions(YoloVersion::kYoloV8));
  EXPECT_EQ(zero_batch.boxes.shape(), std::vector<int64_t>({0, 4, 4}));
  EXPECT_EQ(zero_batch.scores.shape(), std::vector<int64_t>({0, 4, 2}));
}

TEST(YoloDecodeTest, RejectsInvalidInputs) {
  Tensor bad_rank =
      MakeCpuFloatTensor({6, 4}, YoloV8ChannelFirstPredictions());
  Tensor bad_yolo_v5_channels =
      MakeCpuFloatTensor({1, 5, 4}, std::vector<float>(20, 0.0F));

  EXPECT_THROW(static_cast<void>(YoloDecode(Tensor{})), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(YoloDecode(bad_rank)), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(YoloDecode(bad_yolo_v5_channels,
                                            MakeOptions(YoloVersion::kYoloV5))),
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
  return MakeCpuFloatTensor(std::move(shape), data, std::move(name))
      .CopyTo(Device{DeviceType::kCuda, 0});
}

void CUDART_CB DelayDecodeStream(void* user_data) {
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  static_cast<std::atomic<bool>*>(user_data)->store(true,
                                                    std::memory_order_release);
}

TEST(YoloDecodeTest, DecodesCudaYoloV8Predictions) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "CUDA postprocess is unavailable";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  Tensor predictions = MakeCudaFloatTensor(
      {1, 6, 4}, YoloV8ChannelFirstPredictions(), "predictions");

  YoloDecodeResult result =
      YoloDecode(predictions, MakeOptions(YoloVersion::kYoloV8));

  ExpectYoloV8Result(result, DeviceType::kCuda);
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

  EXPECT_EQ(result.boxes.shape(), std::vector<int64_t>({1, 3, 4}));
  ExpectFloatsNear(CopyFloats(result.scores),
                   {0.4F, 0.05F, 0.27F, 0.54F, 0.18F, 0.02F});
}

TEST(YoloDecodeTest, RunsCudaDecodeAsynchronouslyOnProvidedStream) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "CUDA postprocess is unavailable";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  Tensor predictions = MakeCudaFloatTensor(
      {1, 6, 4}, YoloV8ChannelFirstPredictions(), "predictions");
  ExecutionStream stream(Device{DeviceType::kCuda, 0});
  PooledTensorAllocator allocator;
  {
    YoloDecodeResult warm = YoloDecode(
        predictions, MakeOptions(YoloVersion::kYoloV8), allocator, &stream);
    stream.Synchronize();
  }

  std::atomic<bool> delay_completed{false};
  ASSERT_EQ(cudaLaunchHostFunc(stream.cuda_handle(), DelayDecodeStream,
                               &delay_completed),
            cudaSuccess);
  YoloDecodeResult result = YoloDecode(
      predictions, MakeOptions(YoloVersion::kYoloV8), allocator, &stream);
  predictions = Tensor();
  EXPECT_FALSE(delay_completed.load(std::memory_order_acquire));
  stream.Synchronize();

  EXPECT_TRUE(delay_completed.load(std::memory_order_acquire));
  ExpectYoloV8Result(result, DeviceType::kCuda);
}

TEST(YoloDecodeTest, RejectsMismatchedCudaExecutionStreamDevice) {
  if (!HasUsableCudaDevice()) {
    GTEST_SKIP() << "CUDA postprocess is unavailable";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  Tensor predictions = MakeCudaFloatTensor(
      {1, 6, 4}, YoloV8ChannelFirstPredictions(), "predictions");
  ExecutionStream cpu_stream(Device{DeviceType::kCpu, 0});

  EXPECT_THROW(static_cast<void>(YoloDecode(
                   predictions, MakeOptions(YoloVersion::kYoloV8),
                   TensorAllocator::Default(), &cpu_stream)),
               std::invalid_argument);
}

#endif

}  // namespace
}  // namespace mw::infer
