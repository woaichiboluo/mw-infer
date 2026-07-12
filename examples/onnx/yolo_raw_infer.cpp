#include <fmt/core.h>

#include <cstdint>
#include <exception>
#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/imgcodecs.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mw/infer/runtime/backend/backend.h"
#include "mw/infer/runtime/input/opencv_input.h"
#include "mw/infer/runtime/postprocess/nms.h"
#include "mw/infer/runtime/postprocess/yolo_decode.h"
#include "mw/infer/runtime/process/geometry.h"
#include "mw/infer/runtime/process/image_to_tensor.h"
#include "mw/infer/runtime/process/normalize.h"

namespace {

constexpr mw::infer::ImageSize kInputSize{640, 640};
constexpr float kScoreThreshold = 0.25F;
constexpr float kIouThreshold = 0.45F;
constexpr int64_t kMaxDetections = 300;

struct YoloInput {
  mw::infer::Tensor tensor;
  mw::infer::GeometryTrace trace;
};

struct Detection {
  int class_id = -1;
  float score = 0.0F;
  mw::infer::Rect2f box;
};

cv::Mat LoadImage(const std::filesystem::path& path) {
  cv::Mat image = cv::imread(path.string(), cv::IMREAD_COLOR);
  if (image.empty()) {
    throw std::invalid_argument("Failed to read image: " + path.string());
  }
  return image;
}

YoloInput MakeInput(cv::Mat image, mw::infer::Device device,
                    std::string input_name) {
  mw::infer::RawImageBatch raw_images(std::vector<cv::Mat>{std::move(image)});

  mw::infer::GeometryTransformer transformer;
  mw::infer::GeometryResult letterboxed = transformer.LetterBox(
      std::move(raw_images), kInputSize, mw::infer::Interpolation::kLinear,
      mw::infer::FillValue{{114.0, 114.0, 114.0}});

  mw::infer::TensorInfo input_info;
  input_info.name = std::move(input_name);
  input_info.data_type = mw::infer::DataType::kFloat32;
  input_info.shape = {1, 3, kInputSize.height, kInputSize.width};

  mw::infer::Tensor tensor = mw::infer::ToTensor(
      letterboxed.images(), device, input_info, mw::infer::TensorLayout::kBchw);
  tensor = mw::infer::Normalize(tensor, {0.0F, 0.0F, 0.0F}, {1.0F, 1.0F, 1.0F},
                                1.0F / 255.0F, mw::infer::TensorLayout::kBchw);
  return YoloInput{std::move(tensor), letterboxed.trace(0)};
}

std::vector<Detection> DecodeDetections(const mw::infer::Tensor& output,
                                        const mw::infer::GeometryTrace& trace) {
  mw::infer::YoloDecodeResult decoded = mw::infer::YoloDecode(output);
  mw::infer::BatchNmsOptions nms_options;
  nms_options.score_threshold = kScoreThreshold;
  nms_options.iou_threshold = kIouThreshold;
  nms_options.max_detections = kMaxDetections;
  mw::infer::BatchNmsResult selected =
      mw::infer::BatchNms(decoded.boxes, decoded.scores, nms_options);

  const std::vector<int64_t> counts =
      selected.counts.CopyToHostVector<int64_t>();
  const std::vector<float> boxes = selected.boxes.CopyToHostVector<float>();
  const std::vector<float> scores = selected.scores.CopyToHostVector<float>();
  const std::vector<int64_t> class_ids =
      selected.class_ids.CopyToHostVector<int64_t>();
  if (counts.size() != 1 || counts[0] < 0 || counts[0] > kMaxDetections) {
    throw std::runtime_error("BatchNms returned an invalid detection count");
  }
  std::vector<Detection> detections;
  detections.reserve(static_cast<std::size_t>(counts[0]));
  for (int64_t detection = 0; detection < counts[0]; ++detection) {
    const std::size_t index = static_cast<std::size_t>(detection);
    const std::size_t box_offset = index * 4;
    mw::infer::Rect2f box{boxes[box_offset], boxes[box_offset + 1],
                          boxes[box_offset + 2] - boxes[box_offset],
                          boxes[box_offset + 3] - boxes[box_offset + 1]};
    detections.push_back(Detection{static_cast<int>(class_ids[index]),
                                   scores[index], trace.RestoreRect(box)});
  }
  return detections;
}

void PrintDetections(const std::vector<Detection>& detections) {
  fmt::print("detections: {}\n", detections.size());
  for (std::size_t index = 0; index < detections.size(); ++index) {
    const Detection& detection = detections[index];
    fmt::print(
        "  {:>2}: class={} score={:.6f} box=[{:.1f}, {:.1f}, {:.1f}, "
        "{:.1f}]\n",
        index + 1, detection.class_id, detection.score, detection.box.x,
        detection.box.y, detection.box.width, detection.box.height);
  }
}

}  // namespace

int main(int argc, char** argv) {
  cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);
  if (argc < 3 || argc > 4) {
    fmt::print(stderr, "usage: {} <model.onnx> <image> [cpu|cuda:<id>]\n",
               argv[0]);
    return 2;
  }

  try {
    const std::filesystem::path model_path = argv[1];
    const std::filesystem::path image_path = argv[2];
    const mw::infer::Device device(argc == 4 ? argv[3] : "cpu");

    mw::infer::Model model = mw::infer::ModelFromPath(model_path);
    mw::infer::BackendPtr backend =
        mw::infer::CreateBackend(std::move(model), device);
    if (backend->model_info().inputs.size() != 1) {
      throw std::invalid_argument(
          "raw YOLO example expects exactly one model input");
    }

    YoloInput input = MakeInput(LoadImage(image_path), device,
                                backend->model_info().inputs.front().name);
    std::vector<mw::infer::Tensor> outputs = backend->Infer(input.tensor);
    if (outputs.size() != 1) {
      throw std::invalid_argument("raw YOLO example expects one model output");
    }

    const std::vector<Detection> detections =
        DecodeDetections(outputs.front(), input.trace);

    fmt::print("example: yolo_raw_infer\n");
    fmt::print("model: {}\n", backend->model().name);
    fmt::print("device: {}\n", device.ToString());
    PrintDetections(detections);
  } catch (const std::exception& error) {
    fmt::print(stderr, "yolo_raw_infer failed: {}\n", error.what());
    return 1;
  }

  return 0;
}
