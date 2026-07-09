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
#include "mw/infer/runtime/process/geometry.h"
#include "mw/infer/runtime/process/image_to_tensor.h"
#include "mw/infer/runtime/process/normalize.h"

namespace {

constexpr mw::infer::ImageSize kInputSize{640, 640};

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

Detection ReadDetection(const std::vector<float>& predictions,
                        int64_t candidate_count, int class_id, int64_t index) {
  const float center_x =
      predictions[static_cast<std::size_t>(0 * candidate_count + index)];
  const float center_y =
      predictions[static_cast<std::size_t>(1 * candidate_count + index)];
  const float width =
      predictions[static_cast<std::size_t>(2 * candidate_count + index)];
  const float height =
      predictions[static_cast<std::size_t>(3 * candidate_count + index)];
  const float score = predictions[static_cast<std::size_t>(
      (4 + class_id) * candidate_count + index)];
  return Detection{class_id, score,
                   mw::infer::Rect2f{center_x - width * 0.5F,
                                     center_y - height * 0.5F, width, height}};
}

std::vector<Detection> DecodeDetections(
    const std::vector<mw::infer::Tensor>& outputs,
    const mw::infer::GeometryTrace& trace) {
  const mw::infer::Tensor& predictions_tensor = outputs[0];
  const mw::infer::Tensor& selected_indices_tensor = outputs[1];

  const std::vector<float> predictions =
      predictions_tensor.CopyToHostVector<float>();
  const std::vector<int64_t> selected_indices =
      selected_indices_tensor.CopyToHostVector<int64_t>();
  const int64_t candidate_count = predictions_tensor.shape()[2];
  std::vector<Detection> detections;
  for (std::size_t row = 0; row < selected_indices.size() / 3; ++row) {
    const int64_t class_id = selected_indices[row * 3 + 1];
    const int64_t box_index = selected_indices[row * 3 + 2];

    Detection detection = ReadDetection(predictions, candidate_count,
                                        static_cast<int>(class_id), box_index);
    detection.box = trace.RestoreRect(detection.box);
    detections.push_back(detection);
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
          "YOLO NMS example expects exactly one model input");
    }

    YoloInput input = MakeInput(LoadImage(image_path), device,
                                backend->model_info().inputs.front().name);
    std::vector<mw::infer::Tensor> outputs = backend->Infer(input.tensor);
    if (outputs.size() != 2) {
      throw std::invalid_argument("YOLO NMS example expects two model outputs");
    }

    const std::vector<Detection> detections =
        DecodeDetections(outputs, input.trace);

    fmt::print("example: yolo_nms_infer\n");
    fmt::print("model: {}\n", backend->model().name);
    fmt::print("device: {}\n", device.ToString());
    PrintDetections(detections);
  } catch (const std::exception& error) {
    fmt::print(stderr, "yolo_nms_infer failed: {}\n", error.what());
    return 1;
  }

  return 0;
}
