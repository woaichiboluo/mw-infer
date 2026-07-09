#include <fmt/core.h>
#include <fmt/ranges.h>

#include <CLI/CLI.hpp>
#include <algorithm>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/imgcodecs.hpp>
#include <optional>
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

constexpr float kLetterBoxFill = 114.0F;

struct CliArguments {
  std::string engine_path;
  std::vector<std::string> image_paths;
  std::string device = "cuda:0";
  std::string input_size;
  int64_t max_detections = 100;
};

struct YoloInput {
  mw::infer::Tensor tensor;
  std::vector<mw::infer::GeometryTrace> traces;
};

struct Detection {
  int class_id = -1;
  float score = 0.0F;
  mw::infer::Rect2f box;
};

std::string ShapeString(const std::vector<int64_t>& shape) {
  return fmt::format("[{}]", fmt::join(shape, ", "));
}

bool HasDynamicShape(const std::vector<int64_t>& shape) {
  return std::any_of(shape.begin(), shape.end(),
                     [](int64_t dim) { return dim <= 0; });
}

std::optional<cv::Size> ParseInputSize(const std::string& value) {
  if (value.empty()) {
    return std::nullopt;
  }

  const std::size_t separator = value.find_first_of("xX");
  if (separator == std::string::npos || separator == 0 ||
      separator + 1 >= value.size()) {
    throw std::invalid_argument("YOLO input size must use WIDTHxHEIGHT format");
  }

  std::size_t width_end = 0;
  std::size_t height_end = 0;
  const int width = std::stoi(value.substr(0, separator), &width_end);
  const int height = std::stoi(value.substr(separator + 1), &height_end);
  if (width_end != separator || height_end != value.size() - separator - 1 ||
      width <= 0 || height <= 0) {
    throw std::invalid_argument(
        "YOLO input size must contain positive WIDTHxHEIGHT values");
  }
  return cv::Size{width, height};
}

const mw::infer::TensorShapeRange* FindProfileInput(
    const mw::infer::ModelInfo& model_info, const std::string& input_name) {
  if (model_info.profiles.empty()) {
    return nullptr;
  }
  for (const mw::infer::TensorShapeRange& range :
       model_info.profiles.front().inputs) {
    if (range.name == input_name) {
      return &range;
    }
  }
  return nullptr;
}

bool ShapeInRange(const std::vector<int64_t>& shape,
                  const mw::infer::TensorShapeRange& range) {
  if (shape.size() != range.min_shape.size() ||
      shape.size() != range.max_shape.size()) {
    return false;
  }
  for (std::size_t axis = 0; axis < shape.size(); ++axis) {
    if (shape[axis] < range.min_shape[axis] ||
        shape[axis] > range.max_shape[axis]) {
      return false;
    }
  }
  return true;
}

std::vector<int64_t> ResolveYoloInputShape(
    const mw::infer::ModelInfo& model_info,
    const mw::infer::TensorInfo& input_info, std::size_t batch_size,
    const std::optional<cv::Size>& requested_size) {
  if (input_info.shape.size() != 4) {
    throw std::invalid_argument(
        "YOLO TensorRT NMS example expects rank-4 input");
  }
  if (batch_size == 0) {
    throw std::invalid_argument("YOLO TensorRT NMS example requires images");
  }
  if (batch_size >
      static_cast<std::size_t>(std::numeric_limits<int64_t>::max())) {
    throw std::invalid_argument("YOLO TensorRT NMS example batch is too large");
  }
  const int64_t requested_batch = static_cast<int64_t>(batch_size);

  std::vector<int64_t> shape = input_info.shape;
  const mw::infer::TensorShapeRange* range =
      FindProfileInput(model_info, input_info.name);
  if (HasDynamicShape(shape)) {
    if (range == nullptr || range->opt_shape.size() != shape.size()) {
      throw std::invalid_argument(
          "Dynamic TensorRT input requires profile0 input ranges");
    }
    for (std::size_t axis = 0; axis < shape.size(); ++axis) {
      if (shape[axis] <= 0) {
        shape[axis] = range->opt_shape[axis];
      }
    }
    shape[0] = requested_batch;
  } else if (shape[0] != requested_batch) {
    throw std::invalid_argument(
        "Static TensorRT YOLO NMS engine batch does not match image count");
  }

  if (requested_size.has_value()) {
    if (input_info.shape[2] > 0 &&
        input_info.shape[2] != requested_size->height) {
      throw std::invalid_argument(
          "Static TensorRT YOLO NMS engine height does not match requested "
          "size");
    }
    if (input_info.shape[3] > 0 &&
        input_info.shape[3] != requested_size->width) {
      throw std::invalid_argument(
          "Static TensorRT YOLO NMS engine width does not match requested "
          "size");
    }
    shape[2] = requested_size->height;
    shape[3] = requested_size->width;
  }

  if (range != nullptr && !ShapeInRange(shape, *range)) {
    throw std::invalid_argument(
        "TensorRT profile0 does not allow requested YOLO NMS input shape");
  }
  if (shape[1] != 3 || shape[2] <= 0 || shape[3] <= 0) {
    throw std::invalid_argument(
        "YOLO TensorRT NMS example expects input shape [B, 3, H, W]");
  }
  return shape;
}

cv::Mat LoadImage(const std::filesystem::path& path) {
  cv::Mat image = cv::imread(path.string(), cv::IMREAD_COLOR);
  if (image.empty()) {
    throw std::invalid_argument("Failed to read image: " + path.string());
  }
  return image;
}

YoloInput MakeInput(std::vector<cv::Mat> images, mw::infer::Device device,
                    const mw::infer::TensorInfo& input_info,
                    std::vector<int64_t> input_shape) {
  const mw::infer::ImageSize input_size{static_cast<int>(input_shape[3]),
                                        static_cast<int>(input_shape[2])};
  mw::infer::RawImageBatch raw_images(std::move(images));

  mw::infer::GeometryTransformer transformer;
  mw::infer::GeometryResult letterboxed = transformer.LetterBox(
      std::move(raw_images), input_size, mw::infer::Interpolation::kLinear,
      mw::infer::FillValue{{kLetterBoxFill, kLetterBoxFill, kLetterBoxFill}});

  mw::infer::TensorInfo resolved_input = input_info;
  resolved_input.shape = std::move(input_shape);

  mw::infer::Tensor tensor =
      mw::infer::ToTensor(letterboxed.images(), device, resolved_input,
                          mw::infer::TensorLayout::kBchw);
  tensor = mw::infer::Normalize(tensor, {0.0F, 0.0F, 0.0F}, {1.0F, 1.0F, 1.0F},
                                1.0F / 255.0F, mw::infer::TensorLayout::kBchw);
  return YoloInput{std::move(tensor), letterboxed.traces()};
}

std::vector<std::vector<Detection>> DecodeUltralyticsNmsOutput(
    const mw::infer::Tensor& output,
    const std::vector<mw::infer::GeometryTrace>& traces,
    int64_t max_detections) {
  if (output.data_type() != mw::infer::DataType::kFloat32) {
    throw std::invalid_argument(
        "YOLO TensorRT NMS example expects float32 output");
  }
  if (output.shape().size() != 3 || output.shape()[2] < 6) {
    throw std::invalid_argument(
        "YOLO TensorRT NMS example expects output shape [B, M, 6]");
  }
  if (output.shape()[0] != static_cast<int64_t>(traces.size())) {
    throw std::invalid_argument(
        "YOLO TensorRT NMS output batch does not match input batch");
  }

  const std::vector<float> values = output.CopyToHostVector<float>();
  const int64_t batch_count = output.shape()[0];
  const int64_t detection_capacity = output.shape()[1];
  const int64_t stride = output.shape()[2];

  std::vector<std::vector<Detection>> detections(traces.size());
  for (int64_t batch = 0; batch < batch_count; ++batch) {
    std::vector<Detection>& batch_detections =
        detections[static_cast<std::size_t>(batch)];
    for (int64_t index = 0; index < detection_capacity; ++index) {
      if (batch_detections.size() >= static_cast<std::size_t>(max_detections)) {
        break;
      }

      const std::size_t offset = static_cast<std::size_t>(
          (batch * detection_capacity + index) * stride);
      const float x1 = values[offset + 0];
      const float y1 = values[offset + 1];
      const float x2 = values[offset + 2];
      const float y2 = values[offset + 3];
      const float score = values[offset + 4];
      if (score <= 0.0F || x2 <= x1 || y2 <= y1) {
        continue;
      }

      const int class_id = static_cast<int>(values[offset + 5]);
      mw::infer::Rect2f box{x1, y1, x2 - x1, y2 - y1};
      batch_detections.push_back(
          Detection{class_id, score,
                    traces[static_cast<std::size_t>(batch)].RestoreRect(box)});
    }
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

void ConfigureCli(CLI::App* app, CliArguments* arguments) {
  app->description(
      "Run a YOLO TensorRT engine exported with NMS and decode final "
      "detections.");
  app->add_option("engine", arguments->engine_path,
                  "TensorRT engine path (.engine or .plan)")
      ->required()
      ->check(CLI::ExistingFile);
  app->add_option("images", arguments->image_paths, "Image paths")
      ->required()
      ->check(CLI::ExistingFile);
  app->add_option("--device", arguments->device,
                  "Execution device: cuda or cuda:<id>")
      ->capture_default_str();
  app->add_option("--input-size", arguments->input_size,
                  "Network input size WIDTHxHEIGHT; defaults to profile opt "
                  "size for dynamic engines")
      ->capture_default_str();
  app->add_option("--max-detections", arguments->max_detections,
                  "Maximum detections to print per image")
      ->capture_default_str()
      ->check(CLI::PositiveNumber);
}

}  // namespace

int main(int argc, char** argv) {
  cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);

  CliArguments arguments;
  CLI::App app;
  ConfigureCli(&app, &arguments);
  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& error) {
    return app.exit(error);
  }

  try {
    const std::filesystem::path engine_path = arguments.engine_path;
    const mw::infer::Device device(arguments.device);
    if (device.type != mw::infer::DeviceType::kCuda) {
      throw std::invalid_argument(
          "TensorRT YOLO NMS example requires CUDA device");
    }
    const std::optional<cv::Size> requested_input_size =
        ParseInputSize(arguments.input_size);

    mw::infer::Model model = mw::infer::ModelFromPath(engine_path);
    if (model.format != mw::infer::ModelFormat::kTensorRT) {
      throw std::invalid_argument(
          "TensorRT YOLO NMS example expects .engine or .plan");
    }
    mw::infer::BackendPtr backend =
        mw::infer::CreateBackend(std::move(model), device);
    const mw::infer::ModelInfo& model_info = backend->model_info();
    if (model_info.inputs.size() != 1 || model_info.outputs.size() != 1) {
      throw std::invalid_argument(
          "TensorRT YOLO NMS example expects one input and one NMS output");
    }
    const mw::infer::TensorInfo& input_info = model_info.inputs.front();
    if (input_info.data_type != mw::infer::DataType::kFloat32) {
      throw std::invalid_argument(
          "TensorRT YOLO NMS example expects float32 input");
    }

    std::vector<int64_t> input_shape = ResolveYoloInputShape(
        model_info, input_info, arguments.image_paths.size(),
        requested_input_size);
    std::vector<cv::Mat> images;
    std::vector<cv::Size> original_sizes;
    images.reserve(arguments.image_paths.size());
    original_sizes.reserve(arguments.image_paths.size());
    for (const std::string& image_path : arguments.image_paths) {
      cv::Mat image = LoadImage(image_path);
      original_sizes.push_back(image.size());
      images.push_back(std::move(image));
    }
    YoloInput input = MakeInput(std::move(images), device, input_info,
                                std::move(input_shape));

    std::vector<mw::infer::Tensor> outputs = backend->Infer(input.tensor);
    if (outputs.size() != 1) {
      throw std::runtime_error("backend returned unexpected output count");
    }

    const std::vector<std::vector<Detection>> detections =
        DecodeUltralyticsNmsOutput(outputs.front(), input.traces,
                                   arguments.max_detections);

    fmt::print("example: tensorrt_yolo_nms_infer\n");
    fmt::print("model: {}\n", backend->model().name);
    fmt::print("engine: {}\n", engine_path.string());
    fmt::print("device: {}\n", device.ToString());
    fmt::print("images: {}\n", arguments.image_paths.size());
    for (std::size_t index = 0; index < arguments.image_paths.size(); ++index) {
      fmt::print("  [{}] {} ({}x{})\n", index, arguments.image_paths[index],
                 original_sizes[index].width, original_sizes[index].height);
    }
    fmt::print("input: name=\"{}\" shape={} device={}\n", input.tensor.name(),
               ShapeString(input.tensor.shape()),
               input.tensor.device().ToString());
    fmt::print("output: name=\"{}\" shape={} device={}\n",
               outputs.front().name(), ShapeString(outputs.front().shape()),
               outputs.front().device().ToString());
    for (std::size_t index = 0; index < detections.size(); ++index) {
      fmt::print("image[{}]: {}\n", index, arguments.image_paths[index]);
      PrintDetections(detections[index]);
    }
  } catch (const std::exception& error) {
    fmt::print(stderr, "tensorrt_yolo_nms_infer failed: {}\n", error.what());
    return 1;
  }

  return 0;
}
