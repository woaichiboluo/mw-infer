#include <fmt/core.h>
#include <fmt/ranges.h>

#include <CLI/CLI.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "mw/infer/runtime/backend/backend.h"
#include "mw/infer/runtime/input/opencv_input.h"
#include "mw/infer/runtime/postprocess/softmax.h"
#include "mw/infer/runtime/postprocess/topk.h"
#include "mw/infer/runtime/process/geometry.h"
#include "mw/infer/runtime/process/image_to_tensor.h"
#include "mw/infer/runtime/process/normalize.h"

namespace {

std::string DeviceName(mw::infer::Device device) {
  switch (device.type) {
    case mw::infer::DeviceType::kCpu:
      return "cpu";
    case mw::infer::DeviceType::kCuda:
      return "cuda:" + std::to_string(device.id);
  }
  return "unknown";
}

std::string ShapeString(const std::vector<int64_t>& shape) {
  return fmt::format("[{}]", fmt::join(shape, ", "));
}

mw::infer::Device ParseDevice(const std::string& value) {
  if (value == "cpu") {
    return mw::infer::Device{mw::infer::DeviceType::kCpu, 0};
  }
  if (value == "cuda") {
    return mw::infer::Device{mw::infer::DeviceType::kCuda, 0};
  }

  constexpr std::string_view kCudaPrefix = "cuda:";
  if (value.rfind(kCudaPrefix, 0) == 0) {
    const std::string id_text = value.substr(kCudaPrefix.size());
    std::size_t parsed = 0;
    const int device_id = std::stoi(id_text, &parsed);
    if (parsed != id_text.size() || device_id < 0) {
      throw std::invalid_argument("Invalid CUDA device: " + value);
    }
    return mw::infer::Device{mw::infer::DeviceType::kCuda, device_id};
  }

  throw std::invalid_argument("Unsupported device: " + value);
}

int64_t ResolveStaticDim(int64_t model_dim, int64_t fallback,
                         const char* name) {
  const int64_t value = model_dim > 0 ? model_dim : fallback;
  if (value <= 0) {
    throw std::invalid_argument(std::string("Invalid model input ") + name);
  }
  return value;
}

std::vector<int64_t> ResolveBchwInputShape(const mw::infer::TensorInfo& input) {
  if (input.data_type != mw::infer::DataType::kFloat32) {
    throw std::invalid_argument(
        "classification example expects a float32 input tensor");
  }
  if (input.shape.size() != 4) {
    throw std::invalid_argument(
        "classification example expects a BCHW rank-4 input tensor");
  }

  const int64_t batch = ResolveStaticDim(input.shape[0], 1, "batch");
  const int64_t channels = ResolveStaticDim(input.shape[1], 3, "channels");
  const int64_t height = ResolveStaticDim(input.shape[2], 224, "height");
  const int64_t width = ResolveStaticDim(input.shape[3], 224, "width");
  if (batch != 1) {
    throw std::invalid_argument(
        "classification example currently supports batch size 1");
  }
  if (channels != 3) {
    throw std::invalid_argument(
        "classification example currently supports 3-channel inputs");
  }
  return {batch, channels, height, width};
}

cv::Mat MakeSyntheticImage() {
  cv::Mat image(240, 320, CV_8UC3);
  for (int y = 0; y < image.rows; ++y) {
    for (int x = 0; x < image.cols; ++x) {
      image.at<cv::Vec3b>(y, x) =
          cv::Vec3b(static_cast<unsigned char>(x % 256),
                    static_cast<unsigned char>(y % 256),
                    static_cast<unsigned char>((x + y) % 256));
    }
  }
  return image;
}

cv::Mat LoadImage(const std::filesystem::path& path) {
  if (path == "--synthetic") {
    return MakeSyntheticImage();
  }

  cv::Mat image = cv::imread(path.string(), cv::IMREAD_COLOR);
  if (image.empty()) {
    throw std::invalid_argument("Failed to read image: " + path.string());
  }
  return image;
}

std::vector<std::string> LoadLabels(const std::filesystem::path& path) {
  std::ifstream file(path);
  if (!file) {
    throw std::invalid_argument("Failed to read labels: " + path.string());
  }

  std::vector<std::string> labels;
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty()) {
      continue;
    }

    const std::size_t comma = line.find(',');
    if (comma == std::string::npos) {
      labels.push_back(line);
      continue;
    }

    std::size_t parsed = 0;
    const int index = std::stoi(line.substr(0, comma), &parsed);
    if (parsed != comma || index < 0) {
      labels.push_back(line);
      continue;
    }
    if (labels.size() <= static_cast<std::size_t>(index)) {
      labels.resize(static_cast<std::size_t>(index) + 1);
    }
    labels[static_cast<std::size_t>(index)] = line.substr(comma + 1);
  }
  return labels;
}

std::vector<float> CopyFloatValuesToHost(const mw::infer::Tensor& tensor) {
  mw::infer::Tensor host =
      tensor.device().type == mw::infer::DeviceType::kCpu
          ? tensor
          : tensor.CopyTo(mw::infer::Device{mw::infer::DeviceType::kCpu, 0});
  if (host.data_type() != mw::infer::DataType::kFloat32) {
    throw std::invalid_argument("Expected float32 tensor");
  }

  const auto* data = static_cast<const float*>(host.data());
  return std::vector<float>(data, data + host.element_count());
}

std::vector<int64_t> CopyInt64ValuesToHost(const mw::infer::Tensor& tensor) {
  mw::infer::Tensor host =
      tensor.device().type == mw::infer::DeviceType::kCpu
          ? tensor
          : tensor.CopyTo(mw::infer::Device{mw::infer::DeviceType::kCpu, 0});
  if (host.data_type() != mw::infer::DataType::kInt64) {
    throw std::invalid_argument("Expected int64 tensor");
  }

  const auto* data = static_cast<const int64_t*>(host.data());
  return std::vector<int64_t>(data, data + host.element_count());
}

std::string LabelText(const std::vector<std::string>& labels, int64_t index) {
  if (index >= 0 && static_cast<std::size_t>(index) < labels.size() &&
      !labels[static_cast<std::size_t>(index)].empty()) {
    return labels[static_cast<std::size_t>(index)];
  }
  return fmt::format("class {}", index);
}

struct CliArguments {
  std::string model_path;
  std::string image_path;
  std::string labels_path;
  std::string device = "cpu";
  int top_k = 5;
};

struct PreprocessResult {
  mw::infer::Tensor tensor;
  mw::infer::ImageSize resized_size;
  mw::infer::ImageSize crop_size;
};

mw::infer::ImageSize ResizeShortSideSize(cv::Size original_size,
                                         int target_short_side) {
  if (original_size.width <= 0 || original_size.height <= 0) {
    throw std::invalid_argument("Invalid original image size");
  }
  if (target_short_side <= 0) {
    throw std::invalid_argument("Invalid resize short side");
  }

  if (original_size.width <= original_size.height) {
    const int height =
        static_cast<int>(std::lround(static_cast<double>(original_size.height) *
                                     target_short_side / original_size.width));
    return mw::infer::ImageSize{target_short_side, std::max(height, 1)};
  }

  const int width =
      static_cast<int>(std::lround(static_cast<double>(original_size.width) *
                                   target_short_side / original_size.height));
  return mw::infer::ImageSize{std::max(width, 1), target_short_side};
}

int ClassificationResizeShortSide(mw::infer::ImageSize crop_size) {
  constexpr double kImageNetResizeScale = 256.0 / 224.0;
  const int crop_short_side = std::min(crop_size.width, crop_size.height);
  return std::max(
      1, static_cast<int>(std::lround(crop_short_side * kImageNetResizeScale)));
}

PreprocessResult MakeInputTensor(cv::Mat image, mw::infer::Device device,
                                 mw::infer::TensorInfo input_info) {
  const std::vector<int64_t> shape = ResolveBchwInputShape(input_info);
  input_info.shape = shape;
  const mw::infer::ImageSize crop_size{static_cast<int>(shape[3]),
                                       static_cast<int>(shape[2])};
  const mw::infer::ImageSize resize_size = ResizeShortSideSize(
      image.size(), ClassificationResizeShortSide(crop_size));

  mw::infer::RawImageBatch raw_images(std::vector<cv::Mat>{std::move(image)});
  mw::infer::GeometryTransformer transformer;
  mw::infer::GeometryResult resized = transformer.Resize(
      std::move(raw_images), resize_size, mw::infer::Interpolation::kLinear);
  mw::infer::GeometryResult cropped =
      transformer.CenterCrop(std::move(resized), crop_size);

  mw::infer::Tensor tensor = mw::infer::ToTensor(
      cropped.images(), device, input_info, mw::infer::TensorLayout::kBchw);
  tensor = mw::infer::Normalize(tensor, {0.485F, 0.456F, 0.406F},
                                {0.229F, 0.224F, 0.225F}, 1.0F / 255.0F,
                                mw::infer::TensorLayout::kBchw);
  return PreprocessResult{std::move(tensor), resize_size, crop_size};
}

void ConfigureCli(CLI::App* app, CliArguments* arguments) {
  app->description("Run ImageNet-style classification inference with ONNX.");
  app->add_option("model", arguments->model_path, "ONNX model path")
      ->required()
      ->check(CLI::ExistingFile);
  app->add_option("image", arguments->image_path,
                  "Image path, or --synthetic for a generated test image")
      ->required();
  app->add_option("labels", arguments->labels_path, "Label text file")
      ->required()
      ->check(CLI::ExistingFile);
  app->add_option("device", arguments->device,
                  "Execution device: cpu, cuda, or cuda:<id>")
      ->capture_default_str();
  app->add_option("top_k", arguments->top_k, "Number of classes to print")
      ->capture_default_str()
      ->check(CLI::PositiveNumber);
}

}  // namespace

int main(int argc, char** argv) {
  CliArguments arguments;
  CLI::App app;
  ConfigureCli(&app, &arguments);
  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& error) {
    return app.exit(error);
  }

  try {
    const std::filesystem::path model_path = arguments.model_path;
    const std::filesystem::path image_path = arguments.image_path;
    const std::filesystem::path labels_path = arguments.labels_path;
    const mw::infer::Device device = ParseDevice(arguments.device);
    const int top_k = arguments.top_k;

    mw::infer::Model model = mw::infer::ModelFromPath(model_path);
    mw::infer::BackendPtr backend =
        mw::infer::CreateBackend(std::move(model), device);
    const mw::infer::ModelInfo& model_info = backend->model_info();
    if (model_info.inputs.size() != 1 || model_info.outputs.empty()) {
      throw std::invalid_argument(
          "classification example expects one input and at least one output");
    }

    cv::Mat image = LoadImage(image_path);
    const cv::Size original_size = image.size();
    PreprocessResult preprocess =
        MakeInputTensor(std::move(image), device, model_info.inputs.front());
    std::vector<mw::infer::Tensor> outputs = backend->Infer(preprocess.tensor);
    if (outputs.empty()) {
      throw std::runtime_error("backend returned no outputs");
    }

    mw::infer::Tensor probabilities = mw::infer::Softmax(outputs.front());
    mw::infer::TopKResult topk = mw::infer::TopK(probabilities, top_k);
    const std::vector<float> scores = CopyFloatValuesToHost(topk.scores);
    const std::vector<int64_t> indices = CopyInt64ValuesToHost(topk.indices);
    const std::vector<std::string> labels = LoadLabels(labels_path);

    fmt::print("model: {}\n", backend->model().name);
    fmt::print("backend_device: {}\n", DeviceName(device));
    fmt::print("image: {} ({}x{})\n", image_path.string(), original_size.width,
               original_size.height);
    fmt::print("resize: {}x{} -> {}x{}\n", original_size.width,
               original_size.height, preprocess.resized_size.width,
               preprocess.resized_size.height);
    fmt::print("center_crop: {}x{}\n", preprocess.crop_size.width,
               preprocess.crop_size.height);
    fmt::print(
        "preprocess: OpenCV ToTensor RGB, scale=1/255, ImageNet mean/std\n");
    fmt::print("input: name=\"{}\" shape={} device={}\n",
               preprocess.tensor.name(), ShapeString(preprocess.tensor.shape()),
               DeviceName(preprocess.tensor.device()));
    fmt::print("output: name=\"{}\" shape={} device={}\n",
               outputs.front().name(), ShapeString(outputs.front().shape()),
               DeviceName(outputs.front().device()));
    fmt::print("top_{}:\n", top_k);
    for (std::size_t index = 0; index < indices.size(); ++index) {
      fmt::print("  {:>2}: class={} score={:.6f} label=\"{}\"\n", index + 1,
                 indices[index], scores[index],
                 LabelText(labels, indices[index]));
    }
  } catch (const std::exception& error) {
    fmt::print(stderr, "classification_infer failed: {}\n", error.what());
    return 1;
  }

  return 0;
}
