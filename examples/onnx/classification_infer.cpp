#include <fmt/core.h>
#include <fmt/ranges.h>

#include <CLI/CLI.hpp>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mw/infer/runtime/backend/backend.h"
#include "mw/infer/runtime/input/opencv_input.h"
#include "mw/infer/runtime/postprocess/label_map.h"
#include "mw/infer/runtime/postprocess/softmax.h"
#include "mw/infer/runtime/postprocess/topk.h"
#include "mw/infer/runtime/process/geometry.h"
#include "mw/infer/runtime/process/image_to_tensor.h"
#include "mw/infer/runtime/process/normalize.h"

namespace {

constexpr mw::infer::ImageSize kImageNetCropSize{224, 224};
constexpr int kImageNetResizeShortSide = 256;
constexpr int64_t kImageNetBatch = 1;
constexpr int64_t kImageNetChannels = 3;

std::string ShapeString(const std::vector<int64_t>& shape) {
  return fmt::format("[{}]", fmt::join(shape, ", "));
}

mw::infer::TensorInfo MakeImageNetInputInfo(std::string input_name) {
  mw::infer::TensorInfo input_info;
  input_info.name = std::move(input_name);
  input_info.data_type = mw::infer::DataType::kFloat32;
  input_info.shape = {kImageNetBatch, kImageNetChannels,
                      kImageNetCropSize.height, kImageNetCropSize.width};
  return input_info;
}

cv::Mat LoadImage(const std::filesystem::path& path) {
  cv::Mat image = cv::imread(path.string(), cv::IMREAD_COLOR);
  if (image.empty()) {
    throw std::invalid_argument("Failed to read image: " + path.string());
  }
  return image;
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

PreprocessResult MakeInputTensor(cv::Mat image, mw::infer::Device device,
                                 std::string input_name) {
  const mw::infer::ImageSize crop_size = kImageNetCropSize;
  mw::infer::TensorInfo input_info =
      MakeImageNetInputInfo(std::move(input_name));
  const mw::infer::ImageSize original_size{image.cols, image.rows};
  const mw::infer::ImageSize resize_size =
      mw::infer::ResizeShortSideSize(original_size, kImageNetResizeShortSide);

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
  app->description(
      "Run an ImageNet classification example with ONNX. The example expects "
      "a float32 BCHW model input shaped [1, 3, 224, 224].");
  app->add_option("model", arguments->model_path, "ONNX model path")
      ->required()
      ->check(CLI::ExistingFile);
  app->add_option("image", arguments->image_path, "Image path")
      ->required()
      ->check(CLI::ExistingFile);
  app->add_option("labels", arguments->labels_path,
                  "ImageNet label text file with 1000 labels")
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
    const mw::infer::Device device(arguments.device);
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
    PreprocessResult preprocess = MakeInputTensor(
        std::move(image), device, model_info.inputs.front().name);
    std::vector<mw::infer::Tensor> outputs = backend->Infer(preprocess.tensor);
    if (outputs.empty()) {
      throw std::runtime_error("backend returned no outputs");
    }

    mw::infer::Tensor probabilities = mw::infer::Softmax(outputs.front());
    mw::infer::TopKResult topk = mw::infer::TopK(probabilities, top_k);
    const std::vector<float> scores = topk.scores.CopyToHostVector<float>();
    const std::vector<int64_t> indices =
        topk.indices.CopyToHostVector<int64_t>();
    const mw::infer::LabelMap labels = mw::infer::LabelMapFromFile(labels_path);

    fmt::print("model: {}\n", backend->model().name);
    fmt::print("backend_device: {}\n", device.ToString());
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
               preprocess.tensor.device().ToString());
    fmt::print("output: name=\"{}\" shape={} device={}\n",
               outputs.front().name(), ShapeString(outputs.front().shape()),
               outputs.front().device().ToString());
    fmt::print("top_{}:\n", top_k);
    for (std::size_t index = 0; index < indices.size(); ++index) {
      fmt::print("  {:>2}: class={} score={:.6f} label=\"{}\"\n", index + 1,
                 indices[index], scores[index],
                 labels.LabelOrClass(indices[index]));
    }
  } catch (const std::exception& error) {
    fmt::print(stderr, "classification_infer failed: {}\n", error.what());
    return 1;
  }

  return 0;
}
