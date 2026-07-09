#include <fmt/core.h>
#include <fmt/ranges.h>

#include <CLI/CLI.hpp>
#include <algorithm>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mw/infer/runtime/backend/backend.h"

namespace {

std::string ShapeString(const std::vector<int64_t>& shape) {
  return fmt::format("[{}]", fmt::join(shape, ", "));
}

bool HasDynamicShape(const std::vector<int64_t>& shape) {
  return std::any_of(shape.begin(), shape.end(),
                     [](int64_t dim) { return dim <= 0; });
}

std::vector<int64_t> ResolveInputShape(const mw::infer::ModelInfo& model_info,
                                       const mw::infer::TensorInfo& input) {
  if (!HasDynamicShape(input.shape)) {
    return input.shape;
  }

  if (!model_info.profiles.empty()) {
    for (const mw::infer::TensorShapeRange& range :
         model_info.profiles.front().inputs) {
      if (range.name == input.name && !HasDynamicShape(range.opt_shape)) {
        return range.opt_shape;
      }
    }
  }

  throw std::invalid_argument(
      "TensorRT demo requires static input shape or profile0 opt shape");
}

void FillMnistLikeInput(float* values, const std::vector<int64_t>& shape) {
  const std::size_t count = mw::infer::ElementCount(shape);
  std::fill(values, values + count, 0.0F);

  if (shape.size() != 4 || shape[0] != 1 || shape[1] != 1) {
    for (std::size_t index = 0; index < count; ++index) {
      values[index] = static_cast<float>(index % 251) / 250.0F;
    }
    return;
  }

  const int64_t height = shape[2];
  const int64_t width = shape[3];
  if (height < 10 || width < 10) {
    return;
  }

  const int64_t center_x = width / 2;
  for (int64_t y = height / 5; y < height * 4 / 5; ++y) {
    values[static_cast<std::size_t>(y * width + center_x)] = 1.0F;
    if (center_x + 1 < width) {
      values[static_cast<std::size_t>(y * width + center_x + 1)] = 1.0F;
    }
  }
  for (int64_t offset = 0; offset < width / 8; ++offset) {
    const int64_t y = height / 5 + offset;
    const int64_t x = center_x - offset - 1;
    if (y >= 0 && y < height && x >= 0 && x < width) {
      values[static_cast<std::size_t>(y * width + x)] = 1.0F;
    }
  }
}

mw::infer::Tensor MakeInputTensor(const mw::infer::TensorInfo& input_info,
                                  std::vector<int64_t> shape,
                                  mw::infer::Device execution_device) {
  mw::infer::TensorDesc desc;
  desc.info.name = input_info.name;
  desc.info.data_type = mw::infer::DataType::kFloat32;
  desc.info.shape = std::move(shape);
  desc.device = mw::infer::Device{mw::infer::DeviceType::kCpu, 0};

  mw::infer::Tensor host_input = mw::infer::Tensor::Allocate(std::move(desc));
  FillMnistLikeInput(host_input.data<float>(), host_input.shape());
  return host_input.CopyTo(execution_device);
}

std::size_t FlatClassCount(const std::vector<int64_t>& shape) {
  if (shape.empty()) {
    return 0;
  }
  if (shape.size() == 1) {
    return static_cast<std::size_t>(shape[0]);
  }
  if (shape.size() == 2 && shape[0] == 1) {
    return static_cast<std::size_t>(shape[1]);
  }
  return 0;
}

struct CliArguments {
  std::string engine_path;
  std::string device = "cuda:0";
  std::size_t max_values = 10;
};

void ConfigureCli(CLI::App* app, CliArguments* arguments) {
  app->description(
      "Run a minimal TensorRT MNIST-style inference demo. The demo expects a "
      "single float32 input and prints the first output tensor.");
  app->add_option("engine", arguments->engine_path,
                  "TensorRT engine path (.engine or .plan)")
      ->required()
      ->check(CLI::ExistingFile);
  app->add_option("device", arguments->device,
                  "Execution device: cuda or cuda:<id>")
      ->capture_default_str();
  app->add_option("max_values", arguments->max_values,
                  "Maximum output values to print")
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
    const std::filesystem::path engine_path = arguments.engine_path;
    const mw::infer::Device device(arguments.device);
    if (device.type != mw::infer::DeviceType::kCuda) {
      throw std::invalid_argument("TensorRT demo requires a CUDA device");
    }

    mw::infer::Model model = mw::infer::ModelFromPath(engine_path);
    if (model.format != mw::infer::ModelFormat::kTensorRT) {
      throw std::invalid_argument("TensorRT demo expects .engine or .plan");
    }

    mw::infer::BackendPtr backend =
        mw::infer::CreateBackend(std::move(model), device);
    const mw::infer::ModelInfo& model_info = backend->model_info();
    if (model_info.inputs.size() != 1 || model_info.outputs.empty()) {
      throw std::invalid_argument(
          "TensorRT demo expects one input and at least one output");
    }
    const mw::infer::TensorInfo& input_info = model_info.inputs.front();
    if (input_info.data_type != mw::infer::DataType::kFloat32) {
      throw std::invalid_argument("TensorRT demo expects float32 input");
    }

    std::vector<int64_t> input_shape =
        ResolveInputShape(model_info, input_info);
    mw::infer::Tensor input =
        MakeInputTensor(input_info, std::move(input_shape), device);
    std::vector<mw::infer::Tensor> outputs = backend->Infer(input);
    if (outputs.empty()) {
      throw std::runtime_error("backend returned no outputs");
    }
    if (outputs.front().data_type() != mw::infer::DataType::kFloat32) {
      throw std::invalid_argument("TensorRT demo expects float32 output");
    }

    const std::vector<float> values = outputs.front().CopyToHostVector<float>();
    const std::size_t print_count =
        std::min(arguments.max_values, values.size());
    const std::vector<float> printed(values.begin(),
                                     values.begin() + print_count);

    fmt::print("model: {}\n", backend->model().name);
    fmt::print("engine: {}\n", engine_path.string());
    fmt::print("backend_device: {}\n", backend->execution_device().ToString());
    fmt::print("input: name=\"{}\" shape={} device={}\n", input.name(),
               ShapeString(input.shape()), input.device().ToString());
    fmt::print("output: name=\"{}\" shape={} device={}\n",
               outputs.front().name(), ShapeString(outputs.front().shape()),
               outputs.front().device().ToString());
    fmt::print("output_values(first {}): [{}]\n", print_count,
               fmt::join(printed, ", "));

    const std::size_t class_count = FlatClassCount(outputs.front().shape());
    if (class_count > 0 && values.size() >= class_count) {
      const auto best =
          std::max_element(values.begin(), values.begin() + class_count);
      fmt::print("top1: class={} value={:.6f}\n",
                 std::distance(values.begin(), best), *best);
    }
  } catch (const std::exception& error) {
    fmt::print(stderr, "tensorrt_mnist_infer failed: {}\n", error.what());
    return 1;
  }

  return 0;
}
