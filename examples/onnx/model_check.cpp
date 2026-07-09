#include <fmt/core.h>
#include <fmt/ranges.h>

#include <CLI/CLI.hpp>
#include <exception>
#include <filesystem>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "mw/infer/runtime/backend/backend.h"

namespace {

std::string_view DataTypeName(mw::infer::DataType data_type) {
  switch (data_type) {
    case mw::infer::DataType::kUInt8:
      return "uint8";
    case mw::infer::DataType::kInt8:
      return "int8";
    case mw::infer::DataType::kUInt16:
      return "uint16";
    case mw::infer::DataType::kInt16:
      return "int16";
    case mw::infer::DataType::kInt32:
      return "int32";
    case mw::infer::DataType::kInt64:
      return "int64";
    case mw::infer::DataType::kFloat16:
      return "float16";
    case mw::infer::DataType::kFloat32:
      return "float32";
    case mw::infer::DataType::kFloat64:
      return "float64";
    case mw::infer::DataType::kUnknown:
      return "unknown";
  }
  return "unknown";
}

std::string ShapeString(const std::vector<int64_t>& shape) {
  return fmt::format("[{}]", fmt::join(shape, ", "));
}

void PrintTensorInfos(const char* label,
                      const std::vector<mw::infer::TensorInfo>& infos) {
  fmt::print("{}: {}\n", label, infos.size());
  for (std::size_t index = 0; index < infos.size(); ++index) {
    const mw::infer::TensorInfo& info = infos[index];
    fmt::print("  [{}] name=\"{}\" type={} shape={}\n", index, info.name,
               DataTypeName(info.data_type), ShapeString(info.shape));
  }
}

struct CliArguments {
  std::string model_path;
  std::string device = "cpu";
};

void ConfigureCli(CLI::App* app, CliArguments* arguments) {
  app->description("Inspect an ONNX model through the MwInfer backend.");
  app->add_option("model", arguments->model_path, "ONNX model path")
      ->required()
      ->check(CLI::ExistingFile);
  app->add_option("device", arguments->device,
                  "Execution device: cpu, cuda, or cuda:<id>")
      ->capture_default_str();
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
    const mw::infer::Device device(arguments.device);

    mw::infer::Model model = mw::infer::ModelFromPath(model_path);
    mw::infer::BackendPtr backend =
        mw::infer::CreateBackend(std::move(model), device);
    const mw::infer::Model& active_model = backend->model();
    const mw::infer::ModelInfo& model_info = backend->model_info();

    fmt::print("model: {}\n", active_model.name);
    fmt::print("path: {}\n", model_path.string());
    fmt::print("backend_device: {}\n", backend->execution_device().ToString());
    PrintTensorInfos("inputs", model_info.inputs);
    PrintTensorInfos("outputs", model_info.outputs);
  } catch (const std::exception& error) {
    fmt::print(stderr, "model_check failed: {}\n", error.what());
    return 1;
  }

  return 0;
}
