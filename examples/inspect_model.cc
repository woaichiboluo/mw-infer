#include <CLI/CLI.hpp>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

#include "mw/infer/mw_infer.h"

namespace {

mw::infer::BackendKind ParseBackend(const std::string& backend) {
  if (backend == "onnx_cpu") {
    return mw::infer::BackendKind::kOnnxCpu;
  }
  if (backend == "onnx_cuda") {
    return mw::infer::BackendKind::kOnnxCuda;
  }
  if (backend == "tensorrt") {
    return mw::infer::BackendKind::kTensorRT;
  }

  throw std::invalid_argument("unsupported backend: " + backend);
}

}  // namespace

int main(int argc, char** argv) {
  CLI::App app{"Inspect a MwInfer runtime configuration."};

  std::string backend = "onnx_cpu";
  std::string model_path;

  app.add_option("model", model_path,
                 "Path to .onnx or TensorRT serialized engine model")
      ->required();
  app.add_option("--backend", backend,
                 "Backend: onnx_cpu, onnx_cuda, or tensorrt")
      ->check(CLI::IsMember({"onnx_cpu", "onnx_cuda", "tensorrt"}));

  CLI11_PARSE(app, argc, argv);

  try {
    mw::infer::RuntimeConfig config;
    config.backend = ParseBackend(backend);
    config.model = mw::infer::ModelFromPath(model_path);

    mw::infer::ValidateRuntimeConfig(config);
    std::cout << "model=" << config.model.source.path.string()
              << " backend=" << mw::infer::BackendName(config.backend) << '\n';
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }

  return 0;
}
