#include <NvInferPlugin.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mw/infer/runtime/backend/backend.h"

namespace mw::infer {
namespace {

constexpr int32_t kDefaultOptimizationProfileIndex = 0;

class TensorRtLogger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* message) noexcept override {
    static_cast<void>(severity);
    static_cast<void>(message);
  }
};

std::string CudaErrorMessage(cudaError_t status, const char* operation) {
  return std::string(operation) + " failed: " + cudaGetErrorString(status);
}

void CheckCuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(CudaErrorMessage(status, operation));
  }
}

template <typename T>
std::unique_ptr<T> CheckTensorRtPtr(T* ptr, const char* operation) {
  if (ptr == nullptr) {
    throw std::runtime_error(std::string(operation) + " failed");
  }
  return std::unique_ptr<T>(ptr);
}

void ValidateModel(const Model& model) {
  if (model.format != ModelFormat::kTensorRT) {
    throw std::invalid_argument("TensorRT backend requires a TensorRT engine");
  }

  if (model.source.kind == ModelSourceKind::kPath) {
    if (model.source.path.empty()) {
      throw std::invalid_argument("TensorRT engine path is empty");
    }
    if (!std::filesystem::exists(model.source.path)) {
      throw std::invalid_argument("TensorRT engine path does not exist: " +
                                  model.source.path.string());
    }
    return;
  }

  if (model.source.data == nullptr || model.source.bytes == 0) {
    throw std::invalid_argument("TensorRT engine memory source is empty");
  }
}

std::vector<std::uint8_t> ReadFileBytes(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    throw std::invalid_argument("Failed to open TensorRT engine: " +
                                path.string());
  }

  const std::ifstream::pos_type size = input.tellg();
  if (size <= 0) {
    throw std::invalid_argument("TensorRT engine file is empty: " +
                                path.string());
  }

  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
  input.seekg(0, std::ios::beg);
  input.read(reinterpret_cast<char*>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  if (!input) {
    throw std::invalid_argument("Failed to read TensorRT engine: " +
                                path.string());
  }
  return bytes;
}

DataType FromTensorRtDataType(nvinfer1::DataType data_type) {
  switch (data_type) {
    case nvinfer1::DataType::kFLOAT:
      return DataType::kFloat32;
    case nvinfer1::DataType::kHALF:
      return DataType::kFloat16;
    case nvinfer1::DataType::kINT8:
      return DataType::kInt8;
    case nvinfer1::DataType::kINT32:
      return DataType::kInt32;
    case nvinfer1::DataType::kUINT8:
      return DataType::kUInt8;
    case nvinfer1::DataType::kINT64:
      return DataType::kInt64;
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kFP8:
    case nvinfer1::DataType::kBF16:
    case nvinfer1::DataType::kINT4:
      break;
  }
  throw std::invalid_argument("Unsupported TensorRT tensor data type");
}

std::vector<int64_t> FromTensorRtDims(nvinfer1::Dims dims, bool allow_dynamic) {
  if (dims.nbDims < 0) {
    throw std::invalid_argument("TensorRT tensor rank is invalid");
  }

  std::vector<int64_t> shape;
  shape.reserve(static_cast<std::size_t>(dims.nbDims));
  for (int32_t index = 0; index < dims.nbDims; ++index) {
    const int64_t dim = dims.d[index];
    if (!allow_dynamic && dim <= 0) {
      throw std::invalid_argument(
          "TensorRT runtime tensor shape must be concrete");
    }
    shape.push_back(dim);
  }
  return shape;
}

nvinfer1::Dims ToTensorRtDims(const std::vector<int64_t>& shape) {
  if (shape.size() > static_cast<std::size_t>(nvinfer1::Dims::MAX_DIMS)) {
    throw std::invalid_argument("TensorRT tensor rank exceeds max rank");
  }

  nvinfer1::Dims dims{};
  dims.nbDims = static_cast<int32_t>(shape.size());
  for (int32_t index = 0; index < nvinfer1::Dims::MAX_DIMS; ++index) {
    dims.d[index] = 0;
  }
  for (std::size_t index = 0; index < shape.size(); ++index) {
    if (shape[index] <= 0) {
      throw std::invalid_argument(
          "TensorRT input tensor dimensions must be positive");
    }
    dims.d[index] = shape[index];
  }
  return dims;
}

void ValidateTensorMemoryLayout(const nvinfer1::ICudaEngine& engine,
                                const std::string& name) {
  const char* tensor_name = name.c_str();
  const bool has_profiles = engine.getNbOptimizationProfiles() > 0;
  const nvinfer1::TensorFormat format =
      has_profiles ? engine.getTensorFormat(tensor_name,
                                            kDefaultOptimizationProfileIndex)
                   : engine.getTensorFormat(tensor_name);
  const int32_t vectorized_dim =
      has_profiles ? engine.getTensorVectorizedDim(
                         tensor_name, kDefaultOptimizationProfileIndex)
                   : engine.getTensorVectorizedDim(tensor_name);
  const int32_t components_per_element =
      has_profiles ? engine.getTensorComponentsPerElement(
                         tensor_name, kDefaultOptimizationProfileIndex)
                   : engine.getTensorComponentsPerElement(tensor_name);

  if (format == nvinfer1::TensorFormat::kLINEAR && vectorized_dim == -1 &&
      components_per_element <= 1) {
    return;
  }

  const char* format_desc =
      has_profiles ? engine.getTensorFormatDesc(
                         tensor_name, kDefaultOptimizationProfileIndex)
                   : engine.getTensorFormatDesc(tensor_name);
  std::string message =
      "TensorRT backend supports dense linear IO tensors only: " + name;
  if (format_desc != nullptr && format_desc[0] != '\0') {
    message += " (";
    message += format_desc;
    message += ")";
  }
  throw std::invalid_argument(message);
}

TensorInfo ReadTensorInfo(const nvinfer1::ICudaEngine& engine,
                          const std::string& name) {
  if (engine.getTensorLocation(name.c_str()) !=
      nvinfer1::TensorLocation::kDEVICE) {
    throw std::invalid_argument(
        "TensorRT backend supports device IO tensors only");
  }
  if (engine.isShapeInferenceIO(name.c_str())) {
    throw std::invalid_argument(
        "TensorRT backend does not support shape inference IO tensors");
  }
  ValidateTensorMemoryLayout(engine, name);

  TensorInfo info;
  info.name = name;
  info.data_type = FromTensorRtDataType(engine.getTensorDataType(name.c_str()));
  info.shape = FromTensorRtDims(engine.getTensorShape(name.c_str()), true);
  return info;
}

ModelProfile ReadProfile(const nvinfer1::ICudaEngine& engine,
                         const std::vector<TensorInfo>& inputs,
                         int32_t profile_index) {
  ModelProfile profile;
  profile.name = "profile" + std::to_string(profile_index);
  profile.inputs.reserve(inputs.size());
  for (const TensorInfo& input : inputs) {
    const nvinfer1::Dims min_shape = engine.getProfileShape(
        input.name.c_str(), profile_index, nvinfer1::OptProfileSelector::kMIN);
    const nvinfer1::Dims opt_shape = engine.getProfileShape(
        input.name.c_str(), profile_index, nvinfer1::OptProfileSelector::kOPT);
    const nvinfer1::Dims max_shape = engine.getProfileShape(
        input.name.c_str(), profile_index, nvinfer1::OptProfileSelector::kMAX);
    if (min_shape.nbDims < 0 || opt_shape.nbDims < 0 || max_shape.nbDims < 0) {
      continue;
    }

    TensorShapeRange range;
    range.name = input.name;
    range.min_shape = FromTensorRtDims(min_shape, false);
    range.opt_shape = FromTensorRtDims(opt_shape, false);
    range.max_shape = FromTensorRtDims(max_shape, false);
    profile.inputs.push_back(std::move(range));
  }
  return profile;
}

ModelInfo ReadModelInfo(const nvinfer1::ICudaEngine& engine) {
  ModelInfo info;
  const int32_t tensor_count = engine.getNbIOTensors();
  for (int32_t index = 0; index < tensor_count; ++index) {
    const char* tensor_name = engine.getIOTensorName(index);
    if (tensor_name == nullptr || tensor_name[0] == '\0') {
      throw std::invalid_argument("TensorRT engine has an unnamed IO tensor");
    }

    const std::string name = tensor_name;
    const nvinfer1::TensorIOMode mode = engine.getTensorIOMode(tensor_name);
    if (mode == nvinfer1::TensorIOMode::kINPUT) {
      info.inputs.push_back(ReadTensorInfo(engine, name));
    } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
      info.outputs.push_back(ReadTensorInfo(engine, name));
    } else {
      throw std::invalid_argument("TensorRT engine has a non-IO tensor");
    }
  }

  if (info.inputs.empty()) {
    throw std::invalid_argument("TensorRT engine has no inputs");
  }
  if (info.outputs.empty()) {
    throw std::invalid_argument("TensorRT engine has no outputs");
  }

  if (engine.getNbOptimizationProfiles() > 0) {
    info.profiles.push_back(
        ReadProfile(engine, info.inputs, kDefaultOptimizationProfileIndex));
  }
  return info;
}

std::size_t FindTensorInfo(const std::vector<TensorInfo>& infos,
                           const std::string& name, const char* kind) {
  const auto iter = std::find_if(
      infos.begin(), infos.end(),
      [&name](const TensorInfo& info) { return info.name == name; });
  if (iter == infos.end()) {
    throw std::invalid_argument(std::string("Unknown TensorRT ") + kind +
                                " name: " + name);
  }
  return static_cast<std::size_t>(std::distance(infos.begin(), iter));
}

ModelInfo ResolveActiveModelInfo(
    ModelInfo model_info, const std::vector<std::string>& requested_outputs,
    std::vector<std::size_t>* return_output_indices) {
  return_output_indices->clear();
  if (requested_outputs.empty()) {
    return_output_indices->reserve(model_info.outputs.size());
    for (std::size_t index = 0; index < model_info.outputs.size(); ++index) {
      return_output_indices->push_back(index);
    }
    return model_info;
  }

  ModelInfo active_info;
  active_info.inputs = std::move(model_info.inputs);
  active_info.profiles = std::move(model_info.profiles);
  active_info.outputs.reserve(requested_outputs.size());
  return_output_indices->reserve(requested_outputs.size());
  std::vector<bool> assigned(model_info.outputs.size(), false);
  for (const std::string& name : requested_outputs) {
    const std::size_t index =
        FindTensorInfo(model_info.outputs, name, "output");
    if (assigned[index]) {
      throw std::invalid_argument("Duplicate TensorRT output name: " + name);
    }
    active_info.outputs.push_back(model_info.outputs[index]);
    return_output_indices->push_back(index);
    assigned[index] = true;
  }
  return active_info;
}

class TensorRtBackendSession {
 public:
  TensorRtBackendSession(Device execution_device,
                         std::vector<std::string> output_names,
                         const Model& model, ModelInfo& active_info)
      : execution_device_(execution_device) {
    if (execution_device_.type != DeviceType::kCuda ||
        execution_device_.id < 0) {
      throw std::invalid_argument(
          "TensorRT backend requires a non-negative CUDA device");
    }
    ValidateModel(model);
    CheckCuda(cudaSetDevice(execution_device_.id), "cudaSetDevice");

    if (!initLibNvInferPlugins(&logger_, "")) {
      throw std::runtime_error("initLibNvInferPlugins failed");
    }
    runtime_ = CheckTensorRtPtr(nvinfer1::createInferRuntime(logger_),
                                "nvinfer1::createInferRuntime");
    engine_ = DeserializeEngine(model);
    context_ = CheckTensorRtPtr(engine_->createExecutionContext(),
                                "ICudaEngine::createExecutionContext");

    ModelInfo full_info = ReadModelInfo(*engine_);
    input_names_.reserve(full_info.inputs.size());
    for (const TensorInfo& input : full_info.inputs) {
      input_names_.push_back(input.name);
    }
    all_output_names_.reserve(full_info.outputs.size());
    for (const TensorInfo& output : full_info.outputs) {
      all_output_names_.push_back(output.name);
    }
    all_output_infos_ = full_info.outputs;
    active_info = ResolveActiveModelInfo(std::move(full_info), output_names,
                                         &return_output_indices_);

    if (context_->getOptimizationProfile() !=
        kDefaultOptimizationProfileIndex) {
      throw std::runtime_error(
          "TensorRT execution context did not select profile0");
    }
    CheckCuda(cudaStreamCreate(&stream_), "cudaStreamCreate");
  }

  TensorRtBackendSession(const TensorRtBackendSession&) = delete;
  TensorRtBackendSession& operator=(const TensorRtBackendSession&) = delete;

  ~TensorRtBackendSession() {
    if (stream_ != nullptr) {
      static_cast<void>(cudaSetDevice(execution_device_.id));
      static_cast<void>(cudaStreamDestroy(stream_));
    }
  }

  std::vector<Tensor> Infer(const std::vector<Tensor>& inputs,
                            const ModelInfo& model_info) {
    CheckCuda(cudaSetDevice(execution_device_.id), "cudaSetDevice");
    std::vector<const Tensor*> ordered_inputs =
        ResolveInputs(inputs, model_info);
    std::vector<Tensor> staged_inputs;
    std::vector<const Tensor*> bound_inputs =
        StageInputs(ordered_inputs, &staged_inputs);

    BindInputs(bound_inputs);
    InferShapes();
    std::vector<Tensor> all_outputs = AllocateOutputs();
    BindOutputs(all_outputs);

    if (!context_->enqueueV3(stream_)) {
      throw std::runtime_error("TensorRT enqueueV3 failed");
    }
    CheckCuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");

    return SelectOutputs(std::move(all_outputs));
  }

 private:
  std::unique_ptr<nvinfer1::ICudaEngine> DeserializeEngine(const Model& model) {
    std::vector<std::uint8_t> file_bytes;
    const void* data = model.source.data;
    std::size_t bytes = model.source.bytes;
    if (model.source.kind == ModelSourceKind::kPath) {
      file_bytes = ReadFileBytes(model.source.path);
      data = file_bytes.data();
      bytes = file_bytes.size();
    }

    return CheckTensorRtPtr(runtime_->deserializeCudaEngine(data, bytes),
                            "IRuntime::deserializeCudaEngine");
  }

  void ValidateInputTensor(const Tensor& tensor,
                           const TensorInfo& expected) const {
    if (tensor.empty()) {
      throw std::invalid_argument("TensorRT input tensor is empty");
    }
    if (tensor.data_type() != expected.data_type) {
      throw std::invalid_argument("TensorRT input tensor data type mismatch");
    }
    if (tensor.shape().size() != expected.shape.size()) {
      throw std::invalid_argument("TensorRT input tensor rank mismatch");
    }
    for (std::size_t dim = 0; dim < expected.shape.size(); ++dim) {
      if (expected.shape[dim] > 0 &&
          expected.shape[dim] != tensor.shape()[dim]) {
        throw std::invalid_argument("TensorRT input tensor shape mismatch");
      }
    }

    if (tensor.device().type == DeviceType::kCuda &&
        tensor.device().id != execution_device_.id) {
      throw std::invalid_argument(
          "CUDA input tensor device id must match TensorRT backend device id");
    }
    if (tensor.device().type != DeviceType::kCpu &&
        tensor.device().type != DeviceType::kCuda) {
      throw std::invalid_argument("Unsupported TensorRT input tensor device");
    }
  }

  std::vector<const Tensor*> ResolveInputs(const std::vector<Tensor>& inputs,
                                           const ModelInfo& model_info) const {
    if (inputs.size() != model_info.inputs.size()) {
      throw std::invalid_argument("TensorRT input tensor count mismatch");
    }

    std::vector<const Tensor*> ordered(model_info.inputs.size(), nullptr);
    if (inputs.size() == 1 && inputs.front().name().empty()) {
      ValidateInputTensor(inputs.front(), model_info.inputs.front());
      ordered.front() = &inputs.front();
      return ordered;
    }

    std::vector<bool> assigned(model_info.inputs.size(), false);
    for (const Tensor& input : inputs) {
      if (input.name().empty()) {
        throw std::invalid_argument(
            "TensorRT input tensor name is required for multi-input models");
      }
      const std::size_t index =
          FindTensorInfo(model_info.inputs, input.name(), "input");
      if (assigned[index]) {
        throw std::invalid_argument("Duplicate TensorRT input tensor name: " +
                                    input.name());
      }
      ValidateInputTensor(input, model_info.inputs[index]);
      ordered[index] = &input;
      assigned[index] = true;
    }

    return ordered;
  }

  std::vector<const Tensor*> StageInputs(
      const std::vector<const Tensor*>& inputs,
      std::vector<Tensor>* staged_inputs) {
    staged_inputs->clear();
    staged_inputs->reserve(inputs.size());

    std::vector<const Tensor*> bound_inputs;
    bound_inputs.reserve(inputs.size());
    for (const Tensor* input : inputs) {
      if (input->device().type == DeviceType::kCuda) {
        bound_inputs.push_back(input);
        continue;
      }

      staged_inputs->push_back(input->CopyTo(execution_device_));
      bound_inputs.push_back(&staged_inputs->back());
    }
    return bound_inputs;
  }

  void BindInputs(const std::vector<const Tensor*>& inputs) {
    for (std::size_t index = 0; index < inputs.size(); ++index) {
      const std::string& name = input_names_[index];
      const nvinfer1::Dims dims = ToTensorRtDims(inputs[index]->shape());
      if (!context_->setInputShape(name.c_str(), dims)) {
        throw std::invalid_argument(
            "TensorRT input tensor shape is outside "
            "the active optimization profile");
      }
      if (!context_->setInputTensorAddress(name.c_str(),
                                           inputs[index]->data())) {
        throw std::runtime_error("TensorRT setInputTensorAddress failed: " +
                                 name);
      }
    }
  }

  void InferShapes() {
    const int32_t missing_count = context_->inferShapes(0, nullptr);
    if (missing_count < 0) {
      throw std::runtime_error("TensorRT inferShapes failed");
    }
    if (missing_count == 0) {
      return;
    }

    std::vector<const char*> missing_names(
        static_cast<std::size_t>(missing_count), nullptr);
    const int32_t reported_count =
        context_->inferShapes(missing_count, missing_names.data());

    std::string message = "TensorRT input shapes are insufficient";
    if (reported_count > 0) {
      message += ": ";
      const int32_t limit = std::min(reported_count, missing_count);
      for (int32_t index = 0; index < limit; ++index) {
        if (index > 0) {
          message += ", ";
        }
        message += missing_names[static_cast<std::size_t>(index)] != nullptr
                       ? missing_names[static_cast<std::size_t>(index)]
                       : "<unknown>";
      }
    }
    throw std::invalid_argument(message);
  }

  std::vector<Tensor> AllocateOutputs() {
    std::vector<Tensor> outputs;
    outputs.reserve(all_output_infos_.size());
    for (const TensorInfo& output : all_output_infos_) {
      TensorDesc desc;
      desc.info = output;
      desc.info.shape = FromTensorRtDims(
          context_->getTensorShape(output.name.c_str()), false);
      desc.device = execution_device_;
      outputs.push_back(Tensor::Allocate(std::move(desc)));
    }
    return outputs;
  }

  void BindOutputs(std::vector<Tensor>& outputs) {
    for (std::size_t index = 0; index < outputs.size(); ++index) {
      const std::string& name = all_output_names_[index];
      if (!context_->setTensorAddress(name.c_str(), outputs[index].data())) {
        throw std::runtime_error("TensorRT setTensorAddress failed: " + name);
      }
    }
  }

  std::vector<Tensor> SelectOutputs(std::vector<Tensor> all_outputs) const {
    std::vector<Tensor> selected_outputs;
    selected_outputs.reserve(return_output_indices_.size());
    for (std::size_t index : return_output_indices_) {
      selected_outputs.push_back(std::move(all_outputs[index]));
    }
    return selected_outputs;
  }

  Device execution_device_;
  TensorRtLogger logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  cudaStream_t stream_ = nullptr;
  std::vector<std::string> input_names_;
  std::vector<std::string> all_output_names_;
  std::vector<TensorInfo> all_output_infos_;
  std::vector<std::size_t> return_output_indices_;
};

class TensorRtBackend final : public IBackend {
 public:
  TensorRtBackend(Model model, Device execution_device,
                  std::vector<std::string> output_names)
      : IBackend(std::move(model), execution_device),
        session_(execution_device, std::move(output_names), mutable_model(),
                 mutable_model().info) {}

  std::vector<Tensor> Infer(const std::vector<Tensor>& inputs) override {
    return session_.Infer(inputs, model_info());
  }

 private:
  TensorRtBackendSession session_;
};

class TensorRtBackendAdapter final : public BackendAdapter {
 public:
  bool Supports(const Model& model, Device execution_device) const override {
    return model.format == ModelFormat::kTensorRT &&
           execution_device.type == DeviceType::kCuda &&
           execution_device.id >= 0;
  }

  BackendPtr Create(Model model, Device execution_device,
                    std::vector<std::string> output_names) const override {
    return std::make_unique<TensorRtBackend>(std::move(model), execution_device,
                                             std::move(output_names));
  }
};

}  // namespace

std::unique_ptr<BackendAdapter> CreateTensorRtBackendAdapter() {
  return std::make_unique<TensorRtBackendAdapter>();
}

}  // namespace mw::infer
