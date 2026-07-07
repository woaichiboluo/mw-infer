#include "mw/infer/runtime/backend/onnx_runtime_backend.h"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mw::infer {
namespace {

enum class OnnxBackendKind {
  kCpu,
  kGpu,
};

struct OnnxBackendSessionOptions {
  Model model;
  OnnxBackendKind backend = OnnxBackendKind::kCpu;
  int device_id = 0;
  Device output_device = Device{DeviceType::kCpu, 0};
  std::vector<std::string> output_names;
};

void ValidateModel(const Model& model) {
  if (model.format != ModelFormat::kOnnx) {
    throw std::invalid_argument("ONNX Runtime backend requires an ONNX model");
  }

  if (model.source.kind == ModelSourceKind::kPath) {
    if (model.source.path.empty()) {
      throw std::invalid_argument("ONNX model path is empty");
    }
    if (!std::filesystem::exists(model.source.path)) {
      throw std::invalid_argument("ONNX model path does not exist: " +
                                  model.source.path.string());
    }
    return;
  }

  if (model.source.data == nullptr || model.source.bytes == 0) {
    throw std::invalid_argument("ONNX model memory source is empty");
  }
}

DataType FromOrtDataType(ONNXTensorElementDataType data_type) {
  switch (data_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return DataType::kUInt8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      return DataType::kInt8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      return DataType::kUInt16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      return DataType::kInt16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return DataType::kInt32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return DataType::kInt64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return DataType::kFloat16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return DataType::kFloat32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      return DataType::kFloat64;
    default:
      throw std::invalid_argument("Unsupported ONNX tensor data type");
  }
}

ONNXTensorElementDataType ToOrtDataType(DataType data_type) {
  switch (data_type) {
    case DataType::kUInt8:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    case DataType::kInt8:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
    case DataType::kUInt16:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
    case DataType::kInt16:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
    case DataType::kInt32:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    case DataType::kInt64:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    case DataType::kFloat16:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    case DataType::kFloat32:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    case DataType::kFloat64:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
    case DataType::kUnknown:
      throw std::invalid_argument("Tensor data type is unknown");
  }
  throw std::invalid_argument("Tensor data type is unknown");
}

std::size_t RuntimeElementCount(const std::vector<int64_t>& shape) {
  std::size_t count = 1;
  for (int64_t dim : shape) {
    if (dim <= 0) {
      throw std::runtime_error("ONNX Runtime returned a dynamic output shape");
    }
    count *= static_cast<std::size_t>(dim);
  }
  return count;
}

std::size_t RuntimeTensorBytes(const TensorDesc& desc) {
  return RuntimeElementCount(desc.shape) * DataTypeSize(desc.data_type);
}

Ort::MemoryInfo MakeMemoryInfo(Device device) {
  if (device.type == DeviceType::kCpu) {
    return Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  }
  if (device.type == DeviceType::kCuda) {
    return Ort::MemoryInfo("Cuda", OrtDeviceAllocator, device.id,
                           OrtMemTypeDefault);
  }

  throw std::invalid_argument("Unsupported tensor device");
}

Device DeviceFromOrtValue(const Ort::Value& value) {
  const Ort::ConstMemoryInfo memory_info = value.GetTensorMemoryInfo();
  if (memory_info.GetDeviceType() == OrtMemoryInfoDeviceType_GPU) {
    return Device{DeviceType::kCuda, memory_info.GetDeviceId()};
  }
  return Device{DeviceType::kCpu, 0};
}

bool IsOnnxGpuBackendAvailableImpl() {
#if defined(MW_INFER_HAS_ONNXRUNTIME_CUDA_PROVIDER)
  try {
    const std::vector<std::string> providers = Ort::GetAvailableProviders();
    return std::find(providers.begin(), providers.end(),
                     "CUDAExecutionProvider") != providers.end();
  } catch (const Ort::Exception&) {
    return false;
  }
#else
  return false;
#endif
}

Ort::SessionOptions MakeSessionOptions(
    const OnnxBackendSessionOptions& options) {
  Ort::SessionOptions session_options;
  session_options.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

  if (options.backend == OnnxBackendKind::kGpu) {
#if defined(MW_INFER_HAS_ONNXRUNTIME_CUDA_PROVIDER)
    Ort::CUDAProviderOptions cuda_options;
    cuda_options.Update(std::unordered_map<std::string, std::string>{
        {"device_id", std::to_string(options.device_id)}});
    session_options.AppendExecutionProvider_CUDA_V2(*cuda_options);
#else
    throw std::runtime_error(
        "ONNX Runtime CUDA provider is unavailable at build time");
#endif
  }

  return session_options;
}

Ort::Session CreateSession(const Ort::Env& env, const Model& model,
                           const Ort::SessionOptions& options) {
  if (model.source.kind == ModelSourceKind::kMemory) {
    return Ort::Session(env, model.source.data, model.source.bytes, options);
  }

#ifdef _WIN32
  const std::wstring path = model.source.path.wstring();
#else
  const std::string path = model.source.path.string();
#endif
  return Ort::Session(env, path.c_str(), options);
}

std::string ReadName(const Ort::Session& session, bool input,
                     std::size_t index) {
  Ort::AllocatorWithDefaultOptions allocator;
  auto name = input ? session.GetInputNameAllocated(index, allocator)
                    : session.GetOutputNameAllocated(index, allocator);
  return name.get();
}

TensorInfo ReadTensorInfo(const Ort::Session& session, bool input,
                          std::size_t index) {
  TensorInfo info;
  info.name = ReadName(session, input, index);

  Ort::TypeInfo type_info = input ? session.GetInputTypeInfo(index)
                                  : session.GetOutputTypeInfo(index);
  Ort::ConstTensorTypeAndShapeInfo tensor_info =
      type_info.GetTensorTypeAndShapeInfo();
  info.data_type = FromOrtDataType(tensor_info.GetElementType());
  info.shape = tensor_info.GetShape();
  return info;
}

ModelInfo ReadModelInfo(const Ort::Session& session) {
  ModelInfo info;

  const std::size_t input_count = session.GetInputCount();
  if (input_count == 0) {
    throw std::invalid_argument("ONNX model has no inputs");
  }
  info.inputs.reserve(input_count);
  for (std::size_t index = 0; index < input_count; ++index) {
    info.inputs.push_back(ReadTensorInfo(session, true, index));
  }

  const std::size_t output_count = session.GetOutputCount();
  if (output_count == 0) {
    throw std::invalid_argument("ONNX model has no outputs");
  }
  info.outputs.reserve(output_count);
  for (std::size_t index = 0; index < output_count; ++index) {
    info.outputs.push_back(ReadTensorInfo(session, false, index));
  }

  return info;
}

std::size_t FindTensorInfo(const std::vector<TensorInfo>& infos,
                           const std::string& name, const char* kind) {
  const auto iter = std::find_if(
      infos.begin(), infos.end(),
      [&name](const TensorInfo& info) { return info.name == name; });
  if (iter == infos.end()) {
    throw std::invalid_argument(std::string("Unknown ONNX ") + kind +
                                " name: " + name);
  }
  return static_cast<std::size_t>(std::distance(infos.begin(), iter));
}

std::vector<std::string> ResolveOutputNames(
    const ModelInfo& model_info, const std::vector<std::string>& requested) {
  if (requested.empty()) {
    std::vector<std::string> names;
    names.reserve(model_info.outputs.size());
    for (const TensorInfo& output : model_info.outputs) {
      names.push_back(output.name);
    }
    return names;
  }

  for (const std::string& name : requested) {
    FindTensorInfo(model_info.outputs, name, "output");
  }
  return requested;
}

std::vector<const char*> MakeNamePointers(
    const std::vector<std::string>& names) {
  std::vector<const char*> pointers;
  pointers.reserve(names.size());
  for (const std::string& name : names) {
    pointers.push_back(name.c_str());
  }
  return pointers;
}

void ValidateOptions(const OnnxBackendSessionOptions& options) {
  ValidateModel(options.model);
  if (options.device_id < 0) {
    throw std::invalid_argument("ONNX Runtime CUDA device id is negative");
  }
  if (options.backend == OnnxBackendKind::kGpu &&
      !IsOnnxGpuBackendAvailableImpl()) {
    throw std::runtime_error(
        "ONNX Runtime GPU backend requires CUDA provider, but this ONNX "
        "Runtime build does not provide it");
  }
  if (options.backend == OnnxBackendKind::kCpu &&
      options.output_device.type == DeviceType::kCuda) {
    throw std::invalid_argument(
        "ONNX Runtime CPU backend cannot produce CUDA output tensors");
  }
  if (options.output_device.type == DeviceType::kCuda &&
      options.output_device.id != options.device_id) {
    throw std::invalid_argument(
        "ONNX Runtime CUDA output device must match backend device id");
  }
}

void ValidateInputTensor(const Tensor& tensor, const TensorInfo& expected,
                         const OnnxBackendSessionOptions& options) {
  if (tensor.empty()) {
    throw std::invalid_argument("ONNX input tensor is empty");
  }
  if (tensor.data_type() != expected.data_type) {
    throw std::invalid_argument("ONNX input tensor data type mismatch");
  }
  if (tensor.shape().size() != expected.shape.size()) {
    throw std::invalid_argument("ONNX input tensor rank mismatch");
  }
  for (std::size_t dim = 0; dim < expected.shape.size(); ++dim) {
    if (expected.shape[dim] > 0 && expected.shape[dim] != tensor.shape()[dim]) {
      throw std::invalid_argument("ONNX input tensor shape mismatch");
    }
  }

  if (tensor.device().type == DeviceType::kCuda) {
    if (options.backend != OnnxBackendKind::kGpu) {
      throw std::invalid_argument(
          "CUDA input tensor requires ONNX Runtime GPU backend");
    }
    if (tensor.device().id != options.device_id) {
      throw std::invalid_argument(
          "CUDA input tensor device id must match backend device id");
    }
    return;
  }

  if (tensor.device().type != DeviceType::kCpu) {
    throw std::invalid_argument("Unsupported ONNX input tensor device");
  }
}

bool NeedsIoBinding(const std::vector<const Tensor*>& inputs,
                    Device output_device) {
  if (output_device.type == DeviceType::kCuda) {
    return true;
  }
  for (const Tensor* input : inputs) {
    if (input->device().type == DeviceType::kCuda) {
      return true;
    }
  }
  return false;
}

class OnnxBackendSession {
 public:
  explicit OnnxBackendSession(OnnxBackendSessionOptions options)
      : options_(std::move(options)),
        env_(ORT_LOGGING_LEVEL_WARNING, "MwInfer"),
        session_(nullptr) {
    ValidateOptions(options_);
    Ort::SessionOptions session_options = MakeSessionOptions(options_);
    session_ = CreateSession(env_, options_.model, session_options);
    model_info_ = ReadModelInfo(session_);
    output_names_ = ResolveOutputNames(model_info_, options_.output_names);
    input_names_.reserve(model_info_.inputs.size());
    for (const TensorInfo& input : model_info_.inputs) {
      input_names_.push_back(input.name);
    }
    input_name_ptrs_ = MakeNamePointers(input_names_);
    output_name_ptrs_ = MakeNamePointers(output_names_);
  }

  const ModelInfo& model_info() const { return model_info_; }

  std::vector<Tensor> Infer(const std::vector<Tensor>& inputs) {
    std::vector<const Tensor*> ordered_inputs = ResolveInputs(inputs);
    std::vector<Ort::Value> input_values = MakeOrtInputs(ordered_inputs);

    if (NeedsIoBinding(ordered_inputs, options_.output_device)) {
      return InferWithIoBinding(ordered_inputs, std::move(input_values));
    }

    Ort::RunOptions run_options;
    std::vector<Ort::Value> output_values =
        session_.Run(run_options, input_name_ptrs_.data(), input_values.data(),
                     input_values.size(), output_name_ptrs_.data(),
                     output_name_ptrs_.size());
    return WrapOutputs(std::move(output_values));
  }

 private:
  std::vector<const Tensor*> ResolveInputs(
      const std::vector<Tensor>& inputs) const {
    if (inputs.size() != model_info_.inputs.size()) {
      throw std::invalid_argument("ONNX input tensor count mismatch");
    }

    std::vector<const Tensor*> ordered(model_info_.inputs.size(), nullptr);
    if (inputs.size() == 1 && inputs.front().name().empty()) {
      ValidateInputTensor(inputs.front(), model_info_.inputs.front(), options_);
      ordered.front() = &inputs.front();
      return ordered;
    }

    std::vector<bool> assigned(model_info_.inputs.size(), false);
    for (const Tensor& input : inputs) {
      if (input.name().empty()) {
        throw std::invalid_argument(
            "ONNX input tensor name is required for multi-input models");
      }
      const std::size_t index =
          FindTensorInfo(model_info_.inputs, input.name(), "input");
      if (assigned[index]) {
        throw std::invalid_argument("Duplicate ONNX input tensor name: " +
                                    input.name());
      }
      ValidateInputTensor(input, model_info_.inputs[index], options_);
      ordered[index] = &input;
      assigned[index] = true;
    }

    return ordered;
  }

  std::vector<Ort::Value> MakeOrtInputs(
      const std::vector<const Tensor*>& inputs) const {
    std::vector<Ort::Value> values;
    values.reserve(inputs.size());
    for (const Tensor* input : inputs) {
      Ort::MemoryInfo memory_info = MakeMemoryInfo(input->device());
      values.push_back(Ort::Value::CreateTensor(
          memory_info, const_cast<void*>(input->data()), input->bytes(),
          input->shape().data(), input->shape().size(),
          ToOrtDataType(input->data_type())));
    }
    return values;
  }

  std::vector<Tensor> InferWithIoBinding(
      const std::vector<const Tensor*>& inputs,
      std::vector<Ort::Value> input_values) {
    Ort::IoBinding binding(session_);
    for (std::size_t index = 0; index < inputs.size(); ++index) {
      binding.BindInput(input_names_[index].c_str(), input_values[index]);
    }

    Ort::MemoryInfo output_memory_info = MakeMemoryInfo(options_.output_device);
    for (const std::string& output_name : output_names_) {
      binding.BindOutput(output_name.c_str(), output_memory_info);
    }

    Ort::RunOptions run_options;
    binding.SynchronizeInputs();
    session_.Run(run_options, binding);
    binding.SynchronizeOutputs();

    std::vector<Ort::Value> output_values = binding.GetOutputValues();
    return WrapOutputs(std::move(output_values));
  }

  std::vector<Tensor> WrapOutputs(std::vector<Ort::Value> output_values) const {
    if (output_values.size() != output_names_.size()) {
      throw std::runtime_error("ONNX Runtime output count mismatch");
    }

    std::vector<Tensor> outputs;
    outputs.reserve(output_values.size());
    for (std::size_t index = 0; index < output_values.size(); ++index) {
      if (!output_values[index].IsTensor()) {
        throw std::runtime_error("ONNX Runtime returned a non-tensor output");
      }

      auto output_value =
          std::make_shared<Ort::Value>(std::move(output_values[index]));
      Ort::TensorTypeAndShapeInfo tensor_info =
          output_value->GetTensorTypeAndShapeInfo();

      TensorDesc desc;
      desc.name = output_names_[index];
      desc.data_type = FromOrtDataType(tensor_info.GetElementType());
      desc.shape = tensor_info.GetShape();
      desc.device = DeviceFromOrtValue(*output_value);

      const std::size_t bytes = RuntimeTensorBytes(desc);
      void* data = output_value->GetTensorMutableRawData();
      outputs.push_back(Tensor::FromExternal(
          std::move(desc), data, bytes, std::shared_ptr<void>(output_value)));
    }

    return outputs;
  }

  OnnxBackendSessionOptions options_;
  Ort::Env env_;
  Ort::Session session_;
  ModelInfo model_info_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<const char*> input_name_ptrs_;
  std::vector<const char*> output_name_ptrs_;
};

OnnxBackendSessionOptions MakeCpuOptions(OnnxCpuBackendOptions options) {
  OnnxBackendSessionOptions session_options;
  session_options.model = std::move(options.model);
  session_options.backend = OnnxBackendKind::kCpu;
  session_options.output_device = Device{DeviceType::kCpu, 0};
  session_options.output_names = std::move(options.output_names);
  return session_options;
}

OnnxBackendSessionOptions MakeGpuOptions(OnnxGpuBackendOptions options) {
  OnnxBackendSessionOptions session_options;
  session_options.model = std::move(options.model);
  session_options.backend = OnnxBackendKind::kGpu;
  session_options.device_id = options.device_id;
  session_options.output_device = options.output_device;
  session_options.output_names = std::move(options.output_names);
  return session_options;
}

}  // namespace

class OnnxCpuBackend::Impl {
 public:
  explicit Impl(OnnxCpuBackendOptions options)
      : session_(MakeCpuOptions(std::move(options))) {}

  const ModelInfo& model_info() const { return session_.model_info(); }

  std::vector<Tensor> Infer(const std::vector<Tensor>& inputs) {
    return session_.Infer(inputs);
  }

 private:
  OnnxBackendSession session_;
};

class OnnxGpuBackend::Impl {
 public:
  explicit Impl(OnnxGpuBackendOptions options)
      : session_(MakeGpuOptions(std::move(options))) {}

  const ModelInfo& model_info() const { return session_.model_info(); }

  std::vector<Tensor> Infer(const std::vector<Tensor>& inputs) {
    return session_.Infer(inputs);
  }

 private:
  OnnxBackendSession session_;
};

OnnxCpuBackend::OnnxCpuBackend(OnnxCpuBackendOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {}

OnnxCpuBackend::~OnnxCpuBackend() = default;

OnnxCpuBackend::OnnxCpuBackend(OnnxCpuBackend&&) noexcept = default;

OnnxCpuBackend& OnnxCpuBackend::operator=(OnnxCpuBackend&&) noexcept = default;

const ModelInfo& OnnxCpuBackend::model_info() const {
  return impl_->model_info();
}

std::vector<Tensor> OnnxCpuBackend::Infer(const std::vector<Tensor>& inputs) {
  return impl_->Infer(inputs);
}

OnnxGpuBackend::OnnxGpuBackend(OnnxGpuBackendOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {}

OnnxGpuBackend::~OnnxGpuBackend() = default;

OnnxGpuBackend::OnnxGpuBackend(OnnxGpuBackend&&) noexcept = default;

OnnxGpuBackend& OnnxGpuBackend::operator=(OnnxGpuBackend&&) noexcept = default;

const ModelInfo& OnnxGpuBackend::model_info() const {
  return impl_->model_info();
}

std::vector<Tensor> OnnxGpuBackend::Infer(const std::vector<Tensor>& inputs) {
  return impl_->Infer(inputs);
}

bool IsOnnxGpuBackendAvailable() { return IsOnnxGpuBackendAvailableImpl(); }

BackendPtr CreateOnnxCpuBackend(Model model) {
  OnnxCpuBackendOptions options;
  options.model = std::move(model);
  return std::make_unique<OnnxCpuBackend>(std::move(options));
}

BackendPtr CreateOnnxGpuBackend(Model model, int device_id) {
  OnnxGpuBackendOptions options;
  options.model = std::move(model);
  options.device_id = device_id;
  options.output_device = Device{DeviceType::kCpu, 0};
  return std::make_unique<OnnxGpuBackend>(std::move(options));
}

}  // namespace mw::infer
