#include <NvInferPlugin.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mw/infer/runtime/backend/backend.h"

namespace mw::infer {
namespace {

#if !defined(MW_INFER_TENSORRT_API_FAMILY) ||    \
    !defined(MW_INFER_TENSORRT_VERSION_MAJOR) || \
    !defined(MW_INFER_TENSORRT_VERSION_MINOR)
#error "TensorRT version macros are required"
#endif

#if MW_INFER_TENSORRT_API_FAMILY != 8 && MW_INFER_TENSORRT_API_FAMILY != 10
#error "Unsupported TensorRT API family"
#endif

constexpr int32_t kDefaultOptimizationProfileIndex = 0;

class TensorRtLogger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* message) noexcept override {
    try {
      const auto logger = spdlog::default_logger();
      if (!logger) {
        return;
      }
      switch (severity) {
        case Severity::kINTERNAL_ERROR:
          logger->critical("[TensorRT] {}", message);
          return;
        case Severity::kERROR:
          logger->error("[TensorRT] {}", message);
          return;
        case Severity::kWARNING:
          logger->warn("[TensorRT] {}", message);
          return;
        case Severity::kINFO:
          logger->info("[TensorRT] {}", message);
          return;
        case Severity::kVERBOSE:
          logger->debug("[TensorRT] {}", message);
          return;
      }
    } catch (...) {
      return;
    }
  }
};

TensorRtLogger& GetTensorRtLogger() {
  static TensorRtLogger logger;
  return logger;
}

void InitializeTensorRtPlugins() {
  static const bool initialized =
      initLibNvInferPlugins(&GetTensorRtLogger(), "");
  if (!initialized) {
    throw std::runtime_error("initLibNvInferPlugins failed");
  }
}

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
#if MW_INFER_TENSORRT_API_FAMILY == 10 ||    \
    (MW_INFER_TENSORRT_VERSION_MAJOR == 8 && \
     MW_INFER_TENSORRT_VERSION_MINOR >= 5)
    case nvinfer1::DataType::kUINT8:
      return DataType::kUInt8;
#endif
#if MW_INFER_TENSORRT_API_FAMILY == 10
    case nvinfer1::DataType::kINT64:
      return DataType::kInt64;
#endif
    case nvinfer1::DataType::kBOOL:
#if MW_INFER_TENSORRT_API_FAMILY == 10 ||    \
    (MW_INFER_TENSORRT_VERSION_MAJOR == 8 && \
     MW_INFER_TENSORRT_VERSION_MINOR >= 6)
    case nvinfer1::DataType::kFP8:
#endif
#if MW_INFER_TENSORRT_API_FAMILY == 10
    case nvinfer1::DataType::kBF16:
    case nvinfer1::DataType::kINT4:
#if MW_INFER_TENSORRT_VERSION_MINOR >= 8
    case nvinfer1::DataType::kFP4:
#endif
#endif
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
#if MW_INFER_TENSORRT_API_FAMILY == 8
    if (shape[index] > std::numeric_limits<int32_t>::max()) {
      throw std::invalid_argument(
          "TensorRT 8 tensor dimension exceeds int32 range");
    }
    dims.d[index] = static_cast<int32_t>(shape[index]);
#else
    dims.d[index] = shape[index];
#endif
  }
  return dims;
}

#if MW_INFER_TENSORRT_API_FAMILY == 8 && defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#elif MW_INFER_TENSORRT_API_FAMILY == 8 && defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

#if MW_INFER_TENSORRT_API_FAMILY == 10

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

#else

void ValidateTensorMemoryLayout(const nvinfer1::ICudaEngine& engine,
                                int32_t binding_index,
                                const std::string& name) {
  const nvinfer1::TensorFormat format = engine.getBindingFormat(binding_index);
  const int32_t vectorized_dim = engine.getBindingVectorizedDim(binding_index);
  const int32_t components_per_element =
      engine.getBindingComponentsPerElement(binding_index);

  if (format == nvinfer1::TensorFormat::kLINEAR && vectorized_dim == -1 &&
      components_per_element <= 1) {
    return;
  }

  const char* format_desc = engine.getBindingFormatDesc(binding_index);
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
                          int32_t binding_index, const std::string& name) {
  if (engine.getLocation(binding_index) != nvinfer1::TensorLocation::kDEVICE) {
    throw std::invalid_argument(
        "TensorRT backend supports device IO tensors only");
  }
  if (engine.isShapeBinding(binding_index)) {
    throw std::invalid_argument(
        "TensorRT backend does not support shape inference IO tensors");
  }
  if (!engine.isExecutionBinding(binding_index)) {
    throw std::invalid_argument(
        "TensorRT backend supports execution IO bindings only");
  }
  ValidateTensorMemoryLayout(engine, binding_index, name);

  TensorInfo info;
  info.name = name;
  info.data_type =
      FromTensorRtDataType(engine.getBindingDataType(binding_index));
  info.shape =
      FromTensorRtDims(engine.getBindingDimensions(binding_index), true);
  return info;
}

ModelProfile ReadProfile(const nvinfer1::ICudaEngine& engine,
                         const std::vector<TensorInfo>& inputs,
                         const std::vector<int32_t>& input_binding_indices,
                         int32_t profile_index) {
  ModelProfile profile;
  profile.name = "profile" + std::to_string(profile_index);
  profile.inputs.reserve(inputs.size());
  for (std::size_t index = 0; index < inputs.size(); ++index) {
    const int32_t binding_index = input_binding_indices[index];
    const nvinfer1::Dims min_shape = engine.getProfileDimensions(
        binding_index, profile_index, nvinfer1::OptProfileSelector::kMIN);
    const nvinfer1::Dims opt_shape = engine.getProfileDimensions(
        binding_index, profile_index, nvinfer1::OptProfileSelector::kOPT);
    const nvinfer1::Dims max_shape = engine.getProfileDimensions(
        binding_index, profile_index, nvinfer1::OptProfileSelector::kMAX);
    if (min_shape.nbDims < 0 || opt_shape.nbDims < 0 || max_shape.nbDims < 0) {
      continue;
    }

    TensorShapeRange range;
    range.name = inputs[index].name;
    range.min_shape = FromTensorRtDims(min_shape, false);
    range.opt_shape = FromTensorRtDims(opt_shape, false);
    range.max_shape = FromTensorRtDims(max_shape, false);
    profile.inputs.push_back(std::move(range));
  }
  return profile;
}

ModelInfo ReadModelInfo(const nvinfer1::ICudaEngine& engine,
                        std::vector<int32_t>* input_binding_indices,
                        std::vector<int32_t>* output_binding_indices) {
  if (engine.hasImplicitBatchDimension()) {
    throw std::invalid_argument(
        "TensorRT 8 backend requires an explicit-batch engine");
  }

  const int32_t profile_count = engine.getNbOptimizationProfiles();
  const int32_t binding_count = engine.getNbBindings();
  if (profile_count <= 0 || binding_count <= 0 ||
      binding_count % profile_count != 0) {
    throw std::invalid_argument("TensorRT engine binding layout is invalid");
  }

  input_binding_indices->clear();
  output_binding_indices->clear();
  ModelInfo info;
  const int32_t bindings_per_profile = binding_count / profile_count;
  for (int32_t binding_index = 0; binding_index < bindings_per_profile;
       ++binding_index) {
    const char* binding_name = engine.getBindingName(binding_index);
    if (binding_name == nullptr || binding_name[0] == '\0') {
      throw std::invalid_argument("TensorRT engine has an unnamed IO binding");
    }

    const std::string name = binding_name;
    if (engine.bindingIsInput(binding_index)) {
      info.inputs.push_back(ReadTensorInfo(engine, binding_index, name));
      input_binding_indices->push_back(binding_index);
    } else {
      info.outputs.push_back(ReadTensorInfo(engine, binding_index, name));
      output_binding_indices->push_back(binding_index);
    }
  }

  if (info.inputs.empty()) {
    throw std::invalid_argument("TensorRT engine has no inputs");
  }
  if (info.outputs.empty()) {
    throw std::invalid_argument("TensorRT engine has no outputs");
  }

  info.profiles.push_back(ReadProfile(engine, info.inputs,
                                      *input_binding_indices,
                                      kDefaultOptimizationProfileIndex));
  return info;
}

#endif

class TensorRtApi {
 public:
  TensorRtApi(const nvinfer1::ICudaEngine& engine,
              nvinfer1::IExecutionContext& context)
      : context_(context) {
#if MW_INFER_TENSORRT_API_FAMILY == 8
    model_info_ = ReadModelInfo(engine, &input_binding_indices_,
                                &output_binding_indices_);
    binding_addresses_.assign(static_cast<std::size_t>(engine.getNbBindings()),
                              nullptr);
#else
    model_info_ = ReadModelInfo(engine);
#endif
  }

  const ModelInfo& model_info() const { return model_info_; }

  void BindInput(std::size_t index, const Tensor& input) {
#if MW_INFER_TENSORRT_API_FAMILY == 8
    const int32_t binding_index = input_binding_indices_[index];
    if (HasDynamicShape(model_info_.inputs[index])) {
      const nvinfer1::Dims dims = ToTensorRtDims(input.shape());
      if (!context_.setBindingDimensions(binding_index, dims)) {
        throw std::invalid_argument(
            "TensorRT input tensor shape is outside "
            "the active optimization profile");
      }
    }
    binding_addresses_[static_cast<std::size_t>(binding_index)] =
        const_cast<void*>(input.data());
#else
    const std::string& name = model_info_.inputs[index].name;
    const nvinfer1::Dims dims = ToTensorRtDims(input.shape());
    if (!context_.setInputShape(name.c_str(), dims)) {
      throw std::invalid_argument(
          "TensorRT input tensor shape is outside "
          "the active optimization profile");
    }
    if (!context_.setInputTensorAddress(name.c_str(), input.data())) {
      throw std::runtime_error("TensorRT setInputTensorAddress failed: " +
                               name);
    }
#endif
  }

  void InferShapes() {
#if MW_INFER_TENSORRT_API_FAMILY == 8
    if (!context_.allInputDimensionsSpecified()) {
      throw std::invalid_argument("TensorRT input dimensions are insufficient");
    }
    if (!context_.allInputShapesSpecified()) {
      throw std::invalid_argument("TensorRT input shapes are insufficient");
    }
#else
    const int32_t missing_count = context_.inferShapes(0, nullptr);
    if (missing_count < 0) {
      throw std::runtime_error("TensorRT inferShapes failed");
    }
    if (missing_count == 0) {
      return;
    }

    std::vector<const char*> missing_names(
        static_cast<std::size_t>(missing_count), nullptr);
    const int32_t reported_count =
        context_.inferShapes(missing_count, missing_names.data());

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
#endif
  }

  std::vector<int64_t> GetOutputShape(std::size_t index) const {
#if MW_INFER_TENSORRT_API_FAMILY == 8
    return FromTensorRtDims(
        context_.getBindingDimensions(output_binding_indices_[index]), false);
#else
    return FromTensorRtDims(
        context_.getTensorShape(model_info_.outputs[index].name.c_str()),
        false);
#endif
  }

  void BindOutput(std::size_t index, void* data) {
#if MW_INFER_TENSORRT_API_FAMILY == 8
    const int32_t binding_index = output_binding_indices_[index];
    binding_addresses_[static_cast<std::size_t>(binding_index)] = data;
#else
    const std::string& name = model_info_.outputs[index].name;
    if (!context_.setTensorAddress(name.c_str(), data)) {
      throw std::runtime_error("TensorRT setTensorAddress failed: " + name);
    }
#endif
  }

  void Enqueue(cudaStream_t stream) {
#if MW_INFER_TENSORRT_API_FAMILY == 8
    if (!context_.enqueueV2(binding_addresses_.data(), stream, nullptr)) {
      throw std::runtime_error("TensorRT enqueueV2 failed");
    }
#else
    if (!context_.enqueueV3(stream)) {
      throw std::runtime_error("TensorRT enqueueV3 failed");
    }
#endif
  }

 private:
  nvinfer1::IExecutionContext& context_;
  ModelInfo model_info_;
#if MW_INFER_TENSORRT_API_FAMILY == 8
  std::vector<int32_t> input_binding_indices_;
  std::vector<int32_t> output_binding_indices_;
  std::vector<void*> binding_addresses_;
#endif
};

#if MW_INFER_TENSORRT_API_FAMILY == 8 && defined(_MSC_VER)
#pragma warning(pop)
#elif MW_INFER_TENSORRT_API_FAMILY == 8 && defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

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
  class InFlightBuffers {
   public:
    InFlightBuffers() {
      CheckCuda(cudaEventCreateWithFlags(&completion_, cudaEventDisableTiming),
                "cudaEventCreateWithFlags");
    }

    InFlightBuffers(const InFlightBuffers&) = delete;
    InFlightBuffers& operator=(const InFlightBuffers&) = delete;

    InFlightBuffers(InFlightBuffers&& other) noexcept
        : completion_(std::exchange(other.completion_, nullptr)),
          bound_inputs_(std::move(other.bound_inputs_)),
          bound_outputs_(std::move(other.bound_outputs_)) {}

    ~InFlightBuffers() {
      if (completion_ != nullptr) {
        static_cast<void>(cudaEventDestroy(completion_));
      }
    }

    void Record(cudaStream_t stream) {
      CheckCuda(cudaEventRecord(completion_, stream), "cudaEventRecord");
    }

    bool IsComplete() const {
      const cudaError_t status = cudaEventQuery(completion_);
      if (status == cudaSuccess) {
        return true;
      }
      if (status == cudaErrorNotReady) {
        return false;
      }
      CheckCuda(status, "cudaEventQuery");
      return false;
    }

    void Retain(std::vector<Tensor> bound_inputs,
                const std::vector<Tensor>& bound_outputs) {
      bound_inputs_ = std::move(bound_inputs);
      bound_outputs_ = bound_outputs;
    }

   private:
    cudaEvent_t completion_ = nullptr;
    std::vector<Tensor> bound_inputs_;
    std::vector<Tensor> bound_outputs_;
  };

 public:
  TensorRtBackendSession(Device execution_device,
                         std::vector<std::string> output_names,
                         const Model& model, ModelInfo& active_info,
                         std::shared_ptr<ExecutionStream> execution_stream)
      : execution_device_(execution_device),
        execution_stream_(std::move(execution_stream)),
        synchronize_after_infer_(!execution_stream_) {
    if (execution_device_.type != DeviceType::kCuda ||
        execution_device_.id < 0) {
      throw std::invalid_argument(
          "TensorRT backend requires a non-negative CUDA device");
    }
    if (!execution_stream_) {
      execution_stream_ = std::make_shared<ExecutionStream>(
          execution_device_, CudaStreamMode::kBlocking);
    }
    const Device stream_device = execution_stream_->device();
    if (stream_device.type != execution_device_.type ||
        stream_device.id != execution_device_.id) {
      throw std::invalid_argument(
          "TensorRT execution stream device does not match backend device");
    }
    CheckCuda(cudaSetDevice(execution_device_.id), "cudaSetDevice");

    InitializeTensorRtPlugins();
    runtime_ =
        CheckTensorRtPtr(nvinfer1::createInferRuntime(GetTensorRtLogger()),
                         "nvinfer1::createInferRuntime");
    engine_ = DeserializeEngine(model);
    context_ = CheckTensorRtPtr(engine_->createExecutionContext(),
                                "ICudaEngine::createExecutionContext");
    api_ = std::make_unique<TensorRtApi>(*engine_, *context_);

    ModelInfo full_info = api_->model_info();
    all_output_infos_ = full_info.outputs;
    active_info = ResolveActiveModelInfo(std::move(full_info), output_names,
                                         &return_output_indices_);

    if (context_->getOptimizationProfile() !=
        kDefaultOptimizationProfileIndex) {
      throw std::runtime_error(
          "TensorRT execution context did not select profile0");
    }
  }

  TensorRtBackendSession(const TensorRtBackendSession&) = delete;
  TensorRtBackendSession& operator=(const TensorRtBackendSession&) = delete;

  ~TensorRtBackendSession() { execution_stream_->SynchronizeNoThrow(); }

  std::vector<Tensor> Infer(const std::vector<Tensor>& inputs,
                            const ModelInfo& model_info,
                            TensorAllocator& allocator) {
    CheckCuda(cudaSetDevice(execution_device_.id), "cudaSetDevice");
    ReapCompletedBuffers();
    std::vector<const Tensor*> ordered_inputs =
        ResolveInputs(inputs, model_info);
    std::vector<Tensor> staged_inputs;
    std::vector<const Tensor*> bound_inputs =
        StageInputs(ordered_inputs, allocator, &staged_inputs);

    BindInputs(bound_inputs);
    InferShapes();
    std::vector<Tensor> all_outputs = AllocateOutputs(allocator);
    BindOutputs(all_outputs);

    std::unique_ptr<InFlightBuffers> in_flight;
    if (!synchronize_after_infer_) {
      in_flight = std::make_unique<InFlightBuffers>();
      std::vector<Tensor> bound_input_owners;
      bound_input_owners.reserve(bound_inputs.size());
      for (const Tensor* input : bound_inputs) {
        bound_input_owners.push_back(*input);
      }
      in_flight->Retain(std::move(bound_input_owners), all_outputs);
    }
    try {
      api_->Enqueue(execution_stream_->cuda_handle());
      if (synchronize_after_infer_) {
        execution_stream_->Synchronize();
        return SelectOutputs(&all_outputs);
      }
      in_flight->Record(execution_stream_->cuda_handle());
      std::vector<Tensor> selected_outputs = SelectOutputs(&all_outputs);
      in_flight_buffers_.push_back(std::move(*in_flight));
      return selected_outputs;
    } catch (...) {
      execution_stream_->SynchronizeNoThrow();
      throw;
    }
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
      const std::vector<const Tensor*>& inputs, TensorAllocator& allocator,
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

      staged_inputs->push_back(input->CopyTo(execution_device_, allocator));
      bound_inputs.push_back(&staged_inputs->back());
    }
    return bound_inputs;
  }

  void BindInputs(const std::vector<const Tensor*>& inputs) {
    for (std::size_t index = 0; index < inputs.size(); ++index) {
      api_->BindInput(index, *inputs[index]);
    }
  }

  void InferShapes() { api_->InferShapes(); }

  std::vector<Tensor> AllocateOutputs(TensorAllocator& allocator) {
    std::vector<Tensor> outputs;
    outputs.reserve(all_output_infos_.size());
    for (std::size_t index = 0; index < all_output_infos_.size(); ++index) {
      TensorDesc desc;
      desc.info = all_output_infos_[index];
      desc.info.shape = api_->GetOutputShape(index);
      desc.device = execution_device_;
      outputs.push_back(Tensor::Allocate(std::move(desc), allocator));
    }
    return outputs;
  }

  void BindOutputs(std::vector<Tensor>& outputs) {
    for (std::size_t index = 0; index < outputs.size(); ++index) {
      api_->BindOutput(index, outputs[index].data());
    }
  }

  void ReapCompletedBuffers() {
    while (!in_flight_buffers_.empty() &&
           in_flight_buffers_.front().IsComplete()) {
      in_flight_buffers_.pop_front();
    }
  }

  std::vector<Tensor> SelectOutputs(
      std::vector<Tensor>* all_outputs) const {
    std::vector<Tensor> selected_outputs;
    selected_outputs.reserve(return_output_indices_.size());
    for (std::size_t index : return_output_indices_) {
      selected_outputs.push_back(std::move(all_outputs->at(index)));
    }
    return selected_outputs;
  }

  Device execution_device_;
  std::shared_ptr<ExecutionStream> execution_stream_;
  bool synchronize_after_infer_ = true;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  std::unique_ptr<TensorRtApi> api_;
  std::vector<TensorInfo> all_output_infos_;
  std::vector<std::size_t> return_output_indices_;
  std::deque<InFlightBuffers> in_flight_buffers_;
};

class TensorRtBackend final : public IBackend {
 public:
  TensorRtBackend(Model model, Device execution_device,
                  std::vector<std::string> output_names,
                  std::shared_ptr<ExecutionStream> execution_stream = nullptr)
      : IBackend(std::move(model), execution_device),
        session_(execution_device, std::move(output_names), mutable_model(),
                 mutable_model().info, std::move(execution_stream)) {}

  std::vector<Tensor> Infer(const std::vector<Tensor>& inputs) override {
    return Infer(inputs, TensorAllocator::Default());
  }

  std::vector<Tensor> Infer(const std::vector<Tensor>& inputs,
                            TensorAllocator& allocator) override {
    return session_.Infer(inputs, model_info(), allocator);
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

  BackendPtr Create(
      Model model, Device execution_device,
      std::vector<std::string> output_names,
      std::shared_ptr<ExecutionStream> execution_stream) const override {
    return std::make_unique<TensorRtBackend>(std::move(model), execution_device,
                                             std::move(output_names),
                                             std::move(execution_stream));
  }
};

}  // namespace

std::unique_ptr<BackendAdapter> CreateTensorRtBackendAdapter() {
  return std::make_unique<TensorRtBackendAdapter>();
}

}  // namespace mw::infer
