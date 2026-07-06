#include "runtime/backends/onnx_runtime_session.h"

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iterator>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#if MW_INFER_WITH_OPENCV_CUDA
#include <opencv2/cudaarithm.hpp>
#endif

#if MW_INFER_WITH_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace mw::infer {
namespace {

struct MatBatchShape {
  std::size_t batch_size = 0;
  int channels = 0;
  int height = 0;
  int width = 0;
  int type = 0;
};

void ValidateOnnxConfig(const RuntimeConfig& config) {
  if (config.model.format != ModelFormat::kOnnx) {
    throw std::invalid_argument("ONNX backend requires an ONNX model");
  }
  if (config.model.source.type == ModelSourceType::kPath) {
    const auto& path = config.model.source.path;
    if (path.empty()) {
      throw std::invalid_argument("ONNX model path is empty");
    }
    if (!std::filesystem::exists(path)) {
      throw std::invalid_argument("ONNX model path does not exist: " +
                                  path.string());
    }
  } else if (config.model.source.memory.empty()) {
    throw std::invalid_argument("ONNX model memory source is empty");
  }
}

MatBatchShape ValidateMatBatch(const std::vector<cv::Mat>& inputs) {
  if (inputs.empty()) {
    throw std::invalid_argument("ONNX inference input batch is empty");
  }

  const cv::Mat& first = inputs.front();
  if (first.empty()) {
    throw std::invalid_argument("ONNX inference input image is empty");
  }
  if (first.dims != 2) {
    throw std::invalid_argument("ONNX inference expects 2D OpenCV Mat images");
  }

  MatBatchShape shape;
  shape.batch_size = inputs.size();
  shape.channels = first.channels();
  shape.height = first.rows;
  shape.width = first.cols;
  shape.type = first.type();

  for (const cv::Mat& input : inputs) {
    if (input.empty()) {
      throw std::invalid_argument("ONNX inference input image is empty");
    }
    if (input.dims != 2) {
      throw std::invalid_argument(
          "ONNX inference expects 2D OpenCV Mat images");
    }
    if (input.type() != shape.type || input.channels() != shape.channels ||
        input.rows != shape.height || input.cols != shape.width) {
      throw std::invalid_argument(
          "all images in one ONNX inference batch must have the same C/H/W "
          "and OpenCV type");
    }
  }

  return shape;
}

MatBatchShape ValidateGpuMatBatch(const std::vector<cv::cuda::GpuMat>& inputs) {
  if (inputs.empty()) {
    throw std::invalid_argument("ONNX inference input GPU batch is empty");
  }

  const cv::cuda::GpuMat& first = inputs.front();
  if (first.empty()) {
    throw std::invalid_argument("ONNX inference input GPU image is empty");
  }

  MatBatchShape shape;
  shape.batch_size = inputs.size();
  shape.channels = first.channels();
  shape.height = first.rows;
  shape.width = first.cols;
  shape.type = first.type();

  for (const cv::cuda::GpuMat& input : inputs) {
    if (input.empty()) {
      throw std::invalid_argument("ONNX inference input GPU image is empty");
    }
    if (input.type() != shape.type || input.channels() != shape.channels ||
        input.rows != shape.height || input.cols != shape.width) {
      throw std::invalid_argument(
          "all GPU images in one ONNX inference batch must have the same "
          "C/H/W and OpenCV type");
    }
  }

  return shape;
}

std::size_t ElementCount(const Shape& shape) {
  if (shape.empty()) {
    return 1;
  }

  std::size_t count = 1;
  for (int64_t dim : shape) {
    if (dim < 0) {
      throw std::runtime_error("ONNX Runtime returned a dynamic output shape");
    }
    count *= static_cast<std::size_t>(dim);
  }
  return count;
}

#if MW_INFER_WITH_ONNXRUNTIME

InferDataType ConvertElementType(ONNXTensorElementDataType type) {
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return InferDataType::kFloat32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return InferDataType::kFloat16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      return InferDataType::kInt8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return InferDataType::kUint8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return InferDataType::kInt32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return InferDataType::kInt64;
    default:
      throw std::runtime_error("unsupported ONNX output element type");
  }
}

std::size_t ElementSize(ONNXTensorElementDataType type) {
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return sizeof(float);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return sizeof(uint16_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      return sizeof(int8_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return sizeof(uint8_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return sizeof(int32_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return sizeof(int64_t);
    default:
      throw std::runtime_error("unsupported ONNX output element type");
  }
}

Ort::SessionOptions MakeSessionOptions(bool use_cuda_provider) {
  Ort::SessionOptions options;
  options.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

  if (use_cuda_provider) {
#if MW_INFER_WITH_ONNXRUNTIME_CUDA
    Ort::CUDAProviderOptions cuda_options;
    cuda_options.Update(std::unordered_map<std::string, std::string>{
        {"device_id", "0"},
    });
    options.AppendExecutionProvider_CUDA_V2(*cuda_options);
#else
    throw std::runtime_error(
        "ONNX Runtime CUDA provider is disabled at build time");
#endif
  }

  return options;
}

Ort::Session CreateOrtSession(const Ort::Env& env, const Model& model,
                              const Ort::SessionOptions& options) {
  if (model.source.type == ModelSourceType::kMemory) {
    return Ort::Session(env, model.source.memory.data(),
                        model.source.memory.size_bytes(), options);
  }

#ifdef _WIN32
  const std::wstring model_path = model.source.path.wstring();
#else
  const std::string model_path = model.source.path.string();
#endif
  return Ort::Session(env, model_path.c_str(), options);
}

std::vector<std::string> ReadNames(const Ort::Session& session,
                                   bool read_inputs) {
  Ort::AllocatorWithDefaultOptions allocator;
  const std::size_t count =
      read_inputs ? session.GetInputCount() : session.GetOutputCount();
  std::vector<std::string> names;
  names.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    auto name = read_inputs ? session.GetInputNameAllocated(index, allocator)
                            : session.GetOutputNameAllocated(index, allocator);
    names.emplace_back(name.get());
  }
  return names;
}

std::size_t FindName(const std::vector<std::string>& names,
                     const std::string& name, const char* kind) {
  const auto iter = std::find(names.begin(), names.end(), name);
  if (iter == names.end()) {
    throw std::invalid_argument(std::string("unknown ONNX ") + kind +
                                " name: " + name);
  }
  return static_cast<std::size_t>(std::distance(names.begin(), iter));
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

int64_t ResolveDim(int64_t model_dim, std::size_t actual_dim,
                   const char* dim_name) {
  if (model_dim > 0 && model_dim != static_cast<int64_t>(actual_dim)) {
    throw std::invalid_argument(std::string("ONNX input ") + dim_name +
                                " mismatch");
  }
  return static_cast<int64_t>(actual_dim);
}

void CopyHwcBatchToNchw(const std::vector<cv::Mat>& inputs,
                        const MatBatchShape& shape,
                        std::vector<float>* buffer) {
  const std::size_t image_elements = static_cast<std::size_t>(shape.channels) *
                                     static_cast<std::size_t>(shape.height) *
                                     static_cast<std::size_t>(shape.width);
  buffer->resize(shape.batch_size * image_elements);

  for (std::size_t batch_index = 0; batch_index < inputs.size();
       ++batch_index) {
    cv::Mat float_image;
    if (inputs[batch_index].depth() == CV_32F) {
      float_image = inputs[batch_index];
    } else {
      inputs[batch_index].convertTo(float_image, CV_32FC(shape.channels));
    }

    std::vector<cv::Mat> chw;
    chw.reserve(static_cast<std::size_t>(shape.channels));
    for (int channel = 0; channel < shape.channels; ++channel) {
      float* data = buffer->data() + batch_index * image_elements +
                    static_cast<std::size_t>(channel) *
                        static_cast<std::size_t>(shape.height) *
                        static_cast<std::size_t>(shape.width);
      chw.emplace_back(shape.height, shape.width, CV_32FC1, data);
    }
    cv::split(float_image, chw);
  }
}

#if MW_INFER_WITH_OPENCV_CUDA

void CopyHwcGpuBatchToNchw(const std::vector<cv::cuda::GpuMat>& inputs,
                           const MatBatchShape& shape,
                           cv::cuda::GpuMat* buffer) {
  const std::size_t image_elements = static_cast<std::size_t>(shape.channels) *
                                     static_cast<std::size_t>(shape.height) *
                                     static_cast<std::size_t>(shape.width);
  const std::size_t total_elements = shape.batch_size * image_elements;
  if (total_elements > static_cast<std::size_t>(INT_MAX)) {
    throw std::invalid_argument("ONNX GPU input batch is too large");
  }

  buffer->create(1, static_cast<int>(total_elements), CV_32FC1);

  std::vector<cv::cuda::GpuMat> converted_inputs(inputs.size());
  for (std::size_t batch_index = 0; batch_index < inputs.size();
       ++batch_index) {
    const cv::cuda::GpuMat* float_image = &inputs[batch_index];
    if (inputs[batch_index].depth() != CV_32F) {
      inputs[batch_index].convertTo(converted_inputs[batch_index],
                                    CV_32FC(shape.channels));
      float_image = &converted_inputs[batch_index];
    }

    std::vector<cv::cuda::GpuMat> chw;
    chw.reserve(static_cast<std::size_t>(shape.channels));
    for (int channel = 0; channel < shape.channels; ++channel) {
      float* data = buffer->ptr<float>() + batch_index * image_elements +
                    static_cast<std::size_t>(channel) *
                        static_cast<std::size_t>(shape.height) *
                        static_cast<std::size_t>(shape.width);
      chw.emplace_back(shape.height, shape.width, CV_32FC1, data,
                       static_cast<std::size_t>(shape.width) * sizeof(float));
    }
    cv::cuda::split(*float_image, chw);
  }

  cv::cuda::Stream::Null().waitForCompletion();
}

#endif  // MW_INFER_WITH_OPENCV_CUDA

Device OutputDevice(const Ort::Value& value) {
  const Ort::ConstMemoryInfo memory_info = value.GetTensorMemoryInfo();
  if (memory_info.GetDeviceType() == OrtMemoryInfoDeviceType_GPU) {
    return Device::Cuda(memory_info.GetDeviceId());
  }
  return Device::Cpu();
}

#endif  // MW_INFER_WITH_ONNXRUNTIME

}  // namespace

#if MW_INFER_WITH_ONNXRUNTIME

class OnnxRuntimeSession::Impl {
 public:
  Impl(RuntimeConfig config, bool use_cuda_provider)
      : config_(std::move(config)),
        use_cuda_provider_(use_cuda_provider),
        env_(ORT_LOGGING_LEVEL_WARNING, "MwInfer"),
        memory_info_(
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
        cuda_memory_info_("Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault),
        session_(nullptr) {
    ValidateOnnxConfig(config_);
    Ort::SessionOptions options = MakeSessionOptions(use_cuda_provider_);
    session_ = CreateOrtSession(env_, config_.model, options);
    LoadIoMetadata();
  }

  const RuntimeConfig& config() const { return config_; }

  const InferOutputs& Infer(const std::vector<cv::Mat>& inputs) {
    const MatBatchShape batch_shape = ValidateMatBatch(inputs);
    const Shape input_shape = MakeInputShape(batch_shape);
    CopyHwcBatchToNchw(inputs, batch_shape, &input_buffer_);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_, input_buffer_.data(), input_buffer_.size(),
        input_shape.data(), input_shape.size());

    Ort::RunOptions run_options;
    output_values_ =
        session_.Run(run_options, input_name_ptrs_.data(), &input_tensor, 1,
                     output_name_ptrs_.data(), output_name_ptrs_.size());
    return BuildInferOutputs(batch_shape.batch_size);
  }

  const InferOutputs& Infer(const std::vector<cv::cuda::GpuMat>& inputs) {
#if MW_INFER_WITH_OPENCV_CUDA
    if (!use_cuda_provider_) {
      throw std::invalid_argument(
          "cv::cuda::GpuMat batch input requires ONNX Runtime CUDA provider");
    }

    const MatBatchShape batch_shape = ValidateGpuMatBatch(inputs);
    const Shape input_shape = MakeInputShape(batch_shape);
    CopyHwcGpuBatchToNchw(inputs, batch_shape, &gpu_input_buffer_);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        cuda_memory_info_, gpu_input_buffer_.ptr<float>(),
        gpu_input_buffer_.cols, input_shape.data(), input_shape.size());

    Ort::IoBinding binding(session_);
    binding.BindInput(input_name_ptrs_[0], input_tensor);
    for (const char* output_name : output_name_ptrs_) {
      binding.BindOutput(output_name, cuda_memory_info_);
    }

    Ort::RunOptions run_options;
    binding.SynchronizeInputs();
    session_.Run(run_options, binding);
    binding.SynchronizeOutputs();

    output_values_ = binding.GetOutputValues();
    return BuildInferOutputs(batch_shape.batch_size);
#else
    throw std::invalid_argument(
        "cv::cuda::GpuMat batch input requires OpenCV CUDA modules");
#endif
  }

 private:
  void LoadIoMetadata() {
    std::vector<std::string> available_inputs = ReadNames(session_, true);
    if (available_inputs.size() != 1) {
      throw std::invalid_argument("ONNX backend currently supports one input");
    }

    const std::size_t input_index =
        config_.input_name.empty()
            ? 0
            : FindName(available_inputs, config_.input_name, "input");
    input_names_ = {available_inputs[input_index]};

    auto input_type_info = session_.GetInputTypeInfo(input_index);
    auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    const ONNXTensorElementDataType input_element_type =
        input_tensor_info.GetElementType();
    if (input_element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      throw std::invalid_argument(
          "ONNX backend currently supports float32 model inputs");
    }
    model_input_shape_ = input_tensor_info.GetShape();
    if (model_input_shape_.size() != 4) {
      throw std::invalid_argument(
          "ONNX backend currently expects a 4D NCHW model input");
    }

    std::vector<std::string> available_outputs = ReadNames(session_, false);
    if (config_.output_names.empty()) {
      output_names_ = std::move(available_outputs);
    } else {
      for (const std::string& output_name : config_.output_names) {
        FindName(available_outputs, output_name, "output");
      }
      output_names_ = config_.output_names;
    }

    input_name_ptrs_ = MakeNamePointers(input_names_);
    output_name_ptrs_ = MakeNamePointers(output_names_);
  }

  Shape MakeInputShape(const MatBatchShape& batch_shape) const {
    Shape shape = model_input_shape_;
    shape[0] = ResolveDim(shape[0], batch_shape.batch_size, "batch");
    shape[1] = ResolveDim(
        shape[1], static_cast<std::size_t>(batch_shape.channels), "channel");
    shape[2] = ResolveDim(
        shape[2], static_cast<std::size_t>(batch_shape.height), "height");
    shape[3] = ResolveDim(shape[3], static_cast<std::size_t>(batch_shape.width),
                          "width");
    return shape;
  }

  const InferOutputs& BuildInferOutputs(std::size_t batch_size) {
    InferOutputs outputs;
    outputs.batch_size = static_cast<int>(batch_size);
    outputs.outputs.reserve(output_values_.size());

    for (std::size_t index = 0; index < output_values_.size(); ++index) {
      Ort::Value& value = output_values_[index];
      auto tensor_info = value.GetTensorTypeAndShapeInfo();
      const ONNXTensorElementDataType element_type =
          tensor_info.GetElementType();
      Shape shape = tensor_info.GetShape();
      void* data = value.GetTensorMutableRawData();
      const Device device = OutputDevice(value);

      InferOutput output;
      output.name = output_names_[index];
      output.data_type = ConvertElementType(element_type);
      output.shape = shape;
      output.device = device;
      if (device.is_host()) {
        output.buffer.host = data;
      } else {
        output.buffer.device = data;
      }
      output.buffer.size_bytes =
          ElementCount(shape) * ElementSize(element_type);
      outputs.outputs.push_back(std::move(output));
    }

    outputs_ = std::move(outputs);
    return outputs_;
  }

  RuntimeConfig config_;
  bool use_cuda_provider_ = false;
  Ort::Env env_;
  Ort::MemoryInfo memory_info_;
  Ort::MemoryInfo cuda_memory_info_;
  Ort::Session session_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<const char*> input_name_ptrs_;
  std::vector<const char*> output_name_ptrs_;
  Shape model_input_shape_;
  std::vector<float> input_buffer_;
  cv::cuda::GpuMat gpu_input_buffer_;
  std::vector<Ort::Value> output_values_;
  InferOutputs outputs_;
};

OnnxRuntimeSession::OnnxRuntimeSession(RuntimeConfig config,
                                       bool use_cuda_provider)
    : impl_(std::make_unique<Impl>(std::move(config), use_cuda_provider)) {}

OnnxRuntimeSession::~OnnxRuntimeSession() = default;

const RuntimeConfig& OnnxRuntimeSession::config() const {
  return impl_->config();
}

const InferOutputs& OnnxRuntimeSession::Infer(
    const std::vector<cv::Mat>& inputs) {
  return impl_->Infer(inputs);
}

const InferOutputs& OnnxRuntimeSession::Infer(
    const std::vector<cv::cuda::GpuMat>& inputs) {
  return impl_->Infer(inputs);
}

#else

class OnnxRuntimeSession::Impl {};

OnnxRuntimeSession::OnnxRuntimeSession(RuntimeConfig, bool) {
  throw std::runtime_error(
      "ONNX Runtime support requires MW_INFER_ENABLE_ONNXRUNTIME=ON");
}

OnnxRuntimeSession::~OnnxRuntimeSession() = default;

const RuntimeConfig& OnnxRuntimeSession::config() const {
  throw std::runtime_error(
      "ONNX Runtime support requires MW_INFER_ENABLE_ONNXRUNTIME=ON");
}

const InferOutputs& OnnxRuntimeSession::Infer(const std::vector<cv::Mat>&) {
  throw std::runtime_error(
      "ONNX Runtime support requires MW_INFER_ENABLE_ONNXRUNTIME=ON");
}

const InferOutputs& OnnxRuntimeSession::Infer(
    const std::vector<cv::cuda::GpuMat>&) {
  throw std::runtime_error(
      "ONNX Runtime support requires MW_INFER_ENABLE_ONNXRUNTIME=ON");
}

#endif  // MW_INFER_WITH_ONNXRUNTIME

}  // namespace mw::infer
