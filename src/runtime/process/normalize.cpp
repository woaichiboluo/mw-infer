#include "mw/infer/runtime/process/normalize.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace mw::infer {

#if defined(MW_INFER_HAS_CUDA_PREPROCESS)
namespace process_internal {
Tensor RunNormalizeOnDevice(const Tensor& input, const std::vector<float>& mean,
                            const std::vector<float>& stddev, float scale,
                            TensorLayout layout, ExecutionStream* stream,
                            TensorAllocator& allocator);
}  // namespace process_internal
#endif

namespace {

struct ImageTensorShape {
  int64_t batch = 0;
  int64_t channels = 0;
  int64_t height = 0;
  int64_t width = 0;
};

ImageTensorShape ValidateImageTensor(const Tensor& input, TensorLayout layout) {
  if (input.empty()) {
    throw std::invalid_argument("Normalize tensor is empty");
  }
  if (input.data_type() != DataType::kFloat32) {
    throw std::invalid_argument("Normalize tensor must be float32");
  }
  if (input.shape().size() != 4) {
    throw std::invalid_argument("Normalize tensor shape must be rank 4");
  }

  ImageTensorShape shape;
  shape.batch = input.shape()[0];
  switch (layout) {
    case TensorLayout::kBchw:
      shape.channels = input.shape()[1];
      shape.height = input.shape()[2];
      shape.width = input.shape()[3];
      break;
    case TensorLayout::kBhwc:
      shape.height = input.shape()[1];
      shape.width = input.shape()[2];
      shape.channels = input.shape()[3];
      break;
  }

  if (shape.batch <= 0 || shape.channels <= 0 || shape.height <= 0 ||
      shape.width <= 0) {
    throw std::invalid_argument("Normalize tensor dimensions must be positive");
  }
  return shape;
}

void ValidateParameters(const std::vector<float>& mean,
                        const std::vector<float>& stddev, float scale,
                        int64_t channels) {
  if (!std::isfinite(scale)) {
    throw std::invalid_argument("Normalize scale must be finite");
  }
  if (mean.size() != static_cast<std::size_t>(channels) ||
      stddev.size() != static_cast<std::size_t>(channels)) {
    throw std::invalid_argument(
        "Normalize mean and stddev size must match channel count");
  }

  for (std::size_t index = 0; index < mean.size(); ++index) {
    if (!std::isfinite(mean[index]) || !std::isfinite(stddev[index])) {
      throw std::invalid_argument(
          "Normalize mean and stddev values must be finite");
    }
    if (stddev[index] == 0.0F) {
      throw std::invalid_argument("Normalize stddev must be non-zero");
    }
  }
}

TensorDesc MakeOutputDesc(const Tensor& input) {
  TensorDesc desc = input.desc();
  return desc;
}

Tensor RunNormalizeOnHost(const Tensor& input, const std::vector<float>& mean,
                          const std::vector<float>& stddev, float scale,
                          ImageTensorShape shape, TensorLayout layout,
                          TensorAllocator& allocator) {
  Tensor output = Tensor::Allocate(MakeOutputDesc(input), allocator);
  const auto* input_data = static_cast<const float*>(input.data());
  auto* output_data = static_cast<float*>(output.data());

  const int64_t spatial = shape.height * shape.width;
  for (int64_t batch = 0; batch < shape.batch; ++batch) {
    for (int64_t channel = 0; channel < shape.channels; ++channel) {
      const float channel_mean = mean[static_cast<std::size_t>(channel)];
      const float channel_stddev = stddev[static_cast<std::size_t>(channel)];
      for (int64_t index = 0; index < spatial; ++index) {
        std::size_t offset = 0;
        switch (layout) {
          case TensorLayout::kBchw:
            offset = static_cast<std::size_t>(
                ((batch * shape.channels + channel) * spatial) + index);
            break;
          case TensorLayout::kBhwc:
            offset = static_cast<std::size_t>(
                ((batch * spatial + index) * shape.channels) + channel);
            break;
        }
        output_data[offset] =
            (input_data[offset] * scale - channel_mean) / channel_stddev;
      }
    }
  }
  return output;
}

bool SameDevice(Device lhs, Device rhs) {
  return lhs.type == rhs.type && lhs.id == rhs.id;
}

Tensor NormalizeImpl(const Tensor& input, const std::vector<float>& mean,
                     const std::vector<float>& stddev, float scale,
                     TensorLayout layout, ExecutionStream* stream,
                     TensorAllocator& allocator) {
  const ImageTensorShape shape = ValidateImageTensor(input, layout);
  ValidateParameters(mean, stddev, scale, shape.channels);
  if (stream != nullptr && !SameDevice(input.device(), stream->device())) {
    throw std::invalid_argument(
        "Normalize stream device does not match input device");
  }
  if (input.device().type == DeviceType::kCpu) {
    return RunNormalizeOnHost(input, mean, stddev, scale, shape, layout,
                              allocator);
  }
  if (input.device().type == DeviceType::kCuda) {
#if defined(MW_INFER_HAS_CUDA_PREPROCESS)
    return process_internal::RunNormalizeOnDevice(
        input, mean, stddev, scale, layout, stream, allocator);
#else
    throw std::runtime_error("CUDA preprocess is unavailable in this build");
#endif
  }
  throw std::invalid_argument("Normalize tensor device is unsupported");
}

}  // namespace

Tensor Normalize(const Tensor& input, const std::vector<float>& mean,
                 const std::vector<float>& stddev, float scale,
                 TensorLayout layout, TensorAllocator& allocator) {
  return NormalizeImpl(input, mean, stddev, scale, layout, nullptr, allocator);
}

Tensor Normalize(const Tensor& input, const std::vector<float>& mean,
                 const std::vector<float>& stddev, ExecutionStream& stream,
                 float scale, TensorLayout layout, TensorAllocator& allocator) {
  return NormalizeImpl(input, mean, stddev, scale, layout, &stream, allocator);
}

}  // namespace mw::infer
