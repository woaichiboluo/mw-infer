#include "mw/infer/runtime/process/channel.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mw::infer {

#if defined(MW_INFER_HAS_CUDA_PREPROCESS)
namespace process_internal {
Tensor RunReorderChannelsOnDevice(const Tensor& input,
                                  const std::vector<int64_t>& order,
                                  TensorLayout layout,
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
    throw std::invalid_argument("ReorderChannels tensor is empty");
  }
  if (input.data_type() != DataType::kFloat32) {
    throw std::invalid_argument("ReorderChannels tensor must be float32");
  }
  if (input.shape().size() != 4) {
    throw std::invalid_argument("ReorderChannels tensor shape must be rank 4");
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
    throw std::invalid_argument(
        "ReorderChannels tensor dimensions must be positive");
  }
  return shape;
}

void ValidateOrder(const std::vector<int64_t>& order, int64_t channels) {
  if (order.size() != static_cast<std::size_t>(channels)) {
    throw std::invalid_argument(
        "ReorderChannels order size must match channel count");
  }

  std::vector<bool> seen(static_cast<std::size_t>(channels), false);
  for (int64_t channel : order) {
    if (channel < 0 || channel >= channels) {
      throw std::invalid_argument(
          "ReorderChannels order contains an invalid channel");
    }
    if (seen[static_cast<std::size_t>(channel)]) {
      throw std::invalid_argument(
          "ReorderChannels order must be a permutation");
    }
    seen[static_cast<std::size_t>(channel)] = true;
  }
}

TensorDesc MakeOutputDesc(const Tensor& input) {
  TensorDesc desc = input.desc();
  return desc;
}

Tensor RunReorderChannelsOnHost(const Tensor& input,
                                const std::vector<int64_t>& order,
                                ImageTensorShape shape, TensorLayout layout,
                                TensorAllocator& allocator) {
  Tensor output = Tensor::Allocate(MakeOutputDesc(input), allocator);
  const auto* input_data = static_cast<const float*>(input.data());
  auto* output_data = static_cast<float*>(output.data());

  const int64_t spatial = shape.height * shape.width;
  for (int64_t batch = 0; batch < shape.batch; ++batch) {
    for (int64_t channel = 0; channel < shape.channels; ++channel) {
      const int64_t source_channel = order[static_cast<std::size_t>(channel)];
      for (int64_t index = 0; index < spatial; ++index) {
        std::size_t output_index = 0;
        std::size_t input_index = 0;
        switch (layout) {
          case TensorLayout::kBchw:
            output_index = static_cast<std::size_t>(
                ((batch * shape.channels + channel) * spatial) + index);
            input_index = static_cast<std::size_t>(
                ((batch * shape.channels + source_channel) * spatial) + index);
            break;
          case TensorLayout::kBhwc:
            output_index = static_cast<std::size_t>(
                ((batch * spatial + index) * shape.channels) + channel);
            input_index = static_cast<std::size_t>(
                ((batch * spatial + index) * shape.channels) + source_channel);
            break;
        }
        output_data[output_index] = input_data[input_index];
      }
    }
  }
  return output;
}

}  // namespace

Tensor ReorderChannels(const Tensor& input, const std::vector<int64_t>& order,
                       TensorLayout layout, TensorAllocator& allocator) {
  const ImageTensorShape shape = ValidateImageTensor(input, layout);
  ValidateOrder(order, shape.channels);
  if (input.device().type == DeviceType::kCpu) {
    return RunReorderChannelsOnHost(input, order, shape, layout, allocator);
  }
  if (input.device().type == DeviceType::kCuda) {
#if defined(MW_INFER_HAS_CUDA_PREPROCESS)
    return process_internal::RunReorderChannelsOnDevice(input, order, layout,
                                                        allocator);
#else
    throw std::runtime_error("CUDA preprocess is unavailable in this build");
#endif
  }
  throw std::invalid_argument("ReorderChannels tensor device is unsupported");
}

}  // namespace mw::infer
