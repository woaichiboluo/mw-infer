#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "mw/infer/runtime/process/image_to_tensor.h"

namespace mw::infer::process_internal {
namespace {

constexpr int kThreadsPerBlock = 256;
constexpr std::uint64_t kMaxBlocks = 65535;

void CheckCuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) +
                             " failed: " + cudaGetErrorString(status));
  }
}

int GridBlocks(std::uint64_t element_count) {
  const std::uint64_t blocks =
      (element_count + kThreadsPerBlock - 1) / kThreadsPerBlock;
  return static_cast<int>(blocks > kMaxBlocks ? kMaxBlocks : blocks);
}

__device__ int SourceChannel(int target_channel, int channels,
                             PixelFormat pixel_format) {
  if ((pixel_format == PixelFormat::kBgr ||
       pixel_format == PixelFormat::kBgra) &&
      channels >= 3 &&
      (target_channel == 0 || target_channel == 2)) {
    return 2 - target_channel;
  }
  return target_channel;
}

template <typename Target, typename Source>
__device__ Target ConvertValue(Source value) {
  if constexpr (std::is_same_v<Target, std::uint8_t>) {
    const double numeric = static_cast<double>(value);
    if (!(numeric > 0.0)) {
      return 0;
    }
    if (numeric >= 255.0) {
      return 255;
    }
    return static_cast<std::uint8_t>(__double2int_rn(numeric));
  }
  return static_cast<Target>(value);
}

template <typename Source, typename Target>
__global__ void ImageToTensorKernel(
    const std::uint8_t* source, std::size_t source_step, int rows, int cols,
    int channels, PixelFormat pixel_format, Target* output, int layout_id,
    std::uint64_t element_count) {
  const std::uint64_t start =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t stride =
      static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
  const std::uint64_t plane_elements =
      static_cast<std::uint64_t>(rows) * cols;
  for (std::uint64_t index = start; index < element_count; index += stride) {
    const int target_channel =
        layout_id == 0
            ? static_cast<int>((index / plane_elements) % channels)
            : static_cast<int>(index % channels);
    const std::uint64_t pixel_index =
        layout_id == 0 ? index % plane_elements : index / channels;
    const int y = static_cast<int>(pixel_index / cols);
    const int x = static_cast<int>(pixel_index % cols);
    const int source_channel =
        SourceChannel(target_channel, channels, pixel_format);
    const auto* source_row = reinterpret_cast<const Source*>(
        source + static_cast<std::size_t>(y) * source_step);
    const Source value = source_row[static_cast<std::size_t>(x) * channels +
                                    source_channel];
    output[index] = ConvertValue<Target>(value);
  }
}

template <typename Source>
void LaunchForTarget(const void* source, std::size_t source_step, int rows,
                     int cols, int channels, PixelFormat pixel_format,
                     Tensor* output, std::size_t batch_index,
                     TensorLayout layout, cudaStream_t stream,
                     std::uint64_t element_count) {
  const std::size_t batch_elements = static_cast<std::size_t>(element_count);
  const int layout_id = layout == TensorLayout::kBchw ? 0 : 1;
  switch (output->data_type()) {
    case DataType::kUInt8:
      ImageToTensorKernel<Source, std::uint8_t>
          <<<GridBlocks(element_count), kThreadsPerBlock, 0, stream>>>(
              static_cast<const std::uint8_t*>(source), source_step, rows, cols,
              channels, pixel_format,
              static_cast<std::uint8_t*>(output->data()) +
                  batch_index * batch_elements,
              layout_id, element_count);
      return;
    case DataType::kFloat32:
      ImageToTensorKernel<Source, float>
          <<<GridBlocks(element_count), kThreadsPerBlock, 0, stream>>>(
              static_cast<const std::uint8_t*>(source), source_step, rows, cols,
              channels, pixel_format,
              static_cast<float*>(output->data()) +
                  batch_index * batch_elements,
              layout_id, element_count);
      return;
    case DataType::kUnknown:
    case DataType::kInt8:
    case DataType::kUInt16:
    case DataType::kInt16:
    case DataType::kInt32:
    case DataType::kInt64:
    case DataType::kFloat16:
    case DataType::kFloat64:
      throw std::invalid_argument(
          "CUDA image-to-tensor target data type is unsupported");
  }
  throw std::invalid_argument(
      "CUDA image-to-tensor target data type is unsupported");
}

}  // namespace

void RunImageToTensorOnDevice(const void* source, std::size_t source_step,
                              DataType source_type, int rows, int cols,
                              int channels, PixelFormat pixel_format,
                              Tensor* output, std::size_t batch_index,
                              TensorLayout layout, cudaStream_t stream) {
  if (source == nullptr || output == nullptr || output->empty()) {
    throw std::invalid_argument("CUDA image-to-tensor input is empty");
  }
  const std::uint64_t element_count = static_cast<std::uint64_t>(rows) * cols *
                                      static_cast<std::uint64_t>(channels);
  if (element_count == 0 ||
      element_count > std::numeric_limits<std::size_t>::max()) {
    throw std::invalid_argument("CUDA image-to-tensor shape is invalid");
  }

  switch (source_type) {
    case DataType::kUInt8:
      LaunchForTarget<std::uint8_t>(source, source_step, rows, cols, channels,
                                    pixel_format, output, batch_index, layout,
                                    stream, element_count);
      break;
    case DataType::kInt8:
      LaunchForTarget<std::int8_t>(source, source_step, rows, cols, channels,
                                   pixel_format, output, batch_index, layout,
                                   stream, element_count);
      break;
    case DataType::kUInt16:
      LaunchForTarget<std::uint16_t>(source, source_step, rows, cols, channels,
                                     pixel_format, output, batch_index, layout,
                                     stream, element_count);
      break;
    case DataType::kInt16:
      LaunchForTarget<std::int16_t>(source, source_step, rows, cols, channels,
                                    pixel_format, output, batch_index, layout,
                                    stream, element_count);
      break;
    case DataType::kInt32:
      LaunchForTarget<std::int32_t>(source, source_step, rows, cols, channels,
                                    pixel_format, output, batch_index, layout,
                                    stream, element_count);
      break;
    case DataType::kFloat32:
      LaunchForTarget<float>(source, source_step, rows, cols, channels,
                             pixel_format, output, batch_index, layout, stream,
                             element_count);
      break;
    case DataType::kFloat64:
      LaunchForTarget<double>(source, source_step, rows, cols, channels,
                              pixel_format, output, batch_index, layout, stream,
                              element_count);
      break;
    case DataType::kUnknown:
    case DataType::kInt64:
    case DataType::kFloat16:
      throw std::invalid_argument(
          "CUDA image-to-tensor source data type is unsupported");
  }
  CheckCuda(cudaGetLastError(), "ImageToTensorKernel");
}

}  // namespace mw::infer::process_internal
