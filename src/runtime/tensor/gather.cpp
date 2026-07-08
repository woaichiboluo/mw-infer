#include "mw/infer/runtime/tensor/gather.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mw::infer {

#if defined(MW_INFER_HAS_CUDA_TENSOR_OPS)
namespace tensor_internal {
Tensor RunGatherRowsOnDevice(const Tensor& data, const Tensor& indices);
}  // namespace tensor_internal
#endif

namespace {

struct GatherRowsLayout {
  int64_t row_count = 0;
  int64_t selected_count = 0;
  std::size_t row_bytes = 0;
  std::vector<int64_t> output_shape;
};

std::string MakeOutputName(const Tensor& data) {
  if (data.name().empty()) {
    return "gather_rows";
  }
  return data.name() + "_gathered";
}

std::size_t RowBytes(const Tensor& data) {
  std::size_t row_elements = 1;
  for (std::size_t index = 1; index < data.shape().size(); ++index) {
    row_elements *= static_cast<std::size_t>(data.shape()[index]);
  }
  return row_elements * DataTypeSize(data.data_type());
}

GatherRowsLayout ValidateInputs(const Tensor& data, const Tensor& indices) {
  if (data.empty()) {
    throw std::invalid_argument("GatherRows data tensor is empty");
  }
  if (indices.empty()) {
    throw std::invalid_argument("GatherRows indices tensor is empty");
  }
  if (data.data_type() == DataType::kUnknown) {
    throw std::invalid_argument("GatherRows data tensor type is unknown");
  }
  if (data.shape().empty()) {
    throw std::invalid_argument("GatherRows data tensor shape is empty");
  }
  if (indices.data_type() != DataType::kInt64) {
    throw std::invalid_argument("GatherRows indices tensor must be int64");
  }
  if (indices.shape().size() != 1) {
    throw std::invalid_argument("GatherRows indices tensor shape must be [K]");
  }
  if (data.device().type != indices.device().type ||
      data.device().id != indices.device().id) {
    throw std::invalid_argument("GatherRows data and indices device mismatch");
  }

  GatherRowsLayout layout;
  layout.row_count = data.shape()[0];
  layout.selected_count = indices.shape()[0];
  if (layout.row_count <= 0 || layout.selected_count <= 0) {
    throw std::invalid_argument(
        "GatherRows tensor dimensions must be positive");
  }
  layout.row_bytes = RowBytes(data);
  layout.output_shape = data.shape();
  layout.output_shape[0] = layout.selected_count;
  return layout;
}

TensorDesc MakeOutputDesc(const Tensor& data,
                          const std::vector<int64_t>& output_shape) {
  TensorDesc desc;
  desc.info.name = MakeOutputName(data);
  desc.info.data_type = data.data_type();
  desc.info.shape = output_shape;
  desc.device = data.device();
  return desc;
}

void ValidateHostIndices(const int64_t* indices, int64_t selected_count,
                         int64_t row_count) {
  for (int64_t index = 0; index < selected_count; ++index) {
    const int64_t row = indices[index];
    if (row < 0 || row >= row_count) {
      throw std::invalid_argument("GatherRows index is out of range");
    }
  }
}

Tensor RunGatherRowsOnHost(const Tensor& data, const Tensor& indices,
                           const GatherRowsLayout& layout) {
  const auto* index_data = static_cast<const int64_t*>(indices.data());
  ValidateHostIndices(index_data, layout.selected_count, layout.row_count);

  Tensor output = Tensor::Allocate(MakeOutputDesc(data, layout.output_shape));
  const auto* input_bytes = static_cast<const std::uint8_t*>(data.data());
  auto* output_bytes = static_cast<std::uint8_t*>(output.data());
  for (int64_t output_row = 0; output_row < layout.selected_count;
       ++output_row) {
    const int64_t input_row = index_data[output_row];
    std::memcpy(
        output_bytes + static_cast<std::size_t>(output_row) * layout.row_bytes,
        input_bytes + static_cast<std::size_t>(input_row) * layout.row_bytes,
        layout.row_bytes);
  }
  return output;
}

}  // namespace

Tensor GatherRows(const Tensor& data, const Tensor& indices) {
  const GatherRowsLayout layout = ValidateInputs(data, indices);
  if (data.device().type == DeviceType::kCpu) {
    return RunGatherRowsOnHost(data, indices, layout);
  }
  if (data.device().type == DeviceType::kCuda) {
#if defined(MW_INFER_HAS_CUDA_TENSOR_OPS)
    return tensor_internal::RunGatherRowsOnDevice(data, indices);
#else
    throw std::runtime_error("CUDA tensor ops are unavailable in this build");
#endif
  }
  throw std::invalid_argument("GatherRows tensor device is unsupported");
}

Tensor Tensor::GatherRows(const Tensor& indices) const {
  return mw::infer::GatherRows(*this, indices);
}

}  // namespace mw::infer
