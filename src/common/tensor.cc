#include "mw/infer/common/tensor.h"

#include <cstddef>
#include <cstring>
#include <utility>

namespace mw::infer {

Tensor::Tensor(std::vector<int64_t> shape, std::vector<float> data)
    : data_(std::move(data)) {
  spec_.shape = std::move(shape);
  spec_.data_type = DataType::kFloat32;

  std::vector<std::byte> bytes(data_.size() * sizeof(float));
  if (!bytes.empty()) {
    std::memcpy(bytes.data(), data_.data(), bytes.size());
  }
  buffer_ = MemoryBuffer(std::move(bytes));
}

Tensor::Tensor(TensorSpec spec, Device device, MemoryBuffer buffer)
    : spec_(std::move(spec)), device_(device), buffer_(std::move(buffer)) {}

const TensorSpec& Tensor::spec() const { return spec_; }

const std::vector<int64_t>& Tensor::shape() const { return spec_.shape; }

DataType Tensor::data_type() const { return spec_.data_type; }

const Device& Tensor::device() const { return device_; }

const MemoryBuffer& Tensor::buffer() const { return buffer_; }

MemoryBuffer& Tensor::mutable_buffer() { return buffer_; }

const std::vector<float>& Tensor::data() const { return data_; }

std::vector<float>& Tensor::mutable_data() { return data_; }

}  // namespace mw::infer
