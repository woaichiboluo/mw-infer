#ifndef MW_INFER_RUNTIME_TENSOR_TENSOR_H_
#define MW_INFER_RUNTIME_TENSOR_TENSOR_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mw::infer {

class TensorAllocator;

enum class DataType {
  kUnknown,
  kUInt8,
  kInt8,
  kUInt16,
  kInt16,
  kInt32,
  kInt64,
  kFloat16,
  kFloat32,
  kFloat64,
};

enum class DeviceType {
  kCpu,
  kCuda,
};

struct Device {
  DeviceType type = DeviceType::kCpu;
  int id = 0;
};

struct TensorDesc {
  std::string name;
  DataType data_type = DataType::kUnknown;
  std::vector<int64_t> shape;
  Device device;
};

inline std::size_t DataTypeSize(DataType data_type) {
  switch (data_type) {
    case DataType::kUInt8:
    case DataType::kInt8:
      return 1;
    case DataType::kUInt16:
    case DataType::kInt16:
    case DataType::kFloat16:
      return 2;
    case DataType::kInt32:
    case DataType::kFloat32:
      return 4;
    case DataType::kInt64:
    case DataType::kFloat64:
      return 8;
    case DataType::kUnknown:
      throw std::invalid_argument("Data type is unknown");
  }
  throw std::invalid_argument("Data type is unknown");
}

inline std::size_t ElementCount(const std::vector<int64_t>& shape) {
  if (shape.empty()) {
    throw std::invalid_argument("Tensor shape is empty");
  }

  std::size_t count = 1;
  for (int64_t dim : shape) {
    if (dim <= 0) {
      throw std::invalid_argument("Tensor shape dimensions must be positive");
    }
    count *= static_cast<std::size_t>(dim);
  }
  return count;
}

inline std::size_t TensorBytes(const TensorDesc& desc) {
  return ElementCount(desc.shape) * DataTypeSize(desc.data_type);
}

class Tensor {
 public:
  Tensor() = default;

  static Tensor Allocate(TensorDesc desc);
  static Tensor Allocate(TensorDesc desc, const TensorAllocator& allocator);

  static Tensor FromBuffer(TensorDesc desc, void* data, std::size_t bytes,
                           std::function<void(void*)> deleter) {
    ValidateBuffer(desc, data, bytes);
    if (!deleter) {
      throw std::invalid_argument("Tensor buffer deleter is empty");
    }

    return Tensor(std::move(desc),
                  std::shared_ptr<void>(data, std::move(deleter)), bytes);
  }

  static Tensor FromExternal(TensorDesc desc, void* data, std::size_t bytes) {
    ValidateBuffer(desc, data, bytes);
    return Tensor(std::move(desc), std::shared_ptr<void>(data, [](void*) {}),
                  bytes);
  }

  static Tensor FromExternal(TensorDesc desc, void* data, std::size_t bytes,
                             std::shared_ptr<void> owner) {
    ValidateBuffer(desc, data, bytes);
    if (!owner) {
      throw std::invalid_argument("Tensor external owner is empty");
    }

    return Tensor(std::move(desc),
                  std::shared_ptr<void>(std::move(owner), data), bytes);
  }

  bool empty() const { return !data_; }
  const TensorDesc& desc() const { return desc_; }
  const std::string& name() const { return desc_.name; }
  DataType data_type() const { return desc_.data_type; }
  const std::vector<int64_t>& shape() const { return desc_.shape; }
  Device device() const { return desc_.device; }
  void* data() { return data_.get(); }
  const void* data() const { return data_.get(); }
  std::size_t bytes() const { return bytes_; }
  std::size_t capacity_bytes() const { return capacity_bytes_; }
  std::size_t element_count() const { return ElementCount(desc_.shape); }

  Tensor View(TensorDesc desc) const {
    if (empty()) {
      throw std::invalid_argument("Cannot create a view from an empty tensor");
    }
    if (desc.device.type != desc_.device.type ||
        desc.device.id != desc_.device.id) {
      throw std::invalid_argument("Tensor view device mismatch");
    }

    const std::size_t view_bytes = TensorBytes(desc);
    if (view_bytes > capacity_bytes_) {
      throw std::invalid_argument("Tensor view exceeds buffer capacity");
    }

    return Tensor(std::move(desc), data_, view_bytes, capacity_bytes_);
  }

 private:
  Tensor(TensorDesc desc, std::shared_ptr<void> data, std::size_t bytes)
      : Tensor(std::move(desc), std::move(data), bytes, bytes) {}

  Tensor(TensorDesc desc, std::shared_ptr<void> data, std::size_t bytes,
         std::size_t capacity_bytes)
      : desc_(std::move(desc)),
        data_(std::move(data)),
        bytes_(bytes),
        capacity_bytes_(capacity_bytes) {}

  static void ValidateBuffer(const TensorDesc& desc, const void* data,
                             std::size_t bytes) {
    if (data == nullptr) {
      throw std::invalid_argument("Tensor data is null");
    }
    const std::size_t required_bytes = TensorBytes(desc);
    if (bytes < required_bytes) {
      throw std::invalid_argument("Tensor buffer is smaller than tensor shape");
    }
  }

  TensorDesc desc_;
  std::shared_ptr<void> data_;
  std::size_t bytes_ = 0;
  std::size_t capacity_bytes_ = 0;
};

}  // namespace mw::infer

#endif  // MW_INFER_RUNTIME_TENSOR_TENSOR_H_
