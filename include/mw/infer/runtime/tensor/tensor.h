#ifndef MW_INFER_RUNTIME_TENSOR_TENSOR_H_
#define MW_INFER_RUNTIME_TENSOR_TENSOR_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace mw::infer {

class ExecutionStream;

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

  Device() = default;
  Device(DeviceType device_type, int device_id = 0)
      : type(device_type), id(device_id) {}
  explicit Device(std::string_view text) : Device(ParseText(text)) {}

  std::string ToString() const {
    switch (type) {
      case DeviceType::kCpu:
        return "cpu";
      case DeviceType::kCuda:
        return "cuda:" + std::to_string(id);
    }
    return "unknown";
  }

 private:
  static Device ParseText(std::string_view text) {
    if (text == "cpu") {
      return Device(DeviceType::kCpu, 0);
    }
    if (text == "cuda") {
      return Device(DeviceType::kCuda, 0);
    }

    constexpr std::string_view kCudaPrefix = "cuda:";
    if (text.rfind(kCudaPrefix, 0) == 0) {
      return Device(DeviceType::kCuda,
                    ParseDeviceId(text.substr(kCudaPrefix.size())));
    }

    throw std::invalid_argument("Unsupported device: " + std::string(text));
  }

  static int ParseDeviceId(std::string_view text) {
    if (text.empty()) {
      throw std::invalid_argument("Device id is empty");
    }

    int value = 0;
    for (char character : text) {
      if (character < '0' || character > '9') {
        throw std::invalid_argument("Device id is invalid");
      }
      const int digit = character - '0';
      if (value > (std::numeric_limits<int>::max() - digit) / 10) {
        throw std::invalid_argument("Device id exceeds int range");
      }
      value = value * 10 + digit;
    }
    return value;
  }
};

struct TensorInfo {
  std::string name;
  DataType data_type = DataType::kUnknown;
  std::vector<int64_t> shape;
};

struct TensorDesc {
  TensorInfo info;
  Device device;
};

class Tensor;

class TensorAllocator {
 public:
  virtual ~TensorAllocator() = default;

  static TensorAllocator& Default();

  virtual bool Supports(Device device) const = 0;
  virtual Tensor Allocate(TensorDesc desc) = 0;
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
  std::size_t count = 1;
  for (int64_t dim : shape) {
    if (dim < 0) {
      throw std::invalid_argument(
          "Tensor shape dimensions must be non-negative");
    }
    const auto value = static_cast<std::uint64_t>(dim);
    if (value > std::numeric_limits<std::size_t>::max()) {
      throw std::invalid_argument("Tensor shape dimension exceeds size_t");
    }
    const std::size_t size = static_cast<std::size_t>(value);
    if (size != 0 && count > std::numeric_limits<std::size_t>::max() / size) {
      throw std::invalid_argument("Tensor element count overflows size_t");
    }
    count *= size;
  }
  return count;
}

inline std::size_t TensorBytes(const TensorDesc& desc) {
  const std::size_t count = ElementCount(desc.info.shape);
  const std::size_t type_size = DataTypeSize(desc.info.data_type);
  if (type_size != 0 &&
      count > std::numeric_limits<std::size_t>::max() / type_size) {
    throw std::invalid_argument("Tensor byte size overflows size_t");
  }
  return count * type_size;
}

template <typename T>
struct TensorElementType;

template <>
struct TensorElementType<std::uint8_t> {
  static constexpr DataType value = DataType::kUInt8;
};

template <>
struct TensorElementType<std::int8_t> {
  static constexpr DataType value = DataType::kInt8;
};

template <>
struct TensorElementType<std::uint16_t> {
  static constexpr DataType value = DataType::kUInt16;
};

template <>
struct TensorElementType<std::int16_t> {
  static constexpr DataType value = DataType::kInt16;
};

template <>
struct TensorElementType<std::int32_t> {
  static constexpr DataType value = DataType::kInt32;
};

template <>
struct TensorElementType<std::int64_t> {
  static constexpr DataType value = DataType::kInt64;
};

template <>
struct TensorElementType<float> {
  static constexpr DataType value = DataType::kFloat32;
};

template <>
struct TensorElementType<double> {
  static constexpr DataType value = DataType::kFloat64;
};

template <typename T>
constexpr DataType TensorElementDataType() {
  return TensorElementType<std::remove_cv_t<T>>::value;
}

class Tensor {
 public:
  Tensor() = default;

  static Tensor Allocate(
      TensorDesc desc, TensorAllocator& allocator = TensorAllocator::Default());

  static Tensor FromBuffer(TensorDesc desc, void* data, std::size_t bytes,
                           std::function<void(void*)> deleter) {
    const std::size_t required_bytes = ValidateBuffer(desc, data, bytes);
    if (!deleter) {
      throw std::invalid_argument("Tensor buffer deleter is empty");
    }

    return Tensor(std::move(desc),
                  std::shared_ptr<void>(data, std::move(deleter)),
                  required_bytes, bytes);
  }

  static Tensor FromExternal(TensorDesc desc, void* data, std::size_t bytes) {
    const std::size_t required_bytes = ValidateBuffer(desc, data, bytes);
    return Tensor(std::move(desc), std::shared_ptr<void>(data, [](void*) {}),
                  required_bytes, bytes);
  }

  static Tensor FromExternal(TensorDesc desc, void* data, std::size_t bytes,
                             std::shared_ptr<void> owner) {
    const std::size_t required_bytes = ValidateBuffer(desc, data, bytes);
    if (!owner) {
      throw std::invalid_argument("Tensor external owner is empty");
    }

    return Tensor(std::move(desc),
                  std::shared_ptr<void>(std::move(owner), data), required_bytes,
                  bytes);
  }

  bool empty() const { return !initialized_; }
  const TensorDesc& desc() const { return desc_; }
  const TensorInfo& info() const { return desc_.info; }
  const std::string& name() const { return desc_.info.name; }
  DataType data_type() const { return desc_.info.data_type; }
  const std::vector<int64_t>& shape() const { return desc_.info.shape; }
  Device device() const { return desc_.device; }
  void* data() { return data_.get(); }
  const void* data() const { return data_.get(); }
  template <typename T>
  T* data() {
    ValidateTypedAccess<T>();
    return static_cast<T*>(data_.get());
  }
  template <typename T>
  const T* data() const {
    ValidateTypedAccess<T>();
    return static_cast<const T*>(data_.get());
  }
  std::size_t bytes() const { return bytes_; }
  std::size_t capacity_bytes() const { return capacity_bytes_; }
  std::size_t element_count() const {
    if (empty()) {
      throw std::invalid_argument("Tensor is empty");
    }
    return ElementCount(desc_.info.shape);
  }

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

  Tensor CopyTo(Device target_device,
                TensorAllocator& allocator = TensorAllocator::Default()) const;
  Tensor GatherRows(const Tensor& indices,
                    TensorAllocator& allocator = TensorAllocator::Default(),
                    ExecutionStream* execution_stream = nullptr) const;
  template <typename T>
  std::vector<T> CopyToHostVector(
      TensorAllocator& allocator = TensorAllocator::Default()) const {
    if (empty()) {
      throw std::invalid_argument("Tensor is empty");
    }

    Tensor host_storage;
    const Tensor* host = this;
    if (device().type != DeviceType::kCpu) {
      host_storage = CopyTo(Device{DeviceType::kCpu, 0}, allocator);
      host = &host_storage;
    }

    host->ValidateTypedAccess<T>();
    const std::size_t count = host->element_count();
    if (count == 0) {
      return {};
    }

    const T* values = host->data<T>();
    return std::vector<T>(values, values + count);
  }
  template <typename T>
  T At(const std::vector<int64_t>& indices) const {
    ValidateTypedAccess<T>();
    const std::size_t offset = ElementOffset(indices);
    T value{};
    CopyElementToHost(offset, &value, sizeof(T));
    return value;
  }

 private:
  Tensor(TensorDesc desc, std::shared_ptr<void> data, std::size_t bytes)
      : Tensor(std::move(desc), std::move(data), bytes, bytes) {}

  Tensor(TensorDesc desc, std::shared_ptr<void> data, std::size_t bytes,
         std::size_t capacity_bytes)
      : desc_(std::move(desc)),
        data_(std::move(data)),
        bytes_(bytes),
        capacity_bytes_(capacity_bytes),
        initialized_(true) {}

  static std::size_t ValidateBuffer(const TensorDesc& desc, const void* data,
                                    std::size_t bytes) {
    const std::size_t required_bytes = TensorBytes(desc);
    if (bytes < required_bytes) {
      throw std::invalid_argument("Tensor buffer is smaller than tensor shape");
    }
    if (bytes > 0 && data == nullptr) {
      throw std::invalid_argument("Tensor data is null");
    }
    return required_bytes;
  }

  template <typename T>
  void ValidateTypedAccess() const {
    if (empty()) {
      throw std::invalid_argument("Tensor is empty");
    }
    if (data_type() != TensorElementDataType<T>()) {
      throw std::invalid_argument(
          "Tensor data type does not match requested C++ type");
    }
  }

  std::size_t ElementOffset(const std::vector<int64_t>& indices) const {
    if (empty()) {
      throw std::invalid_argument("Tensor is empty");
    }
    if (indices.size() != desc_.info.shape.size()) {
      throw std::invalid_argument("Tensor index rank mismatch");
    }

    std::size_t offset = 0;
    for (std::size_t axis = 0; axis < indices.size(); ++axis) {
      const int64_t dim = desc_.info.shape[axis];
      const int64_t index = indices[axis];
      if (index < 0 || index >= dim) {
        throw std::invalid_argument("Tensor index is out of range");
      }
      const std::size_t size = static_cast<std::size_t>(dim);
      if (size != 0 &&
          offset > std::numeric_limits<std::size_t>::max() / size) {
        throw std::invalid_argument("Tensor index offset overflows size_t");
      }
      offset *= size;
      const std::size_t item = static_cast<std::size_t>(index);
      if (offset > std::numeric_limits<std::size_t>::max() - item) {
        throw std::invalid_argument("Tensor index offset overflows size_t");
      }
      offset += item;
    }
    if (offset >= element_count()) {
      throw std::invalid_argument("Tensor index is out of range");
    }
    return offset;
  }
  void CopyElementToHost(std::size_t element_offset, void* output,
                         std::size_t element_bytes) const;

  TensorDesc desc_;
  std::shared_ptr<void> data_;
  std::size_t bytes_ = 0;
  std::size_t capacity_bytes_ = 0;
  bool initialized_ = false;
};

}  // namespace mw::infer

#endif  // MW_INFER_RUNTIME_TENSOR_TENSOR_H_
