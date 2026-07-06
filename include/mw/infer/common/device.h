#ifndef MW_INFER_COMMON_DEVICE_H_
#define MW_INFER_COMMON_DEVICE_H_

#include <cstdint>

namespace mw::infer {

enum class DeviceType : int32_t {
  kCpu,
  kCuda,
  kMetal,
  kCoreMl,
  kNnapi,
};

class Device {
 public:
  constexpr Device() noexcept = default;
  constexpr Device(DeviceType type, int32_t device_id = 0) noexcept
      : type_(type), device_id_(device_id) {}

  static constexpr Device Cpu(int32_t device_id = 0) noexcept {
    return Device(DeviceType::kCpu, device_id);
  }

  static constexpr Device Cuda(int32_t device_id = 0) noexcept {
    return Device(DeviceType::kCuda, device_id);
  }

  constexpr DeviceType type() const noexcept { return type_; }
  constexpr int32_t device_id() const noexcept { return device_id_; }
  constexpr bool is_host() const noexcept { return type_ == DeviceType::kCpu; }
  constexpr bool is_device() const noexcept { return !is_host(); }

  constexpr bool operator==(const Device& other) const noexcept {
    return type_ == other.type_ && device_id_ == other.device_id_;
  }

  constexpr bool operator!=(const Device& other) const noexcept {
    return !(*this == other);
  }

 private:
  DeviceType type_ = DeviceType::kCpu;
  int32_t device_id_ = 0;
};

bool IsCpuDevice(const Device& device);

}  // namespace mw::infer

#endif  // MW_INFER_COMMON_DEVICE_H_
