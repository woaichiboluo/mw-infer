#ifndef MW_INFER_RUNTIME_DEVICE_H_
#define MW_INFER_RUNTIME_DEVICE_H_

#include <cstdint>

namespace mw::infer {

enum class DeviceType {
  kCpu,
  kCuda,
  kMetal,
  kCoreMl,
  kNnapi,
};

struct Device {
  DeviceType type = DeviceType::kCpu;
  int32_t device_id = 0;
};

bool IsCpuDevice(const Device& device);

}  // namespace mw::infer

#endif  // MW_INFER_RUNTIME_DEVICE_H_
