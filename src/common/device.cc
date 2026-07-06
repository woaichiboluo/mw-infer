#include "mw/infer/common/device.h"

namespace mw::infer {

bool IsCpuDevice(const Device& device) {
  return device.type() == DeviceType::kCpu;
}

}  // namespace mw::infer
