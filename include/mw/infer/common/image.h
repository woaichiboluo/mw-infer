#ifndef MW_INFER_RUNTIME_IMAGE_H_
#define MW_INFER_RUNTIME_IMAGE_H_

#include <cstdint>

#include "mw/infer/common/device.h"
#include "mw/infer/common/memory.h"
#include "mw/infer/common/types.h"

namespace mw::infer {

enum class PixelFormat {
  kRgb,
  kBgr,
  kGray,
  kNv12,
  kNv21,
  kRgba,
  kBgra,
};

struct ImageView {
  int32_t width = 0;
  int32_t height = 0;
  int32_t channels = 0;
  PixelFormat format = PixelFormat::kRgb;
  DataType data_type = DataType::kUint8;
  Device device;
  MemoryView data;
  int64_t stride_bytes = 0;
};

}  // namespace mw::infer

#endif  // MW_INFER_RUNTIME_IMAGE_H_
