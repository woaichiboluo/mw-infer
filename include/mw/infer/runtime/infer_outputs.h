#ifndef MW_INFER_RUNTIME_INFER_OUTPUTS_H_
#define MW_INFER_RUNTIME_INFER_OUTPUTS_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "mw/infer/common/device.h"

namespace mw::infer {

using Shape = std::vector<int64_t>;

enum class InferDataType : int32_t {
  kFloat32,
  kFloat16,
  kInt8,
  kUint8,
  kInt32,
  kInt64,
};

struct BufferView {
  const void* host = nullptr;
  const void* device = nullptr;
  std::size_t size_bytes = 0;
};

struct InferOutput {
  std::string name;
  InferDataType data_type = InferDataType::kFloat32;
  Shape shape;
  Device device = Device::Cpu();
  // Non-owning storage produced by an infer block. It is valid until the
  // producing block runs again, unless that block documents a shorter lifetime.
  BufferView buffer;
};

struct InferOutputs {
  int batch_size = 0;
  std::vector<InferOutput> outputs;
};

}  // namespace mw::infer

#endif  // MW_INFER_RUNTIME_INFER_OUTPUTS_H_
