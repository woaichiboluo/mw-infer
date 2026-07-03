#ifndef MW_INFER_TENSOR_H_
#define MW_INFER_TENSOR_H_

#include <cstdint>
#include <string>
#include <vector>

#include "mw/infer/common/device.h"
#include "mw/infer/common/memory.h"
#include "mw/infer/common/types.h"

namespace mw::infer {

enum class TensorLayout {
  kUnknown,
  kNchw,
  kNhwc,
};

struct TensorSpec {
  std::string name;
  DataType data_type = DataType::kFloat32;
  TensorLayout layout = TensorLayout::kUnknown;
  std::vector<int64_t> shape;
};

class Tensor {
 public:
  Tensor() = default;
  Tensor(std::vector<int64_t> shape, std::vector<float> data);
  Tensor(TensorSpec spec, Device device, MemoryBuffer buffer);

  const TensorSpec& spec() const;
  const std::vector<int64_t>& shape() const;
  DataType data_type() const;
  const Device& device() const;
  const MemoryBuffer& buffer() const;
  MemoryBuffer& mutable_buffer();
  const std::vector<float>& data() const;
  std::vector<float>& mutable_data();

 private:
  TensorSpec spec_;
  Device device_;
  MemoryBuffer buffer_;
  std::vector<float> data_;
};

}  // namespace mw::infer

#endif  // MW_INFER_TENSOR_H_
