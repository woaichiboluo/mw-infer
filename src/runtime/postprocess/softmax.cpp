#include "mw/infer/runtime/postprocess/softmax.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace mw::infer {

#if defined(MW_INFER_HAS_CUDA_POSTPROCESS)
namespace postprocess_internal {
Tensor RunSoftmaxOnDevice(const Tensor& logits, TensorAllocator& allocator);
}  // namespace postprocess_internal
#endif

namespace {

struct MatrixShape {
  int64_t rows = 0;
  int64_t columns = 0;
};

MatrixShape ValidateLogits(const Tensor& logits) {
  if (logits.empty()) {
    throw std::invalid_argument("Softmax tensor is empty");
  }
  if (logits.data_type() != DataType::kFloat32) {
    throw std::invalid_argument("Softmax tensor must be float32");
  }
  if (logits.shape().size() != 1 && logits.shape().size() != 2) {
    throw std::invalid_argument("Softmax tensor shape must be [C] or [N, C]");
  }

  MatrixShape shape;
  shape.rows = logits.shape().size() == 2 ? logits.shape()[0] : 1;
  shape.columns =
      logits.shape().size() == 2 ? logits.shape()[1] : logits.shape()[0];
  if (shape.rows <= 0 || shape.columns <= 0) {
    throw std::invalid_argument("Softmax tensor dimensions must be positive");
  }
  return shape;
}

TensorDesc MakeOutputDesc(const Tensor& logits) {
  TensorDesc desc;
  desc.info.name = "softmax";
  desc.info.data_type = DataType::kFloat32;
  desc.info.shape = logits.shape();
  desc.device = logits.device();
  return desc;
}

Tensor RunSoftmaxOnHost(const Tensor& logits, MatrixShape shape,
                        TensorAllocator& allocator) {
  Tensor output = Tensor::Allocate(MakeOutputDesc(logits), allocator);
  const auto* input = static_cast<const float*>(logits.data());
  auto* result = static_cast<float*>(output.data());
  for (int64_t row = 0; row < shape.rows; ++row) {
    const auto row_offset = static_cast<std::size_t>(row * shape.columns);
    const float* row_input = input + row_offset;
    float* row_output = result + row_offset;

    const float row_max =
        *std::max_element(row_input, row_input + shape.columns);
    float sum = 0.0F;
    for (int64_t column = 0; column < shape.columns; ++column) {
      const float value = std::exp(row_input[column] - row_max);
      row_output[column] = value;
      sum += value;
    }
    for (int64_t column = 0; column < shape.columns; ++column) {
      row_output[column] /= sum;
    }
  }
  return output;
}

}  // namespace

Tensor Softmax(const Tensor& logits, TensorAllocator& allocator) {
  const MatrixShape shape = ValidateLogits(logits);
  if (logits.device().type == DeviceType::kCpu) {
    return RunSoftmaxOnHost(logits, shape, allocator);
  }
  if (logits.device().type == DeviceType::kCuda) {
#if defined(MW_INFER_HAS_CUDA_POSTPROCESS)
    return postprocess_internal::RunSoftmaxOnDevice(logits, allocator);
#else
    throw std::runtime_error("CUDA postprocess is unavailable in this build");
#endif
  }
  throw std::invalid_argument("Softmax tensor device is unsupported");
}

}  // namespace mw::infer
