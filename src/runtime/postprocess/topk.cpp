#include "mw/infer/runtime/postprocess/topk.h"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mw::infer {

#if defined(MW_INFER_HAS_CUDA_POSTPROCESS)
namespace postprocess_internal {
TopKResult RunTopKOnDevice(const Tensor& scores, int64_t k);
}  // namespace postprocess_internal
#endif

namespace {

struct MatrixShape {
  int64_t rows = 0;
  int64_t columns = 0;
  bool batched = false;
};

MatrixShape ValidateScores(const Tensor& scores) {
  if (scores.empty()) {
    throw std::invalid_argument("TopK tensor is empty");
  }
  if (scores.data_type() != DataType::kFloat32) {
    throw std::invalid_argument("TopK tensor must be float32");
  }
  if (scores.shape().size() != 1 && scores.shape().size() != 2) {
    throw std::invalid_argument("TopK tensor shape must be [C] or [N, C]");
  }

  MatrixShape shape;
  shape.batched = scores.shape().size() == 2;
  shape.rows = shape.batched ? scores.shape()[0] : 1;
  shape.columns = shape.batched ? scores.shape()[1] : scores.shape()[0];
  if (shape.rows <= 0 || shape.columns <= 0) {
    throw std::invalid_argument("TopK tensor dimensions must be positive");
  }
  return shape;
}

void ValidateK(int64_t k, int64_t columns) {
  if (k <= 0) {
    throw std::invalid_argument("TopK k must be positive");
  }
  if (k > columns) {
    throw std::invalid_argument("TopK k must not exceed class count");
  }
}

TensorDesc MakeFloatDesc(std::vector<int64_t> shape, Device device) {
  TensorDesc desc;
  desc.info.name = "topk_scores";
  desc.info.data_type = DataType::kFloat32;
  desc.info.shape = std::move(shape);
  desc.device = device;
  return desc;
}

TensorDesc MakeIndexDesc(std::vector<int64_t> shape, Device device) {
  TensorDesc desc;
  desc.info.name = "topk_indices";
  desc.info.data_type = DataType::kInt64;
  desc.info.shape = std::move(shape);
  desc.device = device;
  return desc;
}

std::vector<int64_t> MakeTopKShape(MatrixShape shape, int64_t k) {
  if (!shape.batched) {
    return {k};
  }
  return {shape.rows, k};
}

std::vector<int64_t> MakeScoreOrder(const float* scores, int64_t columns) {
  std::vector<int64_t> order(static_cast<std::size_t>(columns));
  std::iota(order.begin(), order.end(), int64_t{0});
  std::sort(order.begin(), order.end(), [scores](int64_t lhs, int64_t rhs) {
    const float lhs_score = scores[lhs];
    const float rhs_score = scores[rhs];
    if (lhs_score == rhs_score) {
      return lhs < rhs;
    }
    return lhs_score > rhs_score;
  });
  return order;
}

TopKResult RunTopKOnHost(const Tensor& scores, MatrixShape shape, int64_t k) {
  const std::vector<int64_t> output_shape = MakeTopKShape(shape, k);
  Tensor top_scores =
      Tensor::Allocate(MakeFloatDesc(output_shape, scores.device()));
  Tensor top_indices =
      Tensor::Allocate(MakeIndexDesc(output_shape, scores.device()));

  const auto* input = static_cast<const float*>(scores.data());
  auto* output_scores = static_cast<float*>(top_scores.data());
  auto* output_indices = static_cast<int64_t*>(top_indices.data());
  for (int64_t row = 0; row < shape.rows; ++row) {
    const auto row_offset = static_cast<std::size_t>(row * shape.columns);
    std::vector<int64_t> order =
        MakeScoreOrder(input + row_offset, shape.columns);
    const auto output_offset = static_cast<std::size_t>(row * k);
    for (int64_t index = 0; index < k; ++index) {
      const int64_t class_index = order[static_cast<std::size_t>(index)];
      output_scores[output_offset + static_cast<std::size_t>(index)] =
          input[row_offset + static_cast<std::size_t>(class_index)];
      output_indices[output_offset + static_cast<std::size_t>(index)] =
          class_index;
    }
  }

  return TopKResult{std::move(top_scores), std::move(top_indices)};
}

}  // namespace

TopKResult TopK(const Tensor& scores, int64_t k) {
  const MatrixShape shape = ValidateScores(scores);
  ValidateK(k, shape.columns);
  if (scores.device().type == DeviceType::kCpu) {
    return RunTopKOnHost(scores, shape, k);
  }
  if (scores.device().type == DeviceType::kCuda) {
#if defined(MW_INFER_HAS_CUDA_POSTPROCESS)
    return postprocess_internal::RunTopKOnDevice(scores, k);
#else
    throw std::runtime_error("CUDA postprocess is unavailable in this build");
#endif
  }
  throw std::invalid_argument("TopK tensor device is unsupported");
}

}  // namespace mw::infer
