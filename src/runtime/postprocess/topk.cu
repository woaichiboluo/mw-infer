#include <cuda_runtime_api.h>

#include <cfloat>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mw/infer/runtime/postprocess/topk.h"

namespace mw::infer::postprocess_internal {
namespace {

std::string CudaErrorMessage(cudaError_t status, const char* operation) {
  return std::string(operation) + " failed: " + cudaGetErrorString(status);
}

void CheckCuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(CudaErrorMessage(status, operation));
  }
}

struct MatrixShape {
  int rows = 0;
  int columns = 0;
  bool batched = false;
};

MatrixShape CheckedMatrixShape(const Tensor& tensor) {
  MatrixShape shape;
  shape.batched = tensor.shape().size() == 2;
  const int64_t rows = shape.batched ? tensor.shape()[0] : 1;
  const int64_t columns = shape.batched ? tensor.shape()[1] : tensor.shape()[0];
  if (rows > std::numeric_limits<int>::max()) {
    throw std::invalid_argument(
        "CUDA postprocess batch size exceeds int range");
  }
  if (columns > std::numeric_limits<int>::max()) {
    throw std::invalid_argument(
        "CUDA postprocess class count exceeds int range");
  }
  shape.rows = static_cast<int>(rows);
  shape.columns = static_cast<int>(columns);
  return shape;
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

std::vector<int64_t> MakeOutputShape(MatrixShape shape, int64_t k) {
  if (!shape.batched) {
    return {k};
  }
  return {shape.rows, k};
}

__global__ void TopKKernel(const float* scores, float* top_scores,
                           int64_t* top_indices, int rows, int columns, int k) {
  const int row = blockIdx.x;
  if (row >= rows || threadIdx.x != 0) {
    return;
  }

  const int row_offset = row * columns;
  const int output_offset = row * k;
  for (int output_index = 0; output_index < k; ++output_index) {
    int best_index = -1;
    float best_score = -FLT_MAX;
    for (int column = 0; column < columns; ++column) {
      bool already_selected = false;
      for (int previous = 0; previous < output_index; ++previous) {
        if (top_indices[output_offset + previous] == column) {
          already_selected = true;
          break;
        }
      }
      if (already_selected) {
        continue;
      }

      const float candidate = scores[row_offset + column];
      if (best_index < 0 || candidate > best_score ||
          (candidate == best_score && column < best_index)) {
        best_index = column;
        best_score = candidate;
      }
    }

    top_scores[output_offset + output_index] = best_score;
    top_indices[output_offset + output_index] = best_index;
  }
}

}  // namespace

TopKResult RunTopKOnDevice(const Tensor& scores, int64_t k) {
  const MatrixShape shape = CheckedMatrixShape(scores);
  if (k > std::numeric_limits<int>::max()) {
    throw std::invalid_argument("CUDA postprocess k exceeds int range");
  }
  CheckCuda(cudaSetDevice(scores.device().id), "cudaSetDevice");

  const std::vector<int64_t> output_shape = MakeOutputShape(shape, k);
  Tensor top_scores =
      Tensor::Allocate(MakeFloatDesc(output_shape, scores.device()));
  Tensor top_indices =
      Tensor::Allocate(MakeIndexDesc(output_shape, scores.device()));

  TopKKernel<<<shape.rows, 1>>>(static_cast<const float*>(scores.data()),
                                static_cast<float*>(top_scores.data()),
                                static_cast<int64_t*>(top_indices.data()),
                                shape.rows, shape.columns, static_cast<int>(k));
  CheckCuda(cudaGetLastError(), "TopKKernel");
  return TopKResult{std::move(top_scores), std::move(top_indices)};
}

}  // namespace mw::infer::postprocess_internal
