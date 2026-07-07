#ifndef MW_INFER_RUNTIME_CUDA_TENSOR_H_
#define MW_INFER_RUNTIME_CUDA_TENSOR_H_

#include "mw/infer/common/tensor_allocator.h"

namespace mw::infer {

bool IsCudaTensorAllocationAvailable();
Tensor AllocateCudaTensor(TensorDesc desc);
const TensorAllocationAdapter& GetCudaTensorAllocationAdapter();
TensorAllocator MakeCudaTensorAllocator();

}  // namespace mw::infer

#endif  // MW_INFER_RUNTIME_CUDA_TENSOR_H_
