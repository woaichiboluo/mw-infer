#ifndef MW_INFER_RUNTIME_PROCESS_CHANNEL_H_
#define MW_INFER_RUNTIME_PROCESS_CHANNEL_H_

#include <cstdint>
#include <vector>

#include "mw/infer/runtime/execution_stream.h"
#include "mw/infer/runtime/process/image_to_tensor.h"
#include "mw/infer/runtime/tensor/tensor.h"

namespace mw::infer {

Tensor ReorderChannels(const Tensor& input, const std::vector<int64_t>& order,
                       TensorLayout layout = TensorLayout::kBchw,
                       TensorAllocator& allocator = TensorAllocator::Default());

// CUDA work is enqueued on stream. Keep output alive until the stream is
// synchronized. CPU work remains synchronous.
Tensor ReorderChannels(const Tensor& input, const std::vector<int64_t>& order,
                       ExecutionStream& stream,
                       TensorLayout layout = TensorLayout::kBchw,
                       TensorAllocator& allocator = TensorAllocator::Default());

}  // namespace mw::infer

#endif  // MW_INFER_RUNTIME_PROCESS_CHANNEL_H_
