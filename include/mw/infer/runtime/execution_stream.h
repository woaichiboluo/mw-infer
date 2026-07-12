#ifndef MW_INFER_RUNTIME_EXECUTION_STREAM_H_
#define MW_INFER_RUNTIME_EXECUTION_STREAM_H_

#if defined(MW_INFER_HAS_CUDA_RUNTIME)
#include <cuda_runtime_api.h>
#endif

#include "mw/infer/runtime/tensor/tensor.h"

namespace mw::infer {

enum class CudaStreamMode {
  kBlocking,
  kNonBlocking,
};

class ExecutionStream final {
 public:
  explicit ExecutionStream(Device device);
  ExecutionStream(Device device, CudaStreamMode cuda_mode);
  ~ExecutionStream();

  ExecutionStream(const ExecutionStream&) = delete;
  ExecutionStream& operator=(const ExecutionStream&) = delete;
  ExecutionStream(ExecutionStream&&) = delete;
  ExecutionStream& operator=(ExecutionStream&&) = delete;

  Device device() const { return device_; }
  bool is_cpu() const { return device_.type == DeviceType::kCpu; }
  bool is_cuda() const { return device_.type == DeviceType::kCuda; }

  void Synchronize();
  void SynchronizeNoThrow() noexcept;

#if defined(MW_INFER_HAS_CUDA_RUNTIME)
  cudaStream_t cuda_handle() const;
#endif

 private:
  Device device_;
#if defined(MW_INFER_HAS_CUDA_RUNTIME)
  cudaStream_t cuda_stream_ = nullptr;
#endif
};

}  // namespace mw::infer

#endif  // MW_INFER_RUNTIME_EXECUTION_STREAM_H_
