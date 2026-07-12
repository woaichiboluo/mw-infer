#include "mw/infer/runtime/execution_stream.h"

#include <stdexcept>
#include <string>

namespace mw::infer {

namespace {

#if defined(MW_INFER_HAS_CUDA_RUNTIME)
void CheckCuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) +
                             " failed: " + cudaGetErrorString(status));
  }
}
#endif

}  // namespace

ExecutionStream::ExecutionStream(Device device)
    : ExecutionStream(device, CudaStreamMode::kNonBlocking) {}

ExecutionStream::ExecutionStream(Device device, CudaStreamMode cuda_mode)
    : device_(device) {
  if (device_.id < 0) {
    throw std::invalid_argument("Execution stream device id is negative");
  }
  if (device_.type == DeviceType::kCpu) {
    return;
  }
  if (device_.type != DeviceType::kCuda) {
    throw std::invalid_argument("Execution stream device is unsupported");
  }

#if defined(MW_INFER_HAS_CUDA_RUNTIME)
  CheckCuda(cudaSetDevice(device_.id), "cudaSetDevice");
  const unsigned int flags = cuda_mode == CudaStreamMode::kNonBlocking
                                 ? cudaStreamNonBlocking
                                 : cudaStreamDefault;
  CheckCuda(cudaStreamCreateWithFlags(&cuda_stream_, flags),
            "cudaStreamCreateWithFlags");
#else
  throw std::runtime_error(
      "CUDA execution stream is unavailable in this build");
#endif
}

ExecutionStream::~ExecutionStream() {
#if defined(MW_INFER_HAS_CUDA_RUNTIME)
  if (cuda_stream_ != nullptr) {
    SynchronizeNoThrow();
    static_cast<void>(cudaSetDevice(device_.id));
    static_cast<void>(cudaStreamDestroy(cuda_stream_));
  }
#endif
}

void ExecutionStream::Synchronize() {
  if (device_.type == DeviceType::kCpu) {
    return;
  }

#if defined(MW_INFER_HAS_CUDA_RUNTIME)
  CheckCuda(cudaSetDevice(device_.id), "cudaSetDevice");
  CheckCuda(cudaStreamSynchronize(cuda_stream_), "cudaStreamSynchronize");
#else
  throw std::runtime_error(
      "CUDA execution stream is unavailable in this build");
#endif
}

void ExecutionStream::SynchronizeNoThrow() noexcept {
#if defined(MW_INFER_HAS_CUDA_RUNTIME)
  if (cuda_stream_ != nullptr) {
    static_cast<void>(cudaSetDevice(device_.id));
    static_cast<void>(cudaStreamSynchronize(cuda_stream_));
  }
#endif
}

#if defined(MW_INFER_HAS_CUDA_RUNTIME)
cudaStream_t ExecutionStream::cuda_handle() const {
  if (device_.type != DeviceType::kCuda || cuda_stream_ == nullptr) {
    throw std::logic_error("Execution stream is not a CUDA stream");
  }
  return cuda_stream_;
}
#endif

}  // namespace mw::infer
