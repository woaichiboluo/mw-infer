#ifndef MW_INFER_SRC_BACKENDS_BACKEND_REGISTRY_H_
#define MW_INFER_SRC_BACKENDS_BACKEND_REGISTRY_H_

#include <memory>

#include "mw/infer/runtime/backend.h"
#include "mw/infer/runtime/runtime_config.h"

namespace mw::infer {

std::unique_ptr<IBackend> CreateOnnxCpuBackend(const RuntimeConfig& config);
std::unique_ptr<IBackend> CreateOnnxGpuBackend(const RuntimeConfig& config);
std::unique_ptr<IBackend> CreateTensorRTBackend(const RuntimeConfig& config);

}  // namespace mw::infer

#endif  // MW_INFER_SRC_BACKENDS_BACKEND_REGISTRY_H_
