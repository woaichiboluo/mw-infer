#include "mw/infer/builder/builder.h"

#include <stdexcept>

namespace mw::infer::builder {

void ModelBuilder::Build(const BuildConfig& config) const {
  ValidateBuildConfig(config);
  throw std::runtime_error("ModelBuilder::Build is not implemented yet");
}

}  // namespace mw::infer::builder
