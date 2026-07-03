#ifndef MW_INFER_BUILDER_BUILDER_H_
#define MW_INFER_BUILDER_BUILDER_H_

#include "mw/infer/builder/build_config.h"

namespace mw::infer::builder {

class ModelBuilder {
 public:
  ModelBuilder() = default;

  void Build(const BuildConfig& config) const;
};

}  // namespace mw::infer::builder

#endif  // MW_INFER_BUILDER_BUILDER_H_
