#ifndef MW_INFER_PIPELINE_H_
#define MW_INFER_PIPELINE_H_

#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <typeindex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mw/infer/common/geometry.h"

namespace mw::infer {

template <typename InputType, typename OutputType>
class IGeometryTransform {
 public:
  virtual ~IGeometryTransform() = default;

  // Implement this interface when a block changes image coordinate space or
  // bridges image input to coordinate-sensitive model output.
  // Examples: resize, crop, pad, letterbox, rotate, affine transform.
  // Do not implement it for color conversion, normalize, layout conversion,
  // classification, or other coordinate-preserving blocks.
  virtual GeometryUpdate GetGeometryUpdate(const InputType& input,
                                           const OutputType& output) const = 0;
};

class RunContext {
 public:
  RunContext() = default;

  template <typename T>
  const T* RootInput() const {
    return root_input_.Get<T>();
  }

  template <typename T>
  const T* OutputOf(std::string_view block_name) const {
    return GetArtifact<T>(outputs_, block_name);
  }

  bool HasGeometry() const { return has_geometry_; }

  const std::vector<ImageGeometry>& geometries() const { return geometries_; }

  const ImageGeometry* GeometryAt(std::size_t index) const {
    if (!has_geometry_ || index >= geometries_.size()) {
      return nullptr;
    }
    return &geometries_[index];
  }

 private:
  template <typename RootInput, typename CurrentOutput>
  friend class PipelineSession;

  void ResetGeometry(const std::vector<ImageSize>& sizes) {
    geometries_.clear();
    geometries_.reserve(sizes.size());
    for (ImageSize size : sizes) {
      geometries_.push_back(MakeImageGeometry(size));
    }
    has_geometry_ = true;
  }

  void EnsureGeometry(const std::vector<ImageSize>& sizes) {
    if (!has_geometry_) {
      ResetGeometry(sizes);
      return;
    }
    if (geometries_.size() != sizes.size()) {
      throw std::logic_error("pipeline geometry size mismatch");
    }
    for (std::size_t index = 0; index < sizes.size(); ++index) {
      const ImageSize current_size = geometries_[index].current_size();
      if (current_size.width != sizes[index].width ||
          current_size.height != sizes[index].height) {
        throw std::logic_error("pipeline geometry size mismatch");
      }
    }
  }

  void ResizeGeometries(const std::vector<ImageSize>& new_sizes) {
    std::vector<AffineTransform> transforms;
    transforms.reserve(new_sizes.size());
    ValidateGeometryUpdateSize(new_sizes.size());
    for (std::size_t index = 0; index < new_sizes.size(); ++index) {
      const ImageSize old_size = geometries_[index].current_size();
      transforms.push_back(
          ScaleAffine(static_cast<float>(new_sizes[index].width) /
                          static_cast<float>(old_size.width),
                      static_cast<float>(new_sizes[index].height) /
                          static_cast<float>(old_size.height)));
    }
    ApplyImageTransforms(new_sizes, transforms);
  }

  void CropGeometries(const std::vector<RectF>& crops) {
    std::vector<ImageSize> new_sizes;
    std::vector<AffineTransform> transforms;
    new_sizes.reserve(crops.size());
    transforms.reserve(crops.size());
    ValidateGeometryUpdateSize(crops.size());
    for (const RectF& crop : crops) {
      if (crop.width <= 0.0F || crop.height <= 0.0F) {
        throw std::logic_error("pipeline crop geometry size is invalid");
      }
      new_sizes.push_back(
          ImageSize{static_cast<int32_t>(std::round(crop.width)),
                    static_cast<int32_t>(std::round(crop.height))});
      transforms.push_back(TranslateAffine(-crop.x, -crop.y));
    }
    ApplyImageTransforms(new_sizes, transforms);
  }

  void PadGeometries(const std::vector<ImageSize>& new_sizes,
                     const std::vector<PointF>& offsets) {
    std::vector<AffineTransform> transforms;
    transforms.reserve(offsets.size());
    ValidateGeometryUpdateSize(new_sizes.size());
    if (offsets.size() != new_sizes.size()) {
      throw std::logic_error("pipeline pad geometry update size mismatch");
    }
    for (PointF offset : offsets) {
      transforms.push_back(TranslateAffine(offset.x, offset.y));
    }
    ApplyImageTransforms(new_sizes, transforms);
  }

  struct Artifact {
    std::type_index type = std::type_index(typeid(void));
    std::shared_ptr<const void> value;

    template <typename T>
    const T* Get() const {
      using ValueType = std::decay_t<T>;
      if (!value || type != std::type_index(typeid(ValueType))) {
        return nullptr;
      }
      return static_cast<const ValueType*>(value.get());
    }
  };

  using ArtifactMap = std::unordered_map<std::string, Artifact>;

  void ValidateGeometryUpdateSize(std::size_t size) const {
    if (!has_geometry_) {
      throw std::logic_error("pipeline geometry is not initialized");
    }
    if (geometries_.size() != size) {
      throw std::logic_error("pipeline geometry update size mismatch");
    }
  }

  void ApplyImageTransforms(
      const std::vector<ImageSize>& new_sizes,
      const std::vector<AffineTransform>& previous_to_current) {
    ValidateGeometryUpdateSize(new_sizes.size());
    if (previous_to_current.size() != new_sizes.size()) {
      throw std::logic_error("pipeline geometry update size mismatch");
    }
    for (std::size_t index = 0; index < geometries_.size(); ++index) {
      geometries_[index] = ApplyImageTransform(
          geometries_[index], new_sizes[index], previous_to_current[index]);
    }
  }

  void ApplyGeometryUpdate(const GeometryUpdate& update) {
    if (!update.source_sizes.empty()) {
      EnsureGeometry(update.source_sizes);
    }
    for (const GeometryUpdateStep& step : update.steps) {
      switch (step.kind) {
        case GeometryUpdateStepKind::kResize:
          ResizeGeometries(step.sizes);
          break;
        case GeometryUpdateStepKind::kCrop:
          CropGeometries(step.crops);
          break;
        case GeometryUpdateStepKind::kPad:
          PadGeometries(step.sizes, step.offsets);
          break;
      }
    }
  }

  template <typename T>
  void SetRootInput(std::shared_ptr<T> input) {
    root_input_ = MakeArtifact(std::move(input));
  }

  template <typename T>
  void RecordOutput(std::string_view block_name, std::shared_ptr<T> output) {
    RecordArtifact(&outputs_, block_name, MakeArtifact(std::move(output)));
  }

  template <typename T>
  static Artifact MakeArtifact(std::shared_ptr<T> value) {
    using ValueType = std::remove_cv_t<T>;
    std::shared_ptr<const ValueType> const_value = std::move(value);
    return Artifact{std::type_index(typeid(ValueType)), std::move(const_value)};
  }

  static void RecordArtifact(ArtifactMap* artifacts, std::string_view name,
                             Artifact artifact) {
    auto [unused, inserted] =
        artifacts->emplace(std::string(name), std::move(artifact));
    if (!inserted) {
      throw std::invalid_argument("duplicate pipeline block name");
    }
  }

  template <typename T>
  static const T* GetArtifact(const ArtifactMap& artifacts,
                              std::string_view name) {
    const auto iter = artifacts.find(std::string(name));
    if (iter == artifacts.end()) {
      return nullptr;
    }
    return iter->second.Get<T>();
  }

  Artifact root_input_;
  ArtifactMap outputs_;
  std::vector<ImageGeometry> geometries_;
  bool has_geometry_ = false;
};

using PipelineRunContext = RunContext;
using RunScope = RunContext;

template <typename Output>
struct PipelineRunResult {
  Output output;
  RunContext context;
};

template <typename RootInput, typename CurrentOutput = RootInput>
class PipelineSession {
 public:
  PipelineSession() = default;

  template <typename Block>
  auto Then(std::string block_name, Block block) && {
    using BlockType = std::decay_t<Block>;
    using BlockInput = typename BlockType::Input;
    using BlockOutput = typename BlockType::Output;
    static_assert(std::is_same_v<BlockInput, CurrentOutput>,
                  "Block input must match the current pipeline output. Use "
                  "ThenWith() to bind a custom input.");

    auto steps = std::move(steps_);
    auto block_ptr = std::make_shared<BlockType>(std::move(block));
    steps.push_back(
        MakeStep<BlockInput, BlockOutput>(std::move(block_name), block_ptr));
    return PipelineSession<RootInput, BlockOutput>(std::move(steps));
  }

  template <typename Block, typename Binder>
  auto ThenWith(std::string block_name, Block block, Binder binder) && {
    using BlockType = std::decay_t<Block>;
    using BinderType = std::decay_t<Binder>;
    using BlockInput = typename BlockType::Input;
    using BlockOutput = typename BlockType::Output;

    static_assert(std::is_invocable_r_v<BlockInput, BinderType,
                                        const CurrentOutput&, RunContext&>,
                  "Binder must return Block::Input from "
                  "(const CurrentOutput&, RunContext&).");

    auto steps = std::move(steps_);
    auto block_ptr = std::make_shared<BlockType>(std::move(block));
    auto binder_ptr = std::make_shared<BinderType>(std::move(binder));
    steps.push_back(MakeBoundStep<BlockInput, BlockOutput, CurrentOutput>(
        std::move(block_name), block_ptr, binder_ptr));
    return PipelineSession<RootInput, BlockOutput>(std::move(steps));
  }

  CurrentOutput Run(const RootInput& input) const {
    return RunWithContext(input).output;
  }

  PipelineRunResult<CurrentOutput> RunWithContext(
      const RootInput& input) const {
    RunContext context;
    auto root_input = std::make_shared<RootInput>(input);
    context.SetRootInput(root_input);

    std::shared_ptr<const void> current = root_input;
    for (const auto& step : steps_) {
      current = step(current, context);
    }

    auto output = std::static_pointer_cast<const CurrentOutput>(current);
    return PipelineRunResult<CurrentOutput>{*output, std::move(context)};
  }

 private:
  using Step = std::function<std::shared_ptr<const void>(
      std::shared_ptr<const void>, RunContext&)>;

  template <typename OtherRootInput, typename OtherCurrentOutput>
  friend class PipelineSession;

  explicit PipelineSession(std::vector<Step> steps)
      : steps_(std::move(steps)) {}

  template <typename BlockInput, typename BlockOutput, typename BlockType>
  static Step MakeStep(std::string block_name,
                       std::shared_ptr<BlockType> block) {
    return [block_name = std::move(block_name), block = std::move(block)](
               std::shared_ptr<const void> current, RunContext& context) {
      auto input = std::static_pointer_cast<const BlockInput>(current);

      auto output_value = block->Run(*input, context);
      auto output = std::make_shared<BlockOutput>(std::move(output_value));
      MaybeApplyGeometryUpdate(*block, *input, *output, context);
      context.RecordOutput(block_name, output);
      return std::static_pointer_cast<const void>(output);
    };
  }

  template <typename BlockInput, typename BlockOutput, typename Input,
            typename BlockType, typename BinderType>
  static Step MakeBoundStep(std::string block_name,
                            std::shared_ptr<BlockType> block,
                            std::shared_ptr<BinderType> binder) {
    return [block_name = std::move(block_name), block = std::move(block),
            binder = std::move(binder)](std::shared_ptr<const void> current,
                                        RunContext& context) {
      auto input = std::static_pointer_cast<const Input>(current);
      auto bound_input_value = (*binder)(*input, context);
      auto bound_input =
          std::make_shared<BlockInput>(std::move(bound_input_value));

      auto output_value = block->Run(*bound_input, context);
      auto output = std::make_shared<BlockOutput>(std::move(output_value));
      MaybeApplyGeometryUpdate(*block, *bound_input, *output, context);
      context.RecordOutput(block_name, output);
      return std::static_pointer_cast<const void>(output);
    };
  }

  template <typename BlockType, typename BlockInput, typename BlockOutput>
  static void MaybeApplyGeometryUpdate(const BlockType& block,
                                       const BlockInput& input,
                                       const BlockOutput& output,
                                       RunContext& context) {
    using GeometryTransform = IGeometryTransform<BlockInput, BlockOutput>;
    if constexpr (std::is_base_of_v<GeometryTransform, BlockType>) {
      const GeometryTransform& transform = block;
      context.ApplyGeometryUpdate(transform.GetGeometryUpdate(input, output));
    }
  }

  std::vector<Step> steps_;
};

template <typename RootInput>
PipelineSession<RootInput> MakePipeline() {
  return PipelineSession<RootInput>();
}

template <typename RootInput>
PipelineSession<RootInput> MakeSerialPipeline() {
  return MakePipeline<RootInput>();
}

}  // namespace mw::infer

#endif  // MW_INFER_PIPELINE_H_
