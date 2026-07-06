#ifndef MW_INFER_RUNTIME_BLOCKS_H_
#define MW_INFER_RUNTIME_BLOCKS_H_

#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "mw/infer/common/geometry.h"
#include "mw/infer/common/model.h"
#include "mw/infer/runtime/infer_outputs.h"
#include "mw/infer/runtime/pipeline.h"
#include "mw/infer/runtime/runtime_config.h"

#ifndef MW_INFER_WITH_OPENCV_CUDA
#define MW_INFER_WITH_OPENCV_CUDA 0
#endif

namespace mw::infer {

using CpuImageBatch = std::vector<cv::Mat>;
using GpuImageBatch = std::vector<cv::cuda::GpuMat>;

template <typename InputType, typename OutputType, typename Function>
class FunctionBlock {
 public:
  using Input = InputType;
  using Output = OutputType;

  explicit FunctionBlock(Function function) : function_(std::move(function)) {}

  Output Run(const Input& input, RunContext& context) {
    if constexpr (std::is_invocable_r_v<Output, Function&, const Input&,
                                        RunContext&>) {
      return function_(input, context);
    } else {
      static_assert(std::is_invocable_r_v<Output, Function&, const Input&>,
                    "Function must accept either (const Input&, RunContext&) "
                    "or (const Input&).");
      return function_(input);
    }
  }

 private:
  Function function_;
};

template <typename Input, typename Output, typename Function>
FunctionBlock<Input, Output, std::decay_t<Function>> MakeFunction(
    Function&& function) {
  return FunctionBlock<Input, Output, std::decay_t<Function>>(
      std::forward<Function>(function));
}

class Resize : public IGeometryTransform<CpuImageBatch, CpuImageBatch> {
 public:
  using Input = CpuImageBatch;
  using Output = CpuImageBatch;

  explicit Resize(ImageSize size, int interpolation = cv::INTER_LINEAR);

  Output Run(const Input& input, RunContext&) const;
  GeometryUpdate GetGeometryUpdate(const Input& input,
                                   const Output& output) const override;

 private:
  ImageSize size_;
  int interpolation_ = cv::INTER_LINEAR;
};

class ResizeByShortEdge
    : public IGeometryTransform<CpuImageBatch, CpuImageBatch> {
 public:
  using Input = CpuImageBatch;
  using Output = CpuImageBatch;

  explicit ResizeByShortEdge(int short_edge,
                             int interpolation = cv::INTER_LINEAR);

  Output Run(const Input& input, RunContext&) const;
  GeometryUpdate GetGeometryUpdate(const Input& input,
                                   const Output& output) const override;

 private:
  int short_edge_ = 0;
  int interpolation_ = cv::INTER_LINEAR;
};

class CenterCrop : public IGeometryTransform<CpuImageBatch, CpuImageBatch> {
 public:
  using Input = CpuImageBatch;
  using Output = CpuImageBatch;

  explicit CenterCrop(ImageSize size);

  Output Run(const Input& input, RunContext&) const;
  GeometryUpdate GetGeometryUpdate(const Input& input,
                                   const Output& output) const override;

 private:
  ImageSize size_;
};

class LetterBox : public IGeometryTransform<CpuImageBatch, CpuImageBatch> {
 public:
  using Input = CpuImageBatch;
  using Output = CpuImageBatch;

  explicit LetterBox(ImageSize size = ImageSize{640, 640},
                     float pad_value = 114.0F,
                     int interpolation = cv::INTER_LINEAR);

  Output Run(const Input& input, RunContext&) const;
  GeometryUpdate GetGeometryUpdate(const Input& input,
                                   const Output& output) const override;

 private:
  ImageSize size_;
  float pad_value_ = 114.0F;
  int interpolation_ = cv::INTER_LINEAR;
};

class Normalize {
 public:
  using Input = CpuImageBatch;
  using Output = CpuImageBatch;

  Normalize(std::vector<float> mean, std::vector<float> std,
            float scale = 1.0F / 255.0F, bool to_rgb = true);

  Output Run(const Input& input, RunContext&) const;

 private:
  std::vector<float> mean_;
  std::vector<float> std_;
  float scale_ = 1.0F / 255.0F;
  bool to_rgb_ = true;
};

class OnnxInfer : public IGeometryTransform<CpuImageBatch, InferOutputs> {
 public:
  using Input = CpuImageBatch;
  using Output = InferOutputs;

  explicit OnnxInfer(RuntimeConfig config);
  OnnxInfer(Model model, BackendKind backend = BackendKind::kOnnxCpu,
            std::string input_name = {},
            std::vector<std::string> output_names = {});

  Output Run(const Input& input, RunContext& context);
  GeometryUpdate GetGeometryUpdate(const Input& input,
                                   const Output& output) const override;

 private:
  class Impl;

  std::shared_ptr<Impl> impl_;
};

#if MW_INFER_WITH_OPENCV_CUDA

class UploadCuda {
 public:
  using Input = CpuImageBatch;
  using Output = GpuImageBatch;

  Output Run(const Input& input, RunContext&) const;
};

class DownloadCuda {
 public:
  using Input = GpuImageBatch;
  using Output = CpuImageBatch;

  Output Run(const Input& input, RunContext&) const;
};

class CudaResize : public IGeometryTransform<GpuImageBatch, GpuImageBatch> {
 public:
  using Input = GpuImageBatch;
  using Output = GpuImageBatch;

  explicit CudaResize(ImageSize size, int interpolation = cv::INTER_LINEAR);

  Output Run(const Input& input, RunContext&) const;
  GeometryUpdate GetGeometryUpdate(const Input& input,
                                   const Output& output) const override;

 private:
  ImageSize size_;
  int interpolation_ = cv::INTER_LINEAR;
};

class CudaLetterBox : public IGeometryTransform<GpuImageBatch, GpuImageBatch> {
 public:
  using Input = GpuImageBatch;
  using Output = GpuImageBatch;

  explicit CudaLetterBox(ImageSize size = ImageSize{640, 640},
                         float pad_value = 114.0F,
                         int interpolation = cv::INTER_LINEAR);

  Output Run(const Input& input, RunContext&) const;
  GeometryUpdate GetGeometryUpdate(const Input& input,
                                   const Output& output) const override;

 private:
  ImageSize size_;
  float pad_value_ = 114.0F;
  int interpolation_ = cv::INTER_LINEAR;
};

class OnnxGpuInfer : public IGeometryTransform<GpuImageBatch, InferOutputs> {
 public:
  using Input = GpuImageBatch;
  using Output = InferOutputs;

  explicit OnnxGpuInfer(RuntimeConfig config);
  OnnxGpuInfer(Model model, BackendKind backend = BackendKind::kOnnxGpu,
               std::string input_name = {},
               std::vector<std::string> output_names = {});

  Output Run(const Input& input, RunContext& context);
  GeometryUpdate GetGeometryUpdate(const Input& input,
                                   const Output& output) const override;

 private:
  class Impl;

  std::shared_ptr<Impl> impl_;
};

#endif  // MW_INFER_WITH_OPENCV_CUDA

}  // namespace mw::infer

#endif  // MW_INFER_RUNTIME_BLOCKS_H_
