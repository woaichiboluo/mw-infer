#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "mw/infer/runtime/backend.h"
#include "mw/infer/runtime/blocks.h"

namespace mw::infer {
namespace {

std::vector<ImageSize> GetBatchSizes(const std::vector<cv::Mat>& images) {
  if (images.empty()) {
    throw std::invalid_argument("OnnxInfer input batch is empty");
  }

  std::vector<ImageSize> sizes;
  sizes.reserve(images.size());
  for (const cv::Mat& image : images) {
    if (image.empty()) {
      throw std::invalid_argument("OnnxInfer input image is empty");
    }
    sizes.push_back(ImageSize{image.cols, image.rows});
  }
  return sizes;
}

#if MW_INFER_WITH_OPENCV_CUDA

std::vector<ImageSize> GetBatchSizes(
    const std::vector<cv::cuda::GpuMat>& images) {
  if (images.empty()) {
    throw std::invalid_argument("OnnxGpuInfer input batch is empty");
  }

  std::vector<ImageSize> sizes;
  sizes.reserve(images.size());
  for (const cv::cuda::GpuMat& image : images) {
    if (image.empty()) {
      throw std::invalid_argument("OnnxGpuInfer input image is empty");
    }
    sizes.push_back(ImageSize{image.cols, image.rows});
  }
  return sizes;
}

#endif  // MW_INFER_WITH_OPENCV_CUDA

}  // namespace

class OnnxInfer::Impl {
 public:
  explicit Impl(RuntimeConfig config)
      : backend_(CreateBackend(config)), config_(std::move(config)) {}

  InferOutputs Run(const std::vector<cv::Mat>& input) {
    return backend_->InferBatch(input);
  }

 private:
  std::unique_ptr<IBackend> backend_;
  RuntimeConfig config_;
};

OnnxInfer::OnnxInfer(RuntimeConfig config)
    : impl_(std::make_shared<Impl>(std::move(config))) {}

OnnxInfer::OnnxInfer(Model model, BackendKind backend, std::string input_name,
                     std::vector<std::string> output_names) {
  RuntimeConfig config;
  config.backend = backend;
  config.model = std::move(model);
  config.input_name = std::move(input_name);
  config.output_names = std::move(output_names);
  impl_ = std::make_shared<Impl>(std::move(config));
}

OnnxInfer::Output OnnxInfer::Run(const Input& input, RunContext&) {
  return impl_->Run(input);
}

GeometryUpdate OnnxInfer::GetGeometryUpdate(const Input& input,
                                            const Output&) const {
  return GeometryUpdate::FromSource(GetBatchSizes(input));
}

#if MW_INFER_WITH_OPENCV_CUDA

class OnnxGpuInfer::Impl {
 public:
  explicit Impl(RuntimeConfig config)
      : backend_(CreateBackend(config)), config_(std::move(config)) {}

  InferOutputs Run(const std::vector<cv::cuda::GpuMat>& input) {
    return backend_->InferBatch(input);
  }

 private:
  std::unique_ptr<IBackend> backend_;
  RuntimeConfig config_;
};

OnnxGpuInfer::OnnxGpuInfer(RuntimeConfig config)
    : impl_(std::make_shared<Impl>(std::move(config))) {}

OnnxGpuInfer::OnnxGpuInfer(Model model, BackendKind backend,
                           std::string input_name,
                           std::vector<std::string> output_names) {
  RuntimeConfig config;
  config.backend = backend;
  config.model = std::move(model);
  config.input_name = std::move(input_name);
  config.output_names = std::move(output_names);
  impl_ = std::make_shared<Impl>(std::move(config));
}

OnnxGpuInfer::Output OnnxGpuInfer::Run(const Input& input, RunContext&) {
  return impl_->Run(input);
}

GeometryUpdate OnnxGpuInfer::GetGeometryUpdate(const Input& input,
                                               const Output&) const {
  return GeometryUpdate::FromSource(GetBatchSizes(input));
}

#endif  // MW_INFER_WITH_OPENCV_CUDA

}  // namespace mw::infer
