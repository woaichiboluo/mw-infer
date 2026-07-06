#include <CLI/CLI.hpp>
#include <algorithm>
#include <cstddef>
#include <exception>
#include <fstream>
#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgcodecs.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mw/infer/mw_infer.h"

namespace {

struct Classification {
  std::size_t class_index = 0;
  float score = 0.0F;
  std::string label;
};

using ClassificationBatch = std::vector<std::vector<Classification>>;

mw::infer::BackendKind ParseBackend(const std::string& backend) {
  if (backend == "onnx_cpu") {
    return mw::infer::BackendKind::kOnnxCpu;
  }
  if (backend == "onnx_gpu" || backend == "onnx_cuda") {
    return mw::infer::BackendKind::kOnnxGpu;
  }
  throw std::invalid_argument("unsupported backend: " + backend);
}

std::vector<std::string> LoadLabels(const std::string& path) {
  if (path.empty()) {
    return {};
  }

  std::ifstream file(path);
  if (!file) {
    throw std::invalid_argument("failed to read labels: " + path);
  }

  std::vector<std::string> labels;
  std::string line;
  while (std::getline(file, line)) {
    labels.push_back(line);
  }
  return labels;
}

std::vector<cv::Mat> LoadImages(const std::vector<std::string>& image_paths) {
  if (image_paths.empty()) {
    throw std::invalid_argument("image path list is empty");
  }

  std::vector<cv::Mat> images;
  images.reserve(image_paths.size());
  for (const std::string& image_path : image_paths) {
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
      throw std::invalid_argument("failed to read image: " + image_path);
    }
    images.push_back(std::move(image));
  }
  return images;
}

std::vector<float> CopyDeviceOutputToHost(
    const mw::infer::InferOutput& output) {
#if MW_INFER_WITH_OPENCV_CUDA
  const std::size_t count = output.buffer.size_bytes / sizeof(float);
  std::vector<float> values(count);
  cv::cuda::GpuMat device_values(1, static_cast<int>(count), CV_32FC1,
                                 const_cast<void*>(output.buffer.device),
                                 count * sizeof(float));
  cv::Mat host_values;
  device_values.download(host_values);
  const auto* data = host_values.ptr<float>();
  std::copy(data, data + count, values.begin());
  return values;
#else
  throw std::runtime_error("device output requires OpenCV CUDA support");
#endif
}

std::vector<float> CopyOutputToHost(const mw::infer::InferOutput& output) {
  const std::size_t count = output.buffer.size_bytes / sizeof(float);
  std::vector<float> values(count);
  if (output.buffer.host != nullptr) {
    const auto* data = static_cast<const float*>(output.buffer.host);
    std::copy(data, data + count, values.begin());
    return values;
  }

  if (output.buffer.device != nullptr) {
    return CopyDeviceOutputToHost(output);
  }

  throw std::runtime_error("classification output buffer is empty");
}

std::vector<std::size_t> TopK(const float* scores, std::size_t count,
                              std::size_t top_k) {
  std::vector<std::size_t> indices(count);
  std::iota(indices.begin(), indices.end(), 0);

  const std::size_t selected = std::min(top_k, count);
  std::partial_sort(indices.begin(), indices.begin() + selected, indices.end(),
                    [scores](std::size_t lhs, std::size_t rhs) {
                      return scores[lhs] > scores[rhs];
                    });
  indices.resize(selected);
  return indices;
}

std::string LabelForIndex(const std::vector<std::string>& labels,
                          std::size_t class_index) {
  if (class_index >= labels.size()) {
    return "";
  }
  return labels[class_index];
}

ClassificationBatch MakeTopK(const mw::infer::InferOutput& output,
                             const std::vector<std::string>& labels,
                             int top_k) {
  if (output.shape.size() != 2) {
    throw std::runtime_error("classification output must be [batch, classes]");
  }

  const std::size_t batch_size = static_cast<std::size_t>(output.shape[0]);
  const std::size_t class_count = static_cast<std::size_t>(output.shape[1]);
  const std::vector<float> scores = CopyOutputToHost(output);
  if (scores.size() != batch_size * class_count) {
    throw std::runtime_error("classification output byte size mismatch");
  }

  ClassificationBatch batch;
  batch.reserve(batch_size);
  for (std::size_t batch_index = 0; batch_index < batch_size; ++batch_index) {
    const float* sample_scores = scores.data() + batch_index * class_count;
    std::vector<std::size_t> top_indices =
        TopK(sample_scores, class_count, static_cast<std::size_t>(top_k));

    std::vector<Classification> sample;
    sample.reserve(top_indices.size());
    for (std::size_t rank = 0; rank < top_indices.size(); ++rank) {
      const std::size_t class_index = top_indices[rank];
      sample.push_back(Classification{class_index, sample_scores[class_index],
                                      LabelForIndex(labels, class_index)});
    }
    batch.push_back(std::move(sample));
  }
  return batch;
}

class TopKClassifier {
 public:
  using Input = mw::infer::InferOutputs;
  using Output = ClassificationBatch;

  TopKClassifier(std::vector<std::string> labels, int top_k)
      : labels_(std::move(labels)), top_k_(top_k) {}

  Output Run(const Input& outputs, mw::infer::RunContext&) const {
    if (outputs.outputs.empty()) {
      throw std::runtime_error("model produced no outputs");
    }
    return MakeTopK(outputs.outputs.front(), labels_, top_k_);
  }

 private:
  std::vector<std::string> labels_;
  int top_k_ = 5;
};

void PrintClassificationBatch(const ClassificationBatch& batch) {
  for (std::size_t sample_index = 0; sample_index < batch.size();
       ++sample_index) {
    std::cout << "sample=" << sample_index << '\n';
    for (std::size_t rank = 0; rank < batch[sample_index].size(); ++rank) {
      const Classification& classification = batch[sample_index][rank];
      std::cout << "  rank=" << rank + 1
                << " class=" << classification.class_index
                << " score=" << classification.score;
      if (!classification.label.empty()) {
        std::cout << " label=\"" << classification.label << '"';
      }
      std::cout << '\n';
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  CLI::App app{"Run real ONNX image classification batch inference."};

  std::string model_path;
  std::vector<std::string> image_paths;
  std::string backend = "onnx_cpu";
  std::string labels_path;
  int top_k = 5;

  app.add_option("model", model_path, "Path to an ONNX classification model")
      ->required();
  app.add_option("images", image_paths, "Image paths")
      ->required()
      ->expected(1, -1);
  app.add_option("--backend", backend, "Backend: onnx_cpu or onnx_gpu")
      ->check(CLI::IsMember({"onnx_cpu", "onnx_gpu", "onnx_cuda"}));
  app.add_option("--labels", labels_path, "Optional one-label-per-line file");
  app.add_option("--topk", top_k, "Number of classes to print")
      ->check(CLI::PositiveNumber);

  CLI11_PARSE(app, argc, argv);

  try {
    const mw::infer::BackendKind backend_kind = ParseBackend(backend);

    std::vector<cv::Mat> images = LoadImages(image_paths);
    std::vector<std::string> labels = LoadLabels(labels_path);

    auto pipeline =
        mw::infer::MakePipeline<std::vector<cv::Mat>>()
            .Then("resize", mw::infer::ResizeByShortEdge(256))
            .Then("center_crop",
                  mw::infer::CenterCrop(mw::infer::ImageSize{224, 224}))
            .Then("normalize",
                  mw::infer::Normalize({0.5F, 0.5F, 0.5F}, {0.5F, 0.5F, 0.5F},
                                       1.0F / 255.0F))
            .Then("infer",
                  mw::infer::OnnxInfer(mw::infer::ModelFromPath(model_path),
                                       backend_kind))
            .Then("topk", TopKClassifier(std::move(labels), top_k));

    auto run = pipeline.RunWithContext(images);
    const auto* infer_outputs =
        run.context.OutputOf<mw::infer::InferOutputs>("infer");
    if (infer_outputs == nullptr) {
      throw std::runtime_error("pipeline did not record infer outputs");
    }

    std::cout << "backend=" << mw::infer::BackendName(backend_kind)
              << " batch_size=" << infer_outputs->batch_size << '\n';
    PrintClassificationBatch(run.output);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }

  return 0;
}
