#include <fmt/core.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/imgcodecs.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mw/infer/runtime/backend/backend.h"
#include "mw/infer/runtime/input/opencv_input.h"
#include "mw/infer/runtime/postprocess/segmentation.h"
#include "mw/infer/runtime/process/geometry.h"
#include "mw/infer/runtime/process/image_to_tensor.h"
#include "mw/infer/runtime/process/normalize.h"

namespace {

constexpr int kResizeShortSide = 520;
constexpr std::size_t kClassCount = 21;

struct SegmentationInput {
  mw::infer::Tensor tensor;
  std::vector<mw::infer::GeometryTrace> traces;
};

cv::Mat LoadImage(const std::filesystem::path& path) {
  cv::Mat image = cv::imread(path.string(), cv::IMREAD_COLOR);
  if (image.empty()) {
    throw std::invalid_argument("Failed to read image: " + path.string());
  }
  return image;
}

SegmentationInput MakeInput(cv::Mat image, mw::infer::Device device,
                            std::string input_name) {
  const mw::infer::ImageSize original_size{image.cols, image.rows};
  const mw::infer::ImageSize input_size =
      mw::infer::ResizeShortSideSize(original_size, kResizeShortSide);
  mw::infer::RawImageBatch raw_images(std::vector<cv::Mat>{std::move(image)});

  mw::infer::GeometryTransformer transformer;
  mw::infer::GeometryResult resized = transformer.Resize(
      std::move(raw_images), input_size, mw::infer::Interpolation::kLinear);

  mw::infer::TensorInfo input_info;
  input_info.name = std::move(input_name);
  input_info.data_type = mw::infer::DataType::kFloat32;
  input_info.shape = {1, 3, input_size.height, input_size.width};

  mw::infer::Tensor tensor = mw::infer::ToTensor(
      resized.images(), device, input_info, mw::infer::TensorLayout::kBchw);
  return SegmentationInput{
      mw::infer::Normalize(tensor, {0.485F, 0.456F, 0.406F},
                           {0.229F, 0.224F, 0.225F}, 1.0F / 255.0F,
                           mw::infer::TensorLayout::kBchw),
      resized.traces()};
}

cv::Mat MakeMask(const mw::infer::Tensor& class_ids,
                 std::array<int64_t, kClassCount>* pixel_counts) {
  if (class_ids.data_type() != mw::infer::DataType::kInt64 ||
      class_ids.shape().size() != 3 || class_ids.shape()[0] != 1) {
    throw std::invalid_argument(
        "FCN segmentation example expects class id shape [1, H, W]");
  }

  const int height = static_cast<int>(class_ids.shape()[1]);
  const int width = static_cast<int>(class_ids.shape()[2]);
  const std::vector<int64_t> ids = class_ids.CopyToHostVector<int64_t>();
  pixel_counts->fill(0);

  cv::Mat mask(height, width, CV_8UC1);
  for (int y = 0; y < height; ++y) {
    auto* row = mask.ptr<std::uint8_t>(y);
    for (int x = 0; x < width; ++x) {
      const int64_t pixel = static_cast<int64_t>(y) * width + x;
      const int64_t class_id = ids[static_cast<std::size_t>(pixel)];
      row[x] = static_cast<std::uint8_t>(class_id);
      if (class_id >= 0 &&
          static_cast<std::size_t>(class_id) < pixel_counts->size()) {
        ++(*pixel_counts)[static_cast<std::size_t>(class_id)];
      }
    }
  }
  return mask;
}

void PrintClassCounts(const std::array<int64_t, kClassCount>& pixel_counts) {
  for (std::size_t class_id = 0; class_id < kClassCount; ++class_id) {
    if (pixel_counts[class_id] > 0) {
      fmt::print("  class={:<2} pixels={}\n", class_id, pixel_counts[class_id]);
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);
  if (argc < 3 || argc > 5) {
    fmt::print(stderr,
               "usage: {} <fcn-resnet50.onnx> <image> [cpu|cuda:<id>] "
               "[mask.png]\n",
               argv[0]);
    return 2;
  }

  try {
    const std::filesystem::path model_path = argv[1];
    const std::filesystem::path image_path = argv[2];
    const mw::infer::Device device(argc >= 4 ? argv[3] : "cpu");
    const std::filesystem::path mask_path =
        argc == 5 ? argv[4] : "segmentation_mask.png";

    mw::infer::Model model = mw::infer::ModelFromPath(model_path);
    mw::infer::BackendPtr backend =
        mw::infer::CreateBackend(std::move(model), device);
    if (backend->model_info().inputs.size() != 1) {
      throw std::invalid_argument(
          "FCN segmentation example expects exactly one model input");
    }

    SegmentationInput input =
        MakeInput(LoadImage(image_path), device,
                  backend->model_info().inputs.front().name);
    std::vector<mw::infer::Tensor> outputs = backend->Infer(input.tensor);
    if (outputs.empty()) {
      throw std::invalid_argument(
          "FCN segmentation example expects at least one model output");
    }

    mw::infer::SemanticSegmentationResult segmentation =
        mw::infer::SemanticSegmentation(outputs.front(), input.traces);
    std::array<int64_t, kClassCount> pixel_counts;
    cv::Mat mask = MakeMask(segmentation.class_ids, &pixel_counts);
    if (!cv::imwrite(mask_path.string(), mask)) {
      throw std::runtime_error("Failed to write mask: " + mask_path.string());
    }

    fmt::print("example: segmentation_fcn_infer\n");
    fmt::print("model: {}\n", backend->model().name);
    fmt::print("device: {}\n", device.ToString());
    fmt::print("mask: {}\n", mask_path.string());
    fmt::print("classes:\n");
    PrintClassCounts(pixel_counts);
  } catch (const std::exception& error) {
    fmt::print(stderr, "segmentation_fcn_infer failed: {}\n", error.what());
    return 1;
  }

  return 0;
}
