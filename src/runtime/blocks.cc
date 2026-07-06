#include "mw/infer/runtime/blocks.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <string>
#include <utility>

namespace mw::infer {
namespace {

ImageSize GetSize(const cv::Mat& image) {
  return ImageSize{image.cols, image.rows};
}

void ValidateSize(ImageSize size, const char* block_name) {
  if (size.width <= 0 || size.height <= 0) {
    throw std::invalid_argument(std::string(block_name) +
                                " target size is invalid");
  }
}

void ValidateBatch(const std::vector<cv::Mat>& images, const char* block_name) {
  if (images.empty()) {
    throw std::invalid_argument(std::string(block_name) +
                                " input batch is empty");
  }
  for (const cv::Mat& image : images) {
    if (image.empty()) {
      throw std::invalid_argument(std::string(block_name) + " input is empty");
    }
  }
}

std::vector<ImageSize> GetBatchSizes(const std::vector<cv::Mat>& images) {
  std::vector<ImageSize> sizes;
  sizes.reserve(images.size());
  for (const cv::Mat& image : images) {
    sizes.push_back(GetSize(image));
  }
  return sizes;
}

ImageSize ShortEdgeTarget(ImageSize src_size, int short_edge) {
  if (short_edge <= 0) {
    throw std::invalid_argument("ResizeByShortEdge target size is invalid");
  }
  const int current_short_edge = std::min(src_size.width, src_size.height);
  const float scale =
      static_cast<float>(short_edge) / static_cast<float>(current_short_edge);
  return ImageSize{static_cast<int>(src_size.width * scale + 0.5F),
                   static_cast<int>(src_size.height * scale + 0.5F)};
}

cv::Mat ResizeMat(const cv::Mat& image, ImageSize dst_size, int interpolation) {
  cv::Mat output;
  cv::resize(image, output, cv::Size(dst_size.width, dst_size.height), 0.0, 0.0,
             interpolation);
  return output;
}

RectF CenterCropRect(const cv::Mat& image, ImageSize dst_size) {
  ValidateSize(dst_size, "CenterCrop");
  if (image.cols < dst_size.width || image.rows < dst_size.height) {
    throw std::invalid_argument("CenterCrop input is smaller than crop size");
  }

  const int top = (image.rows - dst_size.height) / 2;
  const int left = (image.cols - dst_size.width) / 2;
  return RectF{static_cast<float>(left), static_cast<float>(top),
               static_cast<float>(dst_size.width),
               static_cast<float>(dst_size.height)};
}

cv::Mat CenterCropMat(const cv::Mat& image, const RectF& crop) {
  return image(cv::Rect(static_cast<int>(crop.x), static_cast<int>(crop.y),
                        static_cast<int>(crop.width),
                        static_cast<int>(crop.height)))
      .clone();
}

struct LetterBoxResult {
  cv::Mat image;
  ImageSize resized_size;
  ImageSize size;
  PointF offset;
};

struct LetterBoxGeometry {
  ImageSize resized_size;
  ImageSize size;
  PointF offset;
};

LetterBoxGeometry MakeLetterBoxGeometry(ImageSize src_size,
                                        ImageSize target_size) {
  ValidateSize(target_size, "LetterBox");
  const float ratio =
      std::min(target_size.width * 1.0F / static_cast<float>(src_size.width),
               target_size.height * 1.0F / static_cast<float>(src_size.height));
  const ImageSize resized_size{
      static_cast<int>(std::round(src_size.width * ratio)),
      static_cast<int>(std::round(src_size.height * ratio))};
  const int padding_w = target_size.width - resized_size.width;
  const int padding_h = target_size.height - resized_size.height;
  return LetterBoxGeometry{
      resized_size,
      target_size,
      PointF{static_cast<float>(padding_w / 2),
             static_cast<float>(padding_h / 2)},
  };
}

LetterBoxResult LetterBoxMat(const cv::Mat& image, ImageSize target_size,
                             float pad_value, int interpolation) {
  const ImageSize src_size = GetSize(image);
  const LetterBoxGeometry geometry =
      MakeLetterBoxGeometry(src_size, target_size);

  cv::Mat resized = image;
  if (geometry.resized_size.width != src_size.width ||
      geometry.resized_size.height != src_size.height) {
    resized = ResizeMat(image, geometry.resized_size, interpolation);
  }

  const int top_padding = static_cast<int>(geometry.offset.y);
  const int left_padding = static_cast<int>(geometry.offset.x);
  const int bottom_padding =
      geometry.size.height - geometry.resized_size.height - top_padding;
  const int right_padding =
      geometry.size.width - geometry.resized_size.width - left_padding;

  cv::Mat output;
  cv::copyMakeBorder(resized, output, top_padding, bottom_padding, left_padding,
                     right_padding, cv::BORDER_CONSTANT,
                     cv::Scalar(pad_value, pad_value, pad_value, pad_value));

  return LetterBoxResult{
      std::move(output),
      geometry.resized_size,
      geometry.size,
      geometry.offset,
  };
}

cv::Mat NormalizeMat(const cv::Mat& image, const std::vector<float>& mean,
                     const std::vector<float>& std, float scale, bool to_rgb) {
  if (mean.empty() || std.empty()) {
    throw std::invalid_argument("Normalize requires mean and std");
  }
  if (mean.size() != std.size()) {
    throw std::invalid_argument("Normalize mean/std size mismatch");
  }
  if (mean.size() != static_cast<std::size_t>(image.channels())) {
    throw std::invalid_argument("Normalize channel size mismatch");
  }

  cv::Mat output;
  image.convertTo(output, CV_32FC(image.channels()), scale);
  if (to_rgb && output.channels() == 3) {
    cv::cvtColor(output, output, cv::COLOR_BGR2RGB);
  }

  std::vector<cv::Mat> channels;
  cv::split(output, channels);
  for (std::size_t index = 0; index < channels.size(); ++index) {
    channels[index] = (channels[index] - mean[index]) / std[index];
  }
  cv::merge(channels, output);
  return output;
}

}  // namespace

Resize::Resize(ImageSize size, int interpolation)
    : size_(size), interpolation_(interpolation) {
  ValidateSize(size_, "Resize");
}

Resize::Output Resize::Run(const Input& input, RunContext&) const {
  ValidateBatch(input, "Resize");

  Output outputs;
  outputs.reserve(input.size());

  for (const cv::Mat& image : input) {
    const ImageSize src_size = GetSize(image);
    cv::Mat output = image;
    if (size_.width != src_size.width || size_.height != src_size.height) {
      output = ResizeMat(image, size_, interpolation_);
    }
    outputs.push_back(std::move(output));
  }

  return outputs;
}

GeometryUpdate Resize::GetGeometryUpdate(const Input& input,
                                         const Output& output) const {
  return GeometryUpdate::FromSource(GetBatchSizes(input))
      .ThenResize(GetBatchSizes(output));
}

ResizeByShortEdge::ResizeByShortEdge(int short_edge, int interpolation)
    : short_edge_(short_edge), interpolation_(interpolation) {
  if (short_edge_ <= 0) {
    throw std::invalid_argument("ResizeByShortEdge target size is invalid");
  }
}

ResizeByShortEdge::Output ResizeByShortEdge::Run(const Input& input,
                                                 RunContext&) const {
  ValidateBatch(input, "ResizeByShortEdge");

  Output outputs;
  outputs.reserve(input.size());

  for (const cv::Mat& image : input) {
    const ImageSize src_size = GetSize(image);
    const ImageSize dst_size = ShortEdgeTarget(src_size, short_edge_);
    cv::Mat output = image;
    if (dst_size.width != src_size.width ||
        dst_size.height != src_size.height) {
      output = ResizeMat(image, dst_size, interpolation_);
    }
    outputs.push_back(std::move(output));
  }

  return outputs;
}

GeometryUpdate ResizeByShortEdge::GetGeometryUpdate(
    const Input& input, const Output& output) const {
  return GeometryUpdate::FromSource(GetBatchSizes(input))
      .ThenResize(GetBatchSizes(output));
}

CenterCrop::CenterCrop(ImageSize size) : size_(size) {
  ValidateSize(size_, "CenterCrop");
}

CenterCrop::Output CenterCrop::Run(const Input& input, RunContext&) const {
  ValidateBatch(input, "CenterCrop");

  Output outputs;
  outputs.reserve(input.size());

  for (const cv::Mat& image : input) {
    RectF crop = CenterCropRect(image, size_);
    outputs.push_back(CenterCropMat(image, crop));
  }

  return outputs;
}

GeometryUpdate CenterCrop::GetGeometryUpdate(const Input& input,
                                             const Output&) const {
  std::vector<RectF> crops;
  crops.reserve(input.size());
  for (const cv::Mat& image : input) {
    crops.push_back(CenterCropRect(image, size_));
  }
  return GeometryUpdate::FromSource(GetBatchSizes(input))
      .ThenCrop(std::move(crops));
}

LetterBox::LetterBox(ImageSize size, float pad_value, int interpolation)
    : size_(size), pad_value_(pad_value), interpolation_(interpolation) {
  ValidateSize(size_, "LetterBox");
}

LetterBox::Output LetterBox::Run(const Input& input, RunContext&) const {
  ValidateBatch(input, "LetterBox");

  Output outputs;
  outputs.reserve(input.size());

  for (const cv::Mat& image : input) {
    LetterBoxResult result =
        LetterBoxMat(image, size_, pad_value_, interpolation_);
    outputs.push_back(std::move(result.image));
  }

  return outputs;
}

GeometryUpdate LetterBox::GetGeometryUpdate(const Input& input,
                                            const Output&) const {
  std::vector<ImageSize> resized_sizes;
  std::vector<ImageSize> dst_sizes;
  std::vector<PointF> offsets;
  resized_sizes.reserve(input.size());
  dst_sizes.reserve(input.size());
  offsets.reserve(input.size());
  for (const cv::Mat& image : input) {
    const LetterBoxGeometry geometry =
        MakeLetterBoxGeometry(GetSize(image), size_);
    resized_sizes.push_back(geometry.resized_size);
    dst_sizes.push_back(geometry.size);
    offsets.push_back(geometry.offset);
  }
  return GeometryUpdate::FromSource(GetBatchSizes(input))
      .ThenResize(std::move(resized_sizes))
      .ThenPad(std::move(dst_sizes), std::move(offsets));
}

Normalize::Normalize(std::vector<float> mean, std::vector<float> std,
                     float scale, bool to_rgb)
    : mean_(std::move(mean)),
      std_(std::move(std)),
      scale_(scale),
      to_rgb_(to_rgb) {}

Normalize::Output Normalize::Run(const Input& input, RunContext&) const {
  ValidateBatch(input, "Normalize");

  Output outputs;
  outputs.reserve(input.size());
  for (const cv::Mat& image : input) {
    outputs.push_back(NormalizeMat(image, mean_, std_, scale_, to_rgb_));
  }
  return outputs;
}

}  // namespace mw::infer
