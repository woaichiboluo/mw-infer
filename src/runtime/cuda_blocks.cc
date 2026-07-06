#include <algorithm>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <stdexcept>
#include <string>
#include <utility>

#include "mw/infer/runtime/blocks.h"

namespace mw::infer {
namespace {

ImageSize GetSize(const cv::cuda::GpuMat& image) {
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

void ValidateBatch(const std::vector<cv::cuda::GpuMat>& images,
                   const char* block_name) {
  if (images.empty()) {
    throw std::invalid_argument(std::string(block_name) +
                                " input batch is empty");
  }
  for (const cv::cuda::GpuMat& image : images) {
    if (image.empty()) {
      throw std::invalid_argument(std::string(block_name) + " input is empty");
    }
  }
}

std::vector<ImageSize> GetBatchSizes(const std::vector<cv::Mat>& images) {
  std::vector<ImageSize> sizes;
  sizes.reserve(images.size());
  for (const cv::Mat& image : images) {
    sizes.push_back(ImageSize{image.cols, image.rows});
  }
  return sizes;
}

std::vector<ImageSize> GetBatchSizes(
    const std::vector<cv::cuda::GpuMat>& images) {
  std::vector<ImageSize> sizes;
  sizes.reserve(images.size());
  for (const cv::cuda::GpuMat& image : images) {
    sizes.push_back(GetSize(image));
  }
  return sizes;
}

cv::cuda::GpuMat ResizeGpuMat(const cv::cuda::GpuMat& image, ImageSize dst_size,
                              int interpolation) {
  cv::cuda::GpuMat output;
  cv::cuda::resize(image, output, cv::Size(dst_size.width, dst_size.height),
                   0.0, 0.0, interpolation);
  return output;
}

struct LetterBoxResult {
  cv::cuda::GpuMat image;
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
  ValidateSize(target_size, "CudaLetterBox");
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

LetterBoxResult LetterBoxGpuMat(const cv::cuda::GpuMat& image,
                                ImageSize target_size, float pad_value,
                                int interpolation) {
  const ImageSize src_size = GetSize(image);
  const LetterBoxGeometry geometry =
      MakeLetterBoxGeometry(src_size, target_size);

  cv::cuda::GpuMat resized = image;
  if (geometry.resized_size.width != src_size.width ||
      geometry.resized_size.height != src_size.height) {
    resized = ResizeGpuMat(image, geometry.resized_size, interpolation);
  }

  const int top_padding = static_cast<int>(geometry.offset.y);
  const int left_padding = static_cast<int>(geometry.offset.x);
  const int bottom_padding =
      geometry.size.height - geometry.resized_size.height - top_padding;
  const int right_padding =
      geometry.size.width - geometry.resized_size.width - left_padding;

  cv::cuda::GpuMat output;
  cv::cuda::copyMakeBorder(
      resized, output, top_padding, bottom_padding, left_padding, right_padding,
      cv::BORDER_CONSTANT,
      cv::Scalar(pad_value, pad_value, pad_value, pad_value));

  return LetterBoxResult{
      std::move(output),
      geometry.resized_size,
      geometry.size,
      geometry.offset,
  };
}

}  // namespace

UploadCuda::Output UploadCuda::Run(const Input& input, RunContext&) const {
  ValidateBatch(input, "UploadCuda");

  Output outputs;
  outputs.reserve(input.size());
  for (const cv::Mat& image : input) {
    cv::cuda::GpuMat output;
    output.upload(image);
    outputs.push_back(std::move(output));
  }
  return outputs;
}

DownloadCuda::Output DownloadCuda::Run(const Input& input, RunContext&) const {
  ValidateBatch(input, "DownloadCuda");

  Output outputs;
  outputs.reserve(input.size());
  for (const cv::cuda::GpuMat& image : input) {
    cv::Mat output;
    image.download(output);
    outputs.push_back(std::move(output));
  }
  return outputs;
}

CudaResize::CudaResize(ImageSize size, int interpolation)
    : size_(size), interpolation_(interpolation) {
  ValidateSize(size_, "CudaResize");
}

CudaResize::Output CudaResize::Run(const Input& input, RunContext&) const {
  ValidateBatch(input, "CudaResize");

  Output outputs;
  outputs.reserve(input.size());

  for (const cv::cuda::GpuMat& image : input) {
    const ImageSize src_size = GetSize(image);
    cv::cuda::GpuMat output = image;
    if (size_.width != src_size.width || size_.height != src_size.height) {
      output = ResizeGpuMat(image, size_, interpolation_);
    }
    outputs.push_back(std::move(output));
  }

  return outputs;
}

GeometryUpdate CudaResize::GetGeometryUpdate(const Input& input,
                                             const Output& output) const {
  return GeometryUpdate::FromSource(GetBatchSizes(input))
      .ThenResize(GetBatchSizes(output));
}

CudaLetterBox::CudaLetterBox(ImageSize size, float pad_value, int interpolation)
    : size_(size), pad_value_(pad_value), interpolation_(interpolation) {
  ValidateSize(size_, "CudaLetterBox");
}

CudaLetterBox::Output CudaLetterBox::Run(const Input& input,
                                         RunContext&) const {
  ValidateBatch(input, "CudaLetterBox");

  Output outputs;
  outputs.reserve(input.size());

  for (const cv::cuda::GpuMat& image : input) {
    LetterBoxResult result =
        LetterBoxGpuMat(image, size_, pad_value_, interpolation_);
    outputs.push_back(std::move(result.image));
  }

  return outputs;
}

GeometryUpdate CudaLetterBox::GetGeometryUpdate(const Input& input,
                                                const Output&) const {
  std::vector<ImageSize> resized_sizes;
  std::vector<ImageSize> dst_sizes;
  std::vector<PointF> offsets;
  resized_sizes.reserve(input.size());
  dst_sizes.reserve(input.size());
  offsets.reserve(input.size());
  for (const cv::cuda::GpuMat& image : input) {
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

}  // namespace mw::infer
