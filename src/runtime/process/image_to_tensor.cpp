#include "mw/infer/runtime/process/image_to_tensor.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mw::infer {

#if defined(MW_INFER_HAS_OPENCV_IMAGE_TO_TENSOR_ADAPTER)
std::unique_ptr<ImageToTensorAdapter> CreateOpenCvImageToTensorAdapter();
#endif

#if defined(MW_INFER_HAS_OPENCV_CUDA_IMAGE_TO_TENSOR_ADAPTER)
std::unique_ptr<ImageToTensorAdapter> CreateOpenCvCudaImageToTensorAdapter();
#endif

namespace {

bool IsSupportedDeviceType(DeviceType type) {
  switch (type) {
    case DeviceType::kCpu:
    case DeviceType::kCuda:
      return true;
  }
  return false;
}

bool IsSupportedLayout(TensorLayout layout) {
  switch (layout) {
    case TensorLayout::kBchw:
    case TensorLayout::kBhwc:
      return true;
  }
  return false;
}

void ValidateRawImages(const RawImageBatch& images) {
  if (images.empty()) {
    throw std::invalid_argument("Image-to-tensor input batch is empty");
  }

  const RawImage& image = images.image(0);
  if (image.size().width <= 0 || image.size().height <= 0) {
    throw std::invalid_argument("Image-to-tensor input image size is invalid");
  }
  if (image.channels() <= 0) {
    throw std::invalid_argument(
        "Image-to-tensor input image channel count is invalid");
  }

  for (const RawImage& current : images.images()) {
    if (current.size().width != image.size().width ||
        current.size().height != image.size().height) {
      throw std::invalid_argument(
          "Image-to-tensor input batch cannot mix image sizes");
    }
  }
}

void ValidateTargetDevice(Device target_device) {
  if (!IsSupportedDeviceType(target_device.type)) {
    throw std::invalid_argument("Image-to-tensor target device is unsupported");
  }
  if (target_device.id < 0) {
    throw std::invalid_argument("Image-to-tensor target device id is negative");
  }
}

void ValidateLayout(TensorLayout layout) {
  if (!IsSupportedLayout(layout)) {
    throw std::invalid_argument("Image-to-tensor layout is unsupported");
  }
}

void ValidateInputInfo(const TensorInfo& input) {
  static_cast<void>(DataTypeSize(input.data_type));
  if (input.shape.size() != 4) {
    throw std::invalid_argument(
        "Image-to-tensor model input tensor must be rank 4");
  }
}

void ValidateStaticDim(int64_t expected, int64_t actual, const char* name) {
  if (expected > 0 && expected != actual) {
    throw std::invalid_argument(std::string("Image-to-tensor model input ") +
                                name + " dimension mismatch");
  }
}

std::vector<int64_t> ResolveShape(const RawImageBatch& images,
                                  const TensorInfo& input,
                                  TensorLayout layout) {
  const RawImage& image = images.image(0);
  const int64_t batch = static_cast<int64_t>(images.size());
  const int64_t channels = image.channels();
  const int64_t height = image.size().height;
  const int64_t width = image.size().width;

  switch (layout) {
    case TensorLayout::kBchw:
      ValidateStaticDim(input.shape[0], batch, "batch");
      ValidateStaticDim(input.shape[1], channels, "channel");
      ValidateStaticDim(input.shape[2], height, "height");
      ValidateStaticDim(input.shape[3], width, "width");
      return {batch, channels, height, width};
    case TensorLayout::kBhwc:
      ValidateStaticDim(input.shape[0], batch, "batch");
      ValidateStaticDim(input.shape[1], height, "height");
      ValidateStaticDim(input.shape[2], width, "width");
      ValidateStaticDim(input.shape[3], channels, "channel");
      return {batch, height, width, channels};
  }
  throw std::invalid_argument("Image-to-tensor layout is unsupported");
}

bool IsValidRequest(const RawImageBatch& images, Device target_device,
                    const TensorInfo& input, TensorLayout layout) {
  try {
    ValidateRawImages(images);
    ValidateTargetDevice(target_device);
    ValidateLayout(layout);
    ValidateInputInfo(input);
    static_cast<void>(ResolveShape(images, input, layout));
  } catch (const std::invalid_argument&) {
    return false;
  }
  return true;
}

TensorDesc MakeTensorDesc(const RawImageBatch& images, Device target_device,
                          const TensorInfo& input, TensorLayout layout) {
  TensorDesc desc;
  desc.info = input;
  desc.info.shape = ResolveShape(images, input, layout);
  desc.device = target_device;
  return desc;
}

bool SameDevice(Device lhs, Device rhs) {
  return lhs.type == rhs.type && lhs.id == rhs.id;
}

}  // namespace

ImageToTensorConverter::ImageToTensorConverter() {
#if defined(MW_INFER_HAS_OPENCV_IMAGE_TO_TENSOR_ADAPTER)
  AddAdapter(CreateOpenCvImageToTensorAdapter());
#endif
#if defined(MW_INFER_HAS_OPENCV_CUDA_IMAGE_TO_TENSOR_ADAPTER)
  AddAdapter(CreateOpenCvCudaImageToTensorAdapter());
#endif
}

ImageToTensorConverter::ImageToTensorConverter(
    std::vector<std::unique_ptr<ImageToTensorAdapter>> adapters) {
  if (adapters.empty()) {
    throw std::invalid_argument("Image-to-tensor converter has no adapters");
  }
  for (auto& adapter : adapters) {
    AddAdapter(std::move(adapter));
  }
}

void ImageToTensorConverter::AddAdapter(
    std::unique_ptr<ImageToTensorAdapter> adapter) {
  if (!adapter) {
    throw std::invalid_argument("Image-to-tensor adapter is null");
  }
  adapters_.push_back(std::move(adapter));
}

bool ImageToTensorConverter::Supports(const RawImageBatch& images,
                                      Device target_device,
                                      const TensorInfo& input,
                                      TensorLayout layout) const {
  if (!IsValidRequest(images, target_device, input, layout)) {
    return false;
  }
  for (const auto& adapter : adapters_) {
    if (adapter->Supports(images, target_device, input, layout)) {
      return true;
    }
  }
  return false;
}

Tensor ImageToTensorConverter::Convert(const RawImageBatch& images,
                                       Device target_device,
                                       const TensorInfo& input,
                                       TensorLayout layout,
                                       TensorAllocator& allocator) const {
  ValidateRawImages(images);
  ValidateTargetDevice(target_device);
  ValidateLayout(layout);
  ValidateInputInfo(input);

  const ImageToTensorAdapter& adapter =
      SelectAdapter(images, target_device, input, layout);
  Tensor output = Tensor::Allocate(
      MakeTensorDesc(images, target_device, input, layout), allocator);
  adapter.Convert(images, &output, layout);
  return output;
}

Tensor ImageToTensorConverter::Convert(const RawImageBatch& images,
                                       Device target_device,
                                       const TensorInfo& input,
                                       ExecutionStream& stream,
                                       TensorLayout layout,
                                       TensorAllocator& allocator) const {
  ValidateRawImages(images);
  ValidateTargetDevice(target_device);
  ValidateLayout(layout);
  ValidateInputInfo(input);
  if (!SameDevice(target_device, stream.device())) {
    throw std::invalid_argument(
        "Image-to-tensor stream device does not match target device");
  }

  const ImageToTensorAdapter& adapter =
      SelectAdapter(images, target_device, input, layout);
  const TensorDesc output_desc =
      MakeTensorDesc(images, target_device, input, layout);
  struct StreamOutputOwner {
    Tensor storage;
    RawImageBatch images;
  };
  auto owner = std::make_shared<StreamOutputOwner>();
  owner->storage = Tensor::Allocate(output_desc, allocator);
  owner->images = images;
  std::shared_ptr<void> output_owner = owner;
  Tensor output = Tensor::FromExternal(
      output_desc, owner->storage.data(), owner->storage.bytes(),
      std::move(output_owner));
  try {
    adapter.Convert(owner->images, &output, stream, layout);
  } catch (...) {
    stream.SynchronizeNoThrow();
    throw;
  }
  return output;
}

const ImageToTensorAdapter& ImageToTensorConverter::SelectAdapter(
    const RawImageBatch& images, Device target_device, const TensorInfo& input,
    TensorLayout layout) const {
  for (const auto& adapter : adapters_) {
    if (adapter->Supports(images, target_device, input, layout)) {
      return *adapter;
    }
  }
  throw std::invalid_argument(
      "No image-to-tensor adapter supports this RawImageBatch and model input");
}

Tensor ToTensor(const RawImageBatch& images, Device target_device,
                const TensorInfo& input, TensorLayout layout,
                TensorAllocator& allocator) {
  return ImageToTensorConverter().Convert(images, target_device, input, layout,
                                          allocator);
}

Tensor ToTensor(const RawImageBatch& images, Device target_device,
                const TensorInfo& input, ExecutionStream& stream,
                TensorLayout layout, TensorAllocator& allocator) {
  return ImageToTensorConverter().Convert(images, target_device, input, stream,
                                           layout, allocator);
}

}  // namespace mw::infer
