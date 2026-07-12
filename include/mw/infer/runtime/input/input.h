#ifndef MW_INFER_RUNTIME_INPUT_H_
#define MW_INFER_RUNTIME_INPUT_H_

#include <cstddef>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "mw/infer/runtime/tensor/tensor.h"

namespace mw::infer {

struct ImageSize {
  int width = 0;
  int height = 0;
};

enum class ImageMemoryKind {
  kHost,
  kCuda,
};

enum class ImageHandleKind {
  kNone,
  kOpenCvMat,
  kOpenCvCudaGpuMat,
};

enum class PixelFormat {
  kUnknown,
  kRgb,
  kBgr,
  kRgba,
  kBgra,
  kGray,
  kNv12,
  kNv21,
};

struct ImagePlaneDesc {
  std::size_t row_stride_bytes = 0;
  std::size_t pixel_stride_bytes = 0;
};

struct ImageDesc {
  ImageSize size;
  PixelFormat pixel_format = PixelFormat::kUnknown;
  DataType data_type = DataType::kUnknown;
  int channels = 0;
  ImageMemoryKind memory_kind = ImageMemoryKind::kHost;
  std::vector<ImagePlaneDesc> planes;
  int device_id = 0;
};

class RawImage {
 public:
  RawImage() = default;

  template <typename Image, typename = std::enable_if_t<
                                !std::is_same_v<std::decay_t<Image>, RawImage>>>
  explicit RawImage(Image&& image);

  template <typename Handle>
  static RawImage FromHandle(ImageDesc desc, ImageHandleKind handle_kind,
                             Handle handle) {
    using HandleType = std::decay_t<Handle>;
    return RawImage(std::move(desc),
                    std::make_shared<HandleType>(std::move(handle)),
                    handle_kind);
  }

  template <typename Handle>
  static RawImage FromSharedHandle(ImageDesc desc,
                                   ImageHandleKind handle_kind,
                                   std::shared_ptr<Handle> handle) {
    if (!handle) {
      throw std::invalid_argument("RawImage shared handle is null");
    }
    std::shared_ptr<const void> erased_handle = std::move(handle);
    return RawImage(std::move(desc), std::move(erased_handle), handle_kind);
  }

  bool empty() const { return !handle_; }
  const ImageDesc& desc() const { return desc_; }
  ImageSize size() const { return desc_.size; }
  PixelFormat pixel_format() const { return desc_.pixel_format; }
  DataType data_type() const { return desc_.data_type; }
  int channels() const { return desc_.channels; }
  ImageMemoryKind memory_kind() const { return desc_.memory_kind; }
  Device device() const {
    if (desc_.memory_kind == ImageMemoryKind::kCuda) {
      return Device{DeviceType::kCuda, desc_.device_id};
    }
    return Device{DeviceType::kCpu, 0};
  }
  ImageHandleKind handle_kind() const { return handle_kind_; }
  const void* handle() const { return handle_.get(); }

 private:
  RawImage(ImageDesc desc, std::shared_ptr<const void> handle,
           ImageHandleKind handle_kind)
      : desc_(std::move(desc)),
        handle_(std::move(handle)),
        handle_kind_(handle_kind) {}

  ImageDesc desc_;
  std::shared_ptr<const void> handle_;
  ImageHandleKind handle_kind_ = ImageHandleKind::kNone;
};

class RawImageBatch {
 public:
  RawImageBatch() = default;
  explicit RawImageBatch(std::vector<RawImage> images)
      : images_(std::move(images)) {
    ValidateImages();
  }

  template <typename Image,
            typename = std::enable_if_t<!std::is_same_v<Image, RawImage>>>
  explicit RawImageBatch(std::vector<Image> images);

  bool empty() const { return images_.empty(); }
  std::size_t size() const { return images_.size(); }
  const std::vector<RawImage>& images() const { return images_; }
  const RawImage& image(std::size_t index) const { return images_.at(index); }

  ImageMemoryKind memory_kind() const {
    if (images_.empty()) {
      throw std::logic_error("RawImageBatch is empty");
    }
    return images_.front().memory_kind();
  }

 private:
  void ValidateImages() const {
    if (images_.empty()) {
      return;
    }
    const ImageMemoryKind memory_kind = images_.front().memory_kind();
    const PixelFormat pixel_format = images_.front().pixel_format();
    const DataType data_type = images_.front().data_type();
    const int channels = images_.front().channels();
    const Device device = images_.front().device();
    for (const RawImage& image : images_) {
      if (image.empty()) {
        throw std::invalid_argument("RawImageBatch contains an empty image");
      }
      if (image.memory_kind() != memory_kind) {
        throw std::invalid_argument(
            "RawImageBatch cannot mix image memory kinds");
      }
      if (image.pixel_format() != pixel_format) {
        throw std::invalid_argument("RawImageBatch cannot mix pixel formats");
      }
      if (image.data_type() != data_type) {
        throw std::invalid_argument(
            "RawImageBatch cannot mix image data types");
      }
      if (image.channels() != channels) {
        throw std::invalid_argument(
            "RawImageBatch cannot mix image channel counts");
      }
      if (image.device().type != device.type ||
          image.device().id != device.id) {
        throw std::invalid_argument("RawImageBatch cannot mix image devices");
      }
    }
  }

  std::vector<RawImage> images_;
};

namespace detail {

template <typename>
inline constexpr bool kDependentFalse = false;

}  // namespace detail

template <typename Image>
struct RawImageConverter {
  template <typename Input>
  static RawImage Convert(Input&&) {
    static_assert(
        detail::kDependentFalse<Input>,
        "RawImageConverter is not specialized for this image type. Include "
        "the matching runtime input adapter header.");
    return RawImage();
  }
};

template <>
struct RawImageConverter<RawImage> {
  static RawImage Convert(RawImage image) { return image; }
};

template <typename Image>
RawImage ToRawImage(Image&& image) {
  using ImageType = std::decay_t<Image>;
  return RawImageConverter<ImageType>::Convert(std::forward<Image>(image));
}

template <typename Image>
RawImageBatch ToRawImageBatch(std::vector<Image> images) {
  std::vector<RawImage> raw_images;
  raw_images.reserve(images.size());
  for (Image& image : images) {
    raw_images.push_back(ToRawImage(std::move(image)));
  }
  return RawImageBatch(std::move(raw_images));
}

template <typename Image>
RawImageBatch ToRawImageBatch(std::initializer_list<Image> images) {
  return ToRawImageBatch(std::vector<Image>(images));
}

template <typename Image, typename>
RawImage::RawImage(Image&& image)
    : RawImage(ToRawImage(std::forward<Image>(image))) {}

template <typename Image, typename>
RawImageBatch::RawImageBatch(std::vector<Image> images)
    : RawImageBatch(ToRawImageBatch(std::move(images))) {}

}  // namespace mw::infer

#endif  // MW_INFER_RUNTIME_INPUT_H_
