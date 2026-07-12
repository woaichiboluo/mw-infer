#ifndef MW_INFER_RUNTIME_PROCESS_IMAGE_TO_TENSOR_H_
#define MW_INFER_RUNTIME_PROCESS_IMAGE_TO_TENSOR_H_

#include <memory>
#include <vector>

#include "mw/infer/runtime/execution_stream.h"
#include "mw/infer/runtime/input/input.h"
#include "mw/infer/runtime/tensor/tensor.h"
#include "mw/infer/runtime/tensor/tensor_allocator.h"

namespace mw::infer {

enum class TensorLayout {
  kBchw,
  kBhwc,
};

class ImageToTensorAdapter {
 public:
  virtual ~ImageToTensorAdapter() = default;

  virtual bool Supports(const RawImageBatch& images, Device target_device,
                        const TensorInfo& input, TensorLayout layout) const = 0;
  virtual void Convert(const RawImageBatch& images, Tensor* output,
                       TensorLayout layout) const = 0;
  virtual void Convert(const RawImageBatch& images, Tensor* output,
                       ExecutionStream& stream, TensorLayout layout) const {
    static_cast<void>(stream);
    Convert(images, output, layout);
  }
};

class ImageToTensorConverter {
 public:
  ImageToTensorConverter();
  explicit ImageToTensorConverter(
      std::vector<std::unique_ptr<ImageToTensorAdapter>> adapters);
  ImageToTensorConverter(const ImageToTensorConverter&) = delete;
  ImageToTensorConverter& operator=(const ImageToTensorConverter&) = delete;
  ImageToTensorConverter(ImageToTensorConverter&&) noexcept = default;
  ImageToTensorConverter& operator=(ImageToTensorConverter&&) noexcept =
      default;

  bool Supports(const RawImageBatch& images, Device target_device,
                const TensorInfo& input,
                TensorLayout layout = TensorLayout::kBchw) const;
  Tensor Convert(const RawImageBatch& images, Device target_device,
                 const TensorInfo& input,
                 TensorLayout layout = TensorLayout::kBchw,
                 TensorAllocator& allocator = TensorAllocator::Default()) const;
  // CUDA work is enqueued on stream. Keep output alive until the stream is
  // synchronized. CPU adapters remain synchronous by default.
  Tensor Convert(const RawImageBatch& images, Device target_device,
                 const TensorInfo& input, ExecutionStream& stream,
                 TensorLayout layout = TensorLayout::kBchw,
                 TensorAllocator& allocator = TensorAllocator::Default()) const;

 private:
  void AddAdapter(std::unique_ptr<ImageToTensorAdapter> adapter);
  const ImageToTensorAdapter& SelectAdapter(const RawImageBatch& images,
                                            Device target_device,
                                            const TensorInfo& input,
                                            TensorLayout layout) const;

  std::vector<std::unique_ptr<ImageToTensorAdapter>> adapters_;
};

Tensor ToTensor(const RawImageBatch& images, Device target_device,
                const TensorInfo& input,
                TensorLayout layout = TensorLayout::kBchw,
                TensorAllocator& allocator = TensorAllocator::Default());
Tensor ToTensor(const RawImageBatch& images, Device target_device,
                const TensorInfo& input, ExecutionStream& stream,
                TensorLayout layout = TensorLayout::kBchw,
                TensorAllocator& allocator = TensorAllocator::Default());

}  // namespace mw::infer

#endif  // MW_INFER_RUNTIME_PROCESS_IMAGE_TO_TENSOR_H_
