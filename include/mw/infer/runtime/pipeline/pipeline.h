#ifndef MW_INFER_RUNTIME_PIPELINE_PIPELINE_H_
#define MW_INFER_RUNTIME_PIPELINE_PIPELINE_H_

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mw/infer/runtime/backend/backend.h"
#include "mw/infer/runtime/execution_stream.h"
#include "mw/infer/runtime/input/input.h"
#include "mw/infer/runtime/tensor/tensor_allocator.h"

namespace mw::infer {

using ModelId = std::size_t;
using ModelOutputs = std::vector<Tensor>;

template <typename ImageResult>
class Pipeline {
 public:
  // result[i] corresponds to the input image at index i.
  using BatchResult = std::vector<ImageResult>;

  // Pipeline instances are single-thread-affine because their backend state
  // and pooled tensor allocator are reused across calls.
  virtual ~Pipeline() { stream_->SynchronizeNoThrow(); }

  Pipeline(const Pipeline&) = delete;
  Pipeline& operator=(const Pipeline&) = delete;
  Pipeline(Pipeline&&) = delete;
  Pipeline& operator=(Pipeline&&) = delete;

  template <typename Input>
  BatchResult Infer(std::vector<Input> images) {
    return Infer(ToRawImageBatch(std::move(images)));
  }

  BatchResult Infer(RawImageBatch images) { return Execute(std::move(images)); }

 protected:
  explicit Pipeline(Device device) : Pipeline(device, BackendFactory()) {}

  Pipeline(Device device, BackendFactory backend_factory)
      : device_(device),
        stream_(std::make_shared<ExecutionStream>(device)),
        backend_factory_(std::move(backend_factory)) {}

  // CUDA implementations must pass stream() to process and postprocess
  // overloads so every stage is ordered on the pipeline stream. CUDA images
  // produced outside the pipeline must be ready before Infer is called.
  virtual BatchResult Process(const RawImageBatch& images) = 0;

  ModelId AddModel(Model model, std::vector<std::string> output_names = {}) {
    models_.push_back(backend_factory_.Create(
        std::move(model), device_, std::move(output_names), stream_));
    return models_.size() - 1;
  }

  const ModelInfo& model_info(ModelId model) const {
    return models_.at(model)->model_info();
  }

  ModelOutputs InferModel(ModelId model, const Tensor& input) {
    return models_.at(model)->Infer(input, allocator_);
  }

  ModelOutputs InferModel(ModelId model, const std::vector<Tensor>& inputs) {
    return models_.at(model)->Infer(inputs, allocator_);
  }

  TensorAllocator& allocator() { return allocator_; }
  ExecutionStream& stream() { return *stream_; }
  Device device() const { return device_; }

 private:
  BatchResult Execute(RawImageBatch images) {
    BatchResult result;
    try {
      if (images.empty()) {
        throw std::invalid_argument("Pipeline input batch is empty");
      }
      ValidateInputDevice(images);
      const std::size_t batch_size = images.size();
      result = Process(images);
      if (result.size() != batch_size) {
        throw std::logic_error(
            "Pipeline result count does not match input batch size");
      }
      stream_->Synchronize();
    } catch (...) {
      stream_->SynchronizeNoThrow();
      throw;
    }
    return result;
  }

  void ValidateInputDevice(const RawImageBatch& images) const {
    if (images.memory_kind() == ImageMemoryKind::kHost) {
      return;
    }
    const Device input_device = images.image(0).device();
    if (device_.type != DeviceType::kCuda || input_device.id != device_.id) {
      throw std::invalid_argument(
          "CUDA input batch device does not match pipeline device");
    }
  }

  Device device_;
  std::shared_ptr<ExecutionStream> stream_;
  BackendFactory backend_factory_;
  std::vector<BackendPtr> models_;
  PooledTensorAllocator allocator_;
};

}  // namespace mw::infer

#endif  // MW_INFER_RUNTIME_PIPELINE_PIPELINE_H_
