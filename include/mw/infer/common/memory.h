#ifndef MW_INFER_COMMON_MEMORY_H_
#define MW_INFER_COMMON_MEMORY_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "mw/infer/common/device.h"

namespace mw::infer {

class MemoryView {
 public:
  MemoryView() = default;
  MemoryView(const void* data, std::size_t size_bytes);

  const void* data() const;
  std::size_t size_bytes() const;
  bool empty() const;

 private:
  const void* data_ = nullptr;
  std::size_t size_bytes_ = 0;
};

class Buffer {
 public:
  Buffer() = default;
  explicit Buffer(std::vector<std::byte> bytes, Device device = Device::Cpu());
  Buffer(Device device, std::size_t size_bytes);
  Buffer(Device device, std::shared_ptr<void> data, std::size_t size_bytes);
  Buffer(Device device, void* data, std::size_t size_bytes);

  void* data();
  const void* data() const;
  std::size_t size_bytes() const;
  bool empty() const;
  const Device& device() const;
  MemoryView view() const;
  Buffer Slice(std::size_t offset, std::size_t size_bytes) const;

 private:
  Buffer(Device device, std::shared_ptr<void> data, std::size_t size_bytes,
         bool);

  Device device_;
  std::shared_ptr<void> data_;
  std::size_t size_bytes_ = 0;
};

}  // namespace mw::infer

#endif  // MW_INFER_COMMON_MEMORY_H_
