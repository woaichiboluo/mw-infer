#ifndef MW_INFER_RUNTIME_MEMORY_H_
#define MW_INFER_RUNTIME_MEMORY_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

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

class MemoryBuffer {
 public:
  MemoryBuffer() = default;
  explicit MemoryBuffer(std::vector<std::byte> bytes);
  MemoryBuffer(std::shared_ptr<void> data, std::size_t size_bytes);

  void* data();
  const void* data() const;
  std::size_t size_bytes() const;
  bool empty() const;
  MemoryView view() const;

 private:
  std::vector<std::byte> owned_bytes_;
  std::shared_ptr<void> external_data_;
  std::size_t external_size_bytes_ = 0;
};

}  // namespace mw::infer

#endif  // MW_INFER_RUNTIME_MEMORY_H_
