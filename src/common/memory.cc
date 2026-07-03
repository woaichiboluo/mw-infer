#include "mw/infer/common/memory.h"

#include <utility>

namespace mw::infer {

MemoryView::MemoryView(const void* data, std::size_t size_bytes)
    : data_(data), size_bytes_(size_bytes) {}

const void* MemoryView::data() const { return data_; }

std::size_t MemoryView::size_bytes() const { return size_bytes_; }

bool MemoryView::empty() const { return data_ == nullptr || size_bytes_ == 0; }

MemoryBuffer::MemoryBuffer(std::vector<std::byte> bytes)
    : owned_bytes_(std::move(bytes)) {}

MemoryBuffer::MemoryBuffer(std::shared_ptr<void> data, std::size_t size_bytes)
    : external_data_(std::move(data)), external_size_bytes_(size_bytes) {}

void* MemoryBuffer::data() {
  if (!owned_bytes_.empty()) {
    return owned_bytes_.data();
  }

  return external_data_.get();
}

const void* MemoryBuffer::data() const {
  if (!owned_bytes_.empty()) {
    return owned_bytes_.data();
  }

  return external_data_.get();
}

std::size_t MemoryBuffer::size_bytes() const {
  if (!owned_bytes_.empty()) {
    return owned_bytes_.size();
  }

  return external_size_bytes_;
}

bool MemoryBuffer::empty() const {
  return data() == nullptr || size_bytes() == 0;
}

MemoryView MemoryBuffer::view() const {
  return MemoryView(data(), size_bytes());
}

}  // namespace mw::infer
