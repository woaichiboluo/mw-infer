#include "mw/infer/common/memory.h"

#include <stdexcept>
#include <utility>

namespace mw::infer {

namespace {

void* OffsetPtr(void* ptr, std::size_t offset) {
  return static_cast<void*>(static_cast<std::byte*>(ptr) + offset);
}

}  // namespace

MemoryView::MemoryView(const void* data, std::size_t size_bytes)
    : data_(data), size_bytes_(size_bytes) {}

const void* MemoryView::data() const { return data_; }

std::size_t MemoryView::size_bytes() const { return size_bytes_; }

bool MemoryView::empty() const { return data_ == nullptr || size_bytes_ == 0; }

Buffer::Buffer(std::vector<std::byte> bytes, Device device)
    : device_(device), size_bytes_(bytes.size()) {
  auto storage = std::make_shared<std::vector<std::byte>>(std::move(bytes));
  data_ = std::shared_ptr<void>(storage, storage->data());
}

Buffer::Buffer(Device device, std::size_t size_bytes)
    : Buffer(std::vector<std::byte>(size_bytes), device) {}

Buffer::Buffer(Device device, std::shared_ptr<void> data,
               std::size_t size_bytes)
    : device_(device), data_(std::move(data)), size_bytes_(size_bytes) {}

Buffer::Buffer(Device device, void* data, std::size_t size_bytes)
    : Buffer(device, std::shared_ptr<void>(data, [](void*) {}), size_bytes) {}

Buffer::Buffer(Device device, std::shared_ptr<void> data,
               std::size_t size_bytes, bool)
    : device_(device), data_(std::move(data)), size_bytes_(size_bytes) {}

void* Buffer::data() { return data_.get(); }

const void* Buffer::data() const { return data_.get(); }

std::size_t Buffer::size_bytes() const { return size_bytes_; }

bool Buffer::empty() const { return data() == nullptr || size_bytes() == 0; }

const Device& Buffer::device() const { return device_; }

MemoryView Buffer::view() const { return MemoryView(data(), size_bytes()); }

Buffer Buffer::Slice(std::size_t offset, std::size_t size_bytes) const {
  if (offset > size_bytes_ || size_bytes > size_bytes_ - offset) {
    throw std::out_of_range("buffer slice is out of range");
  }
  if (data_ == nullptr && size_bytes != 0) {
    throw std::invalid_argument("buffer slice source is empty");
  }

  std::shared_ptr<void> sliced;
  if (data_ != nullptr) {
    sliced = std::shared_ptr<void>(data_, OffsetPtr(data_.get(), offset));
  }
  return Buffer(device_, std::move(sliced), size_bytes, true);
}

}  // namespace mw::infer
