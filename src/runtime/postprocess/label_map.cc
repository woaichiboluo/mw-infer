#include "mw/infer/runtime/postprocess/label_map.h"

#include <cctype>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <utility>

namespace mw::infer {
namespace {

std::string_view Trim(std::string_view text) {
  while (!text.empty() &&
         std::isspace(static_cast<unsigned char>(text.front())) != 0) {
    text.remove_prefix(1);
  }
  while (!text.empty() &&
         std::isspace(static_cast<unsigned char>(text.back())) != 0) {
    text.remove_suffix(1);
  }
  return text;
}

std::string_view RemoveUtf8Bom(std::string_view text) {
  constexpr std::string_view kBom = "\xEF\xBB\xBF";
  if (text.substr(0, kBom.size()) == kBom) {
    text.remove_prefix(kBom.size());
  }
  return text;
}

std::string_view NextLine(std::string_view* text) {
  const std::size_t newline = text->find('\n');
  if (newline == std::string_view::npos) {
    std::string_view line = *text;
    *text = {};
    if (!line.empty() && line.back() == '\r') {
      line.remove_suffix(1);
    }
    return line;
  }

  std::string_view line = text->substr(0, newline);
  text->remove_prefix(newline + 1);
  if (!line.empty() && line.back() == '\r') {
    line.remove_suffix(1);
  }
  return line;
}

bool ParseClassIdPrefix(std::string_view line, std::size_t* position,
                        int64_t* class_id) {
  if (line.empty() ||
      std::isdigit(static_cast<unsigned char>(line.front())) == 0) {
    return false;
  }

  uint64_t value = 0;
  std::size_t index = 0;
  while (index < line.size() &&
         std::isdigit(static_cast<unsigned char>(line[index])) != 0) {
    const uint64_t digit = static_cast<uint64_t>(line[index] - '0');
    if (value >
        (static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) - digit) /
            10U) {
      throw std::invalid_argument("Label class id is too large");
    }
    value = value * 10U + digit;
    ++index;
  }

  *position = index;
  *class_id = static_cast<int64_t>(value);
  return true;
}

bool TryParseIndexedLine(std::string_view line, int64_t* class_id,
                         std::string* label) {
  std::size_t position = 0;
  if (!ParseClassIdPrefix(line, &position, class_id)) {
    return false;
  }
  if (position >= line.size()) {
    return false;
  }

  std::string_view rest;
  const char separator = line[position];
  if (separator == ',' || separator == ':') {
    rest = line.substr(position + 1);
  } else if (std::isspace(static_cast<unsigned char>(separator)) != 0) {
    rest = line.substr(position + 1);
  } else {
    return false;
  }

  rest = Trim(rest);
  if (rest.empty()) {
    return false;
  }
  *label = std::string(rest);
  return true;
}

std::string LineError(std::size_t line_number, std::string_view message) {
  return "Invalid label map line " + std::to_string(line_number) + ": " +
         std::string(message);
}

void ParseIndexedLine(std::string_view line, std::size_t line_number,
                      int64_t* class_id, std::string* label) {
  if (!TryParseIndexedLine(line, class_id, label)) {
    throw std::invalid_argument(
        LineError(line_number,
                  "expected '<id> <label>', '<id>: <label>' or "
                  "'<id>,<label>'"));
  }
}

}  // namespace

LabelMap::LabelMap(std::vector<std::string> labels)
    : labels_(std::move(labels)) {}

bool LabelMap::empty() const { return labels_.empty(); }

std::size_t LabelMap::size() const { return labels_.size(); }

const std::vector<std::string>& LabelMap::labels() const { return labels_; }

const std::string* LabelMap::Find(int64_t class_id) const {
  if (class_id < 0 || static_cast<std::size_t>(class_id) >= labels_.size()) {
    return nullptr;
  }

  const std::string& label = labels_[static_cast<std::size_t>(class_id)];
  if (label.empty()) {
    return nullptr;
  }
  return &label;
}

const std::string& LabelMap::At(int64_t class_id) const {
  const std::string* label = Find(class_id);
  if (label == nullptr) {
    throw std::out_of_range("Label class id not found: " +
                            std::to_string(class_id));
  }
  return *label;
}

std::string LabelMap::LabelOrClass(int64_t class_id) const {
  const std::string* label = Find(class_id);
  if (label != nullptr) {
    return *label;
  }
  return "class " + std::to_string(class_id);
}

void LabelMap::Add(std::string label) { labels_.push_back(std::move(label)); }

void LabelMap::Set(int64_t class_id, std::string label) {
  if (class_id < 0) {
    throw std::invalid_argument("Label class id must be non-negative");
  }
  const std::size_t index = static_cast<std::size_t>(class_id);
  if (labels_.size() <= index) {
    labels_.resize(index + 1);
  }
  labels_[index] = std::move(label);
}

LabelMap LabelMapFromText(std::string_view text, LabelMapFormat format) {
  text = RemoveUtf8Bom(text);
  LabelMap labels;
  std::size_t line_number = 0;
  while (!text.empty()) {
    ++line_number;
    std::string_view line = Trim(NextLine(&text));
    if (line.empty() || line.front() == '#') {
      continue;
    }

    switch (format) {
      case LabelMapFormat::kPlain:
        labels.Add(std::string(line));
        break;
      case LabelMapFormat::kIndexed: {
        int64_t class_id = 0;
        std::string label;
        ParseIndexedLine(line, line_number, &class_id, &label);
        labels.Set(class_id, std::move(label));
        break;
      }
      case LabelMapFormat::kAuto: {
        int64_t class_id = 0;
        std::string label;
        if (TryParseIndexedLine(line, &class_id, &label)) {
          labels.Set(class_id, std::move(label));
        } else {
          labels.Add(std::string(line));
        }
        break;
      }
    }
  }
  return labels;
}

LabelMap LabelMapFromFile(const std::filesystem::path& path,
                          LabelMapFormat format) {
  std::ifstream file(path);
  if (!file) {
    throw std::invalid_argument("Failed to read labels: " + path.string());
  }

  std::string text((std::istreambuf_iterator<char>(file)),
                   std::istreambuf_iterator<char>());
  return LabelMapFromText(text, format);
}

}  // namespace mw::infer
