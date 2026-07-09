#ifndef MW_INFER_RUNTIME_POSTPROCESS_LABEL_MAP_H_
#define MW_INFER_RUNTIME_POSTPROCESS_LABEL_MAP_H_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace mw::infer {

enum class LabelMapFormat {
  kAuto,
  kPlain,
  kIndexed,
};

class LabelMap {
 public:
  LabelMap() = default;
  explicit LabelMap(std::vector<std::string> labels);

  bool empty() const;
  std::size_t size() const;
  const std::vector<std::string>& labels() const;
  const std::string* Find(int64_t class_id) const;
  const std::string& At(int64_t class_id) const;
  std::string LabelOrClass(int64_t class_id) const;

 private:
  friend LabelMap LabelMapFromText(std::string_view text,
                                   LabelMapFormat format);

  void Add(std::string label);
  void Set(int64_t class_id, std::string label);

  std::vector<std::string> labels_;
};

LabelMap LabelMapFromText(std::string_view text,
                          LabelMapFormat format = LabelMapFormat::kAuto);
LabelMap LabelMapFromFile(const std::filesystem::path& path,
                          LabelMapFormat format = LabelMapFormat::kAuto);

}  // namespace mw::infer

#endif  // MW_INFER_RUNTIME_POSTPROCESS_LABEL_MAP_H_
