#include "mw/infer/runtime/postprocess/label_map.h"

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mw::infer {
namespace {

std::filesystem::path TemporaryLabelPath() {
  const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
  return std::filesystem::temp_directory_path() /
         ("mw_infer_label_map_test_" + std::to_string(now) + ".txt");
}

TEST(LabelMapTest, ParsesPlainLineLabels) {
  LabelMap labels = LabelMapFromText("cat\n dog \n# comment\n\nbird\n",
                                     LabelMapFormat::kPlain);

  EXPECT_EQ(labels.size(), 3U);
  EXPECT_EQ(labels.At(0), "cat");
  EXPECT_EQ(labels.At(1), "dog");
  EXPECT_EQ(labels.At(2), "bird");
  EXPECT_EQ(labels.LabelOrClass(3), "class 3");
  EXPECT_EQ(labels.LabelOrClass(-1), "class -1");
}

TEST(LabelMapTest, ParsesIndexedCommonFormats) {
  LabelMap labels = LabelMapFromText("0,person\n2: car\n4 bicycle\n",
                                     LabelMapFormat::kIndexed);

  EXPECT_EQ(labels.size(), 5U);
  EXPECT_EQ(labels.At(0), "person");
  EXPECT_EQ(labels.Find(1), nullptr);
  EXPECT_EQ(labels.At(2), "car");
  EXPECT_EQ(labels.At(4), "bicycle");
  EXPECT_EQ(labels.LabelOrClass(1), "class 1");
}

TEST(LabelMapTest, AutoDetectsPlainAndIndexedLines) {
  LabelMap labels = LabelMapFromText("background\n2,cat\n5: dog\n");

  EXPECT_EQ(labels.size(), 6U);
  EXPECT_EQ(labels.At(0), "background");
  EXPECT_EQ(labels.LabelOrClass(1), "class 1");
  EXPECT_EQ(labels.At(2), "cat");
  EXPECT_EQ(labels.At(5), "dog");
}

TEST(LabelMapTest, PlainFormatKeepsLeadingNumberAsLabelText) {
  LabelMap labels =
      LabelMapFromText("0 person\n1 car\n", LabelMapFormat::kPlain);

  EXPECT_EQ(labels.size(), 2U);
  EXPECT_EQ(labels.At(0), "0 person");
  EXPECT_EQ(labels.At(1), "1 car");
}

TEST(LabelMapTest, RejectsInvalidIndexedLines) {
  EXPECT_THROW(
      static_cast<void>(LabelMapFromText("cat\n", LabelMapFormat::kIndexed)),
      std::invalid_argument);
  EXPECT_THROW(
      static_cast<void>(LabelMapFromText("0,\n", LabelMapFormat::kIndexed)),
      std::invalid_argument);
}

TEST(LabelMapTest, LoadsLabelMapFromFile) {
  const std::filesystem::path path = TemporaryLabelPath();
  {
    std::ofstream file(path);
    ASSERT_TRUE(file);
    file << "\xEF\xBB\xBF"
         << "0,zero\r\n"
         << "1: one\r\n";
  }

  LabelMap labels = LabelMapFromFile(path);
  std::filesystem::remove(path);

  EXPECT_EQ(labels.size(), 2U);
  EXPECT_EQ(labels.At(0), "zero");
  EXPECT_EQ(labels.At(1), "one");
}

TEST(LabelMapTest, ThrowsWhenLabelIsMissing) {
  LabelMap labels(std::vector<std::string>{"cat"});

  EXPECT_EQ(labels.At(0), "cat");
  EXPECT_THROW(static_cast<void>(labels.At(1)), std::out_of_range);
}

}  // namespace
}  // namespace mw::infer
