#include "mw/infer/runtime/blocks.h"

#include <gtest/gtest.h>

namespace mw::infer {
namespace {

TEST(FunctionBlockTest, AdaptsPlainFunctionsIntoPipelineBlocks) {
  auto pipeline = MakePipeline<int>().Then(
      "double",
      MakeFunction<int, int>([](const int& input) { return input * 2; }));

  EXPECT_EQ(pipeline.Run(21), 42);
}

TEST(FunctionBlockTest, AllowsFunctionsToUseRunContext) {
  auto pipeline =
      MakePipeline<int>()
          .Then("add_one", MakeFunction<int, int>(
                               [](const int& input) { return input + 1; }))
          .Then("add_root", MakeFunction<int, int>(
                                [](const int& input, RunContext& context) {
                                  const int* root = context.RootInput<int>();
                                  return input + (root == nullptr ? 0 : *root);
                                }));

  EXPECT_EQ(pipeline.Run(10), 21);
}

}  // namespace
}  // namespace mw::infer
