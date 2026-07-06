#include "mw/infer/runtime/pipeline.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mw/infer/runtime/infer_outputs.h"

namespace mw::infer {
namespace {

struct AddOne {
  using Input = int;
  using Output = int;

  Output Run(const Input& input, RunContext&) const { return input + 1; }
};

struct ToString {
  using Input = int;
  using Output = std::string;

  Output Run(const Input& input, RunContext&) const {
    return std::to_string(input);
  }
};

struct TestImage {
  int width = 0;
  int height = 0;
};

struct TestRect {
  int x = 0;
  int y = 0;
  int width = 0;
  int height = 0;
};

struct TestFrame {
  ImageSize size;
};

using TestFrameBatch = std::vector<TestFrame>;

std::vector<ImageSize> GetTestFrameSizes(const TestFrameBatch& frames) {
  std::vector<ImageSize> sizes;
  sizes.reserve(frames.size());
  for (const TestFrame& frame : frames) {
    sizes.push_back(frame.size);
  }
  return sizes;
}

struct ExplicitGeometryResize
    : public IGeometryTransform<TestFrameBatch, TestFrameBatch> {
  using Input = TestFrameBatch;
  using Output = TestFrameBatch;

  Output Run(const Input& input, RunContext&) const {
    Output output;
    output.reserve(input.size());
    for (const TestFrame& frame : input) {
      output.push_back(
          TestFrame{ImageSize{frame.size.width * 2, frame.size.height * 2}});
    }
    return output;
  }

  GeometryUpdate GetGeometryUpdate(const Input& input,
                                   const Output& output) const override {
    return GeometryUpdate::FromSource(GetTestFrameSizes(input))
        .ThenResize(GetTestFrameSizes(output));
  }
};

struct AccidentalGeometryMethod {
  using Input = TestFrameBatch;
  using Output = TestFrameBatch;

  Output Run(const Input& input, RunContext&) const { return input; }

  GeometryUpdate GetGeometryUpdate(const Input& input, const Output&) const {
    return GeometryUpdate::FromSource(GetTestFrameSizes(input));
  }
};

struct DetectionResult {
  std::vector<TestRect> boxes;
};

struct RoiMeta {
  int detection_index = -1;
  TestRect rect;
};

struct RoiBatch {
  std::vector<int> areas;
  std::vector<RoiMeta> metas;
};

struct ClassificationBatch {
  std::vector<std::string> labels;
  std::vector<RoiMeta> metas;
};

struct ClassifiedDetections {
  std::vector<TestRect> boxes;
  std::vector<std::string> labels;
};

struct Detector {
  using Input = TestImage;
  using Output = DetectionResult;

  Output Run(const Input&, RunContext&) const {
    return DetectionResult{
        {TestRect{10, 20, 30, 40}, TestRect{100, 120, 20, 10}}};
  }
};

struct RoiExtractInput {
  const TestImage* image = nullptr;
  const DetectionResult* detections = nullptr;
};

struct RoiExtract {
  using Input = RoiExtractInput;
  using Output = RoiBatch;

  Output Run(const Input& input, RunContext&) const {
    EXPECT_NE(input.image, nullptr);
    EXPECT_NE(input.detections, nullptr);

    RoiBatch batch;
    for (size_t index = 0; index < input.detections->boxes.size(); ++index) {
      const auto& box = input.detections->boxes[index];
      batch.areas.push_back(box.width * box.height);
      batch.metas.push_back(
          RoiMeta{static_cast<int>(index),
                  TestRect{box.x, box.y, box.width, box.height}});
    }
    return batch;
  }
};

struct Classifier {
  using Input = RoiBatch;
  using Output = ClassificationBatch;

  Output Run(const Input& input, RunContext&) const {
    ClassificationBatch batch;
    batch.metas = input.metas;
    for (int area : input.areas) {
      batch.labels.push_back(area >= 1000 ? "large" : "small");
    }
    return batch;
  }
};

struct MergeInput {
  const DetectionResult* detections = nullptr;
  const ClassificationBatch* classifications = nullptr;
};

struct Merge {
  using Input = MergeInput;
  using Output = ClassifiedDetections;

  Output Run(const Input& input, RunContext&) const {
    EXPECT_NE(input.detections, nullptr);
    EXPECT_NE(input.classifications, nullptr);
    EXPECT_EQ(input.detections->boxes.size(),
              input.classifications->labels.size());

    ClassifiedDetections result;
    result.boxes = input.detections->boxes;
    result.labels = input.classifications->labels;
    return result;
  }
};

class ReusableInfer {
 public:
  using Input = int;
  using Output = InferOutputs;

  Output Run(const Input& input, RunContext&) {
    if (input <= 0) {
      throw std::invalid_argument("input size must be positive");
    }

    const size_t size = static_cast<size_t>(input);
    if (buffer_.size() < size) {
      buffer_.resize(size);
    }

    ++run_id_;
    std::fill(buffer_.begin(), buffer_.begin() + input, run_id_);

    InferOutput output;
    output.name = "scores";
    output.data_type = InferDataType::kInt32;
    output.shape = Shape{1, input};
    output.device = Device::Cpu();
    output.buffer.host = buffer_.data();
    output.buffer.size_bytes = size * sizeof(int);
    return InferOutputs{1, {std::move(output)}};
  }

 private:
  std::vector<int> buffer_;
  int run_id_ = 0;
};

struct SumInferOutputs {
  using Input = InferOutputs;
  using Output = int;

  Output Run(const Input& input, RunContext&) const {
    const auto& output = input.outputs.at(0);
    const auto* values = static_cast<const int*>(output.buffer.host);
    const size_t count = output.buffer.size_bytes / sizeof(int);

    int sum = 0;
    for (size_t index = 0; index < count; ++index) {
      sum += values[index];
    }
    return sum;
  }
};

TEST(SerialPipelineTest, RunsBlocksAndRecordsOutputs) {
  auto pipeline = MakeSerialPipeline<int>()
                      .Then("add_one", AddOne{})
                      .Then("to_string", ToString{});

  auto run = pipeline.RunWithContext(41);

  EXPECT_EQ(run.output, "42");

  ASSERT_NE(run.context.RootInput<int>(), nullptr);
  EXPECT_EQ(*run.context.RootInput<int>(), 41);

  ASSERT_NE(run.context.OutputOf<int>("add_one"), nullptr);
  EXPECT_EQ(*run.context.OutputOf<int>("add_one"), 42);

  EXPECT_EQ(run.context.OutputOf<std::string>("add_one"), nullptr);
  EXPECT_EQ(run.context.OutputOf<int>("missing"), nullptr);
}

TEST(SerialPipelineTest, BindsPreviousOutputsForRoiClassification) {
  auto pipeline =
      MakeSerialPipeline<TestImage>()
          .Then("detect", Detector{})
          .ThenWith("roi_extract", RoiExtract{},
                    [](const DetectionResult& current, RunContext& context) {
                      return RoiExtractInput{context.RootInput<TestImage>(),
                                             &current};
                    })
          .Then("classify", Classifier{})
          .ThenWith(
              "merge", Merge{},
              [](const ClassificationBatch& current, RunContext& context) {
                return MergeInput{context.OutputOf<DetectionResult>("detect"),
                                  &current};
              });

  auto run = pipeline.RunWithContext(TestImage{640, 480});

  ASSERT_EQ(run.output.boxes.size(), 2U);
  ASSERT_EQ(run.output.labels.size(), 2U);
  EXPECT_EQ(run.output.labels[0], "large");
  EXPECT_EQ(run.output.labels[1], "small");

  const auto* roi_batch = run.context.OutputOf<RoiBatch>("roi_extract");
  ASSERT_NE(roi_batch, nullptr);
  ASSERT_EQ(roi_batch->metas.size(), 2U);
  EXPECT_EQ(roi_batch->metas[0].detection_index, 0);
  EXPECT_EQ(roi_batch->metas[1].detection_index, 1);
}

TEST(SerialPipelineTest, AppliesGeometryOnlyForExplicitInterface) {
  auto explicit_pipeline = MakeSerialPipeline<TestFrameBatch>().Then(
      "resize", ExplicitGeometryResize{});

  auto explicit_run =
      explicit_pipeline.RunWithContext({TestFrame{ImageSize{20, 10}}});

  ASSERT_TRUE(explicit_run.context.HasGeometry());
  ASSERT_NE(explicit_run.context.GeometryAt(0), nullptr);
  const PointF origin = explicit_run.context.GeometryAt(0)->MapPointToOriginal(
      PointF{20.0F, 10.0F});
  EXPECT_FLOAT_EQ(origin.x, 10.0F);
  EXPECT_FLOAT_EQ(origin.y, 5.0F);

  auto accidental_pipeline = MakeSerialPipeline<TestFrameBatch>().Then(
      "accidental", AccidentalGeometryMethod{});

  auto accidental_run =
      accidental_pipeline.RunWithContext({TestFrame{ImageSize{20, 10}}});

  EXPECT_FALSE(accidental_run.context.HasGeometry());
}

TEST(SerialPipelineTest, RejectsDuplicateBlockNames) {
  auto pipeline = MakeSerialPipeline<int>()
                      .Then("duplicate", AddOne{})
                      .Then("duplicate", AddOne{});

  EXPECT_THROW(static_cast<void>(pipeline.Run(1)), std::invalid_argument);
}

TEST(PipelineSessionTest, ReusesStatefulBlocksAcrossRuns) {
  auto pipeline = MakePipeline<int>()
                      .Then("infer", ReusableInfer{})
                      .Then("sum", SumInferOutputs{});

  auto first_run = pipeline.RunWithContext(3);
  const auto* first_outputs = first_run.context.OutputOf<InferOutputs>("infer");
  ASSERT_NE(first_outputs, nullptr);
  ASSERT_EQ(first_outputs->outputs.size(), 1U);
  const void* first_ptr = first_outputs->outputs[0].buffer.host;

  auto second_run = pipeline.RunWithContext(3);
  const auto* second_outputs =
      second_run.context.OutputOf<InferOutputs>("infer");
  ASSERT_NE(second_outputs, nullptr);
  ASSERT_EQ(second_outputs->outputs.size(), 1U);

  EXPECT_EQ(first_run.output, 3);
  EXPECT_EQ(second_run.output, 6);
  EXPECT_EQ(second_outputs->outputs[0].buffer.host, first_ptr);
}

}  // namespace
}  // namespace mw::infer
