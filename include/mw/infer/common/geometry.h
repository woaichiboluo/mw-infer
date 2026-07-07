#ifndef MW_INFER_COMMON_GEOMETRY_H_
#define MW_INFER_COMMON_GEOMETRY_H_

namespace mw::infer {

struct ImageSize {
  int width = 0;
  int height = 0;
};

struct Point2f {
  float x = 0.0F;
  float y = 0.0F;
};

struct Rect {
  int x = 0;
  int y = 0;
  int width = 0;
  int height = 0;
};

struct Rect2f {
  float x = 0.0F;
  float y = 0.0F;
  float width = 0.0F;
  float height = 0.0F;
};

struct Padding {
  int left = 0;
  int top = 0;
  int right = 0;
  int bottom = 0;
};

}  // namespace mw::infer

#endif  // MW_INFER_COMMON_GEOMETRY_H_
