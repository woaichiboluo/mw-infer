add_executable(model_check onnx/model_check.cpp)
target_link_libraries(model_check PRIVATE MwInfer::Runtime CLI11::CLI11)
if(TARGET fmt::fmt-header-only)
  target_link_libraries(model_check PRIVATE fmt::fmt-header-only)
else()
  target_link_libraries(model_check PRIVATE fmt::fmt)
endif()

find_package(OpenCV QUIET COMPONENTS imgcodecs)
if(TARGET opencv_imgcodecs)
  add_executable(classification_infer onnx/classification_infer.cpp)
  target_link_libraries(classification_infer PRIVATE MwInfer::Runtime
                                                     opencv_imgcodecs
                                                     CLI11::CLI11)
  if(TARGET fmt::fmt-header-only)
    target_link_libraries(classification_infer PRIVATE fmt::fmt-header-only)
  else()
    target_link_libraries(classification_infer PRIVATE fmt::fmt)
  endif()

  add_executable(yolo_raw_infer onnx/yolo_raw_infer.cpp)
  target_link_libraries(yolo_raw_infer PRIVATE MwInfer::Runtime
                                               opencv_imgcodecs)
  if(TARGET fmt::fmt-header-only)
    target_link_libraries(yolo_raw_infer PRIVATE fmt::fmt-header-only)
  else()
    target_link_libraries(yolo_raw_infer PRIVATE fmt::fmt)
  endif()

  add_executable(yolo_nms_infer onnx/yolo_nms_infer.cpp)
  target_link_libraries(yolo_nms_infer PRIVATE MwInfer::Runtime
                                               opencv_imgcodecs)
  if(TARGET fmt::fmt-header-only)
    target_link_libraries(yolo_nms_infer PRIVATE fmt::fmt-header-only)
  else()
    target_link_libraries(yolo_nms_infer PRIVATE fmt::fmt)
  endif()

  add_executable(segmentation_fcn_infer onnx/segmentation_fcn_infer.cpp)
  target_link_libraries(segmentation_fcn_infer PRIVATE MwInfer::Runtime
                                                       opencv_imgcodecs)
  if(TARGET fmt::fmt-header-only)
    target_link_libraries(segmentation_fcn_infer PRIVATE fmt::fmt-header-only)
  else()
    target_link_libraries(segmentation_fcn_infer PRIVATE fmt::fmt)
  endif()
endif()

if(WIN32 AND ONNXRUNTIME_DIR)
  file(GLOB _MW_INFER_ONNXRUNTIME_DLLS
       "${ONNXRUNTIME_DIR}/lib/onnxruntime*.dll")
  if(_MW_INFER_ONNXRUNTIME_DLLS)
    add_custom_command(
      TARGET model_check
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              ${_MW_INFER_ONNXRUNTIME_DLLS} $<TARGET_FILE_DIR:model_check>)
    if(TARGET classification_infer)
      add_custom_command(
        TARGET classification_infer
        POST_BUILD
        COMMAND
          ${CMAKE_COMMAND} -E copy_if_different ${_MW_INFER_ONNXRUNTIME_DLLS}
          $<TARGET_FILE_DIR:classification_infer>)
    endif()
    if(TARGET yolo_raw_infer)
      add_custom_command(
        TARGET yolo_raw_infer
        POST_BUILD
        COMMAND
          ${CMAKE_COMMAND} -E copy_if_different ${_MW_INFER_ONNXRUNTIME_DLLS}
          $<TARGET_FILE_DIR:yolo_raw_infer>)
    endif()
    if(TARGET yolo_nms_infer)
      add_custom_command(
        TARGET yolo_nms_infer
        POST_BUILD
        COMMAND
          ${CMAKE_COMMAND} -E copy_if_different ${_MW_INFER_ONNXRUNTIME_DLLS}
          $<TARGET_FILE_DIR:yolo_nms_infer>)
    endif()
    if(TARGET segmentation_fcn_infer)
      add_custom_command(
        TARGET segmentation_fcn_infer
        POST_BUILD
        COMMAND
          ${CMAKE_COMMAND} -E copy_if_different ${_MW_INFER_ONNXRUNTIME_DLLS}
          $<TARGET_FILE_DIR:segmentation_fcn_infer>)
    endif()
  endif()
endif()
