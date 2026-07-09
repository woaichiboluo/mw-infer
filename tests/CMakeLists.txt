if(MW_INFER_FETCH_DEPS)
  CPMFindPackage(
    NAME GTest
    VERSION 1.17.0
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.17.0
    OPTIONS "BUILD_GMOCK OFF" "gtest_force_shared_crt ON" "INSTALL_GTEST OFF")
else()
  find_package(GTest 1.17.0 REQUIRED)
endif()

if(TARGET GTest::gtest_main)
  set(MW_INFER_DEPENDENCY_GTEST_STATUS "found" PARENT_SCOPE)
endif()

include(GoogleTest)

set(_MW_INFER_GTEST_DISCOVERY_TIMEOUT 30)

function(mw_infer_add_gtest target source label)
  add_executable(${target} ${source})
  target_link_libraries(${target} PRIVATE MwInfer::Runtime GTest::gtest_main)
  gtest_discover_tests(${target}
                       DISCOVERY_TIMEOUT ${_MW_INFER_GTEST_DISCOVERY_TIMEOUT}
                       PROPERTIES LABELS ${label})
endfunction()

function(mw_infer_copy_onnxruntime_dlls target)
  if(WIN32 AND ONNXRUNTIME_DIR)
    file(GLOB _MW_INFER_ONNXRUNTIME_DLLS
         "${ONNXRUNTIME_DIR}/lib/onnxruntime*.dll")
    if(_MW_INFER_ONNXRUNTIME_DLLS)
      add_custom_command(
        TARGET ${target}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${_MW_INFER_ONNXRUNTIME_DLLS} $<TARGET_FILE_DIR:${target}>)
    endif()
  endif()
endfunction()

mw_infer_add_gtest(mw_infer_model_test runtime/backend/model_test.cpp
                   runtime_backend)
mw_infer_add_gtest(mw_infer_input_test runtime/input/input_test.cpp
                   runtime_input)
mw_infer_add_gtest(mw_infer_tensor_test runtime/tensor/tensor_test.cpp
                   runtime_tensor)
mw_infer_add_gtest(mw_infer_gather_test runtime/tensor/gather_test.cpp
                   runtime_tensor)
mw_infer_add_gtest(mw_infer_geometry_test runtime/process/geometry_test.cpp
                   runtime_process)
mw_infer_add_gtest(mw_infer_image_to_tensor_test
                   runtime/process/image_to_tensor_test.cpp runtime_process)
mw_infer_add_gtest(mw_infer_preprocess_test
                   runtime/process/preprocess_test.cpp runtime_process)
mw_infer_add_gtest(mw_infer_classification_postprocess_test
                   runtime/postprocess/classification_test.cpp
                   runtime_postprocess)
mw_infer_add_gtest(mw_infer_label_map_test
                   runtime/postprocess/label_map_test.cc runtime_postprocess)
mw_infer_add_gtest(mw_infer_nms_test runtime/postprocess/nms_test.cpp
                   runtime_postprocess)
mw_infer_add_gtest(mw_infer_segmentation_test
                   runtime/postprocess/segmentation_test.cc
                   runtime_postprocess)
mw_infer_add_gtest(mw_infer_yolo_decode_test
                   runtime/postprocess/yolo_decode_test.cpp
                   runtime_postprocess)

if(MW_INFER_FEATURE_OPENCV_ADAPTER_STATUS STREQUAL "enabled")
  mw_infer_add_gtest(mw_infer_opencv_input_test
                     runtime/input/opencv_input_test.cpp runtime_input)
  mw_infer_add_gtest(mw_infer_opencv_geometry_test
                     runtime/process/opencv_geometry_test.cpp runtime_process)
  mw_infer_add_gtest(mw_infer_opencv_image_to_tensor_test
                     runtime/process/opencv_image_to_tensor_test.cpp
                     runtime_process)
endif()

if(MW_INFER_FEATURE_OPENCV_CUDA_ADAPTER_STATUS STREQUAL "enabled")
  mw_infer_add_gtest(mw_infer_opencv_cuda_input_test
                     runtime/input/opencv_cuda_input_test.cpp runtime_input)
  mw_infer_add_gtest(mw_infer_opencv_cuda_geometry_test
                     runtime/process/opencv_cuda_geometry_test.cpp
                     runtime_process)
endif()

if(MW_INFER_FEATURE_OPENCV_CUDA_ADAPTER_STATUS STREQUAL "enabled"
   AND MW_INFER_FEATURE_CUDA_TENSOR_STATUS STREQUAL "enabled")
  mw_infer_add_gtest(mw_infer_opencv_cuda_image_to_tensor_test
                     runtime/process/opencv_cuda_image_to_tensor_test.cpp
                     runtime_process)
endif()

if(MW_INFER_FEATURE_ONNXRUNTIME_BACKEND_STATUS STREQUAL "enabled")
  mw_infer_add_gtest(mw_infer_onnx_backend_test
                     runtime/backend/onnx_backend_test.cpp runtime_backend)
  mw_infer_copy_onnxruntime_dlls(mw_infer_onnx_backend_test)
  if(MW_INFER_FEATURE_OPENCV_ADAPTER_STATUS STREQUAL "enabled")
    mw_infer_add_gtest(mw_infer_onnx_dynamic_image_pipeline_test
                       runtime/backend/onnx_dynamic_image_pipeline_test.cc
                       runtime_backend)
    mw_infer_copy_onnxruntime_dlls(mw_infer_onnx_dynamic_image_pipeline_test)
  endif()
endif()

if(MW_INFER_FEATURE_CUDA_TENSOR_STATUS STREQUAL "enabled")
  mw_infer_add_gtest(mw_infer_cuda_tensor_allocator_test
                     runtime/tensor/cuda_tensor_allocator_test.cpp
                     runtime_tensor)
endif()
