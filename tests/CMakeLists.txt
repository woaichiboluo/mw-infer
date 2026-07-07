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

add_executable(mw_infer_input_test input_test.cpp)
target_link_libraries(mw_infer_input_test PRIVATE MwInfer::Runtime
                                                  GTest::gtest_main)

add_executable(mw_infer_tensor_test tensor_test.cpp)
target_link_libraries(mw_infer_tensor_test PRIVATE MwInfer::Runtime
                                                   GTest::gtest_main)

add_executable(mw_infer_model_test model_test.cpp)
target_link_libraries(mw_infer_model_test PRIVATE MwInfer::Runtime
                                                  GTest::gtest_main)

add_executable(mw_infer_geometry_test geometry_test.cpp)
target_link_libraries(mw_infer_geometry_test PRIVATE MwInfer::Runtime
                                                     GTest::gtest_main)

if(MW_INFER_FEATURE_OPENCV_ADAPTER_STATUS STREQUAL "enabled")
  add_executable(mw_infer_opencv_input_test opencv_input_test.cpp)
  target_link_libraries(mw_infer_opencv_input_test
                        PRIVATE MwInfer::Runtime GTest::gtest_main)

  add_executable(mw_infer_opencv_geometry_test opencv_geometry_test.cpp)
  target_link_libraries(mw_infer_opencv_geometry_test
                        PRIVATE MwInfer::Runtime GTest::gtest_main)
endif()

if(MW_INFER_FEATURE_OPENCV_CUDA_ADAPTER_STATUS STREQUAL "enabled")
  add_executable(mw_infer_opencv_cuda_input_test opencv_cuda_input_test.cpp)
  target_link_libraries(mw_infer_opencv_cuda_input_test
                        PRIVATE MwInfer::Runtime GTest::gtest_main)

  add_executable(mw_infer_opencv_cuda_geometry_test
                 opencv_cuda_geometry_test.cpp)
  target_link_libraries(mw_infer_opencv_cuda_geometry_test
                        PRIVATE MwInfer::Runtime GTest::gtest_main)
endif()

if(MW_INFER_FEATURE_ONNXRUNTIME_BACKEND_STATUS STREQUAL "enabled")
  add_executable(mw_infer_onnx_runtime_backends_test
                 onnx_runtime_backend_test.cpp)
  target_link_libraries(mw_infer_onnx_runtime_backends_test
                        PRIVATE MwInfer::Runtime GTest::gtest_main)
endif()

if(MW_INFER_FEATURE_CUDA_TENSOR_STATUS STREQUAL "enabled")
  add_executable(mw_infer_cuda_tensor_test cuda_tensor_test.cpp)
  target_link_libraries(mw_infer_cuda_tensor_test
                        PRIVATE MwInfer::Runtime GTest::gtest_main)
endif()

include(GoogleTest)

set(_MW_INFER_GTEST_DISCOVERY_TIMEOUT 30)

gtest_discover_tests(mw_infer_input_test DISCOVERY_TIMEOUT
                     ${_MW_INFER_GTEST_DISCOVERY_TIMEOUT})
gtest_discover_tests(mw_infer_tensor_test DISCOVERY_TIMEOUT
                     ${_MW_INFER_GTEST_DISCOVERY_TIMEOUT})
gtest_discover_tests(mw_infer_model_test DISCOVERY_TIMEOUT
                     ${_MW_INFER_GTEST_DISCOVERY_TIMEOUT})
gtest_discover_tests(mw_infer_geometry_test DISCOVERY_TIMEOUT
                     ${_MW_INFER_GTEST_DISCOVERY_TIMEOUT})

if(MW_INFER_FEATURE_OPENCV_ADAPTER_STATUS STREQUAL "enabled")
  gtest_discover_tests(mw_infer_opencv_input_test DISCOVERY_TIMEOUT
                       ${_MW_INFER_GTEST_DISCOVERY_TIMEOUT})
  gtest_discover_tests(mw_infer_opencv_geometry_test DISCOVERY_TIMEOUT
                       ${_MW_INFER_GTEST_DISCOVERY_TIMEOUT})
endif()

if(MW_INFER_FEATURE_OPENCV_CUDA_ADAPTER_STATUS STREQUAL "enabled")
  gtest_discover_tests(mw_infer_opencv_cuda_input_test DISCOVERY_TIMEOUT
                       ${_MW_INFER_GTEST_DISCOVERY_TIMEOUT})
  gtest_discover_tests(mw_infer_opencv_cuda_geometry_test DISCOVERY_TIMEOUT
                       ${_MW_INFER_GTEST_DISCOVERY_TIMEOUT})
endif()

if(MW_INFER_FEATURE_ONNXRUNTIME_BACKEND_STATUS STREQUAL "enabled")
  gtest_discover_tests(mw_infer_onnx_runtime_backends_test DISCOVERY_TIMEOUT
                       ${_MW_INFER_GTEST_DISCOVERY_TIMEOUT})
endif()

if(MW_INFER_FEATURE_CUDA_TENSOR_STATUS STREQUAL "enabled")
  gtest_discover_tests(mw_infer_cuda_tensor_test DISCOVERY_TIMEOUT
                       ${_MW_INFER_GTEST_DISCOVERY_TIMEOUT})
endif()
