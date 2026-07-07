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

add_executable(mw_infer_input_test input_test.cpp)
target_link_libraries(mw_infer_input_test PRIVATE MwInfer::Runtime
                                                  GTest::gtest_main)

add_executable(mw_infer_tensor_test tensor_test.cpp)
target_link_libraries(mw_infer_tensor_test PRIVATE MwInfer::Common
                                                   GTest::gtest_main)

add_executable(mw_infer_geometry_test geometry_test.cpp)
target_link_libraries(mw_infer_geometry_test PRIVATE MwInfer::Runtime
                                                     GTest::gtest_main)

if(TARGET MwInfer::OpenCVAdapter)
  add_executable(mw_infer_opencv_input_test opencv_input_test.cpp)
  target_link_libraries(mw_infer_opencv_input_test
                        PRIVATE MwInfer::OpenCVAdapter GTest::gtest_main)

  add_executable(mw_infer_opencv_geometry_test opencv_geometry_test.cpp)
  target_link_libraries(mw_infer_opencv_geometry_test
                        PRIVATE MwInfer::OpenCVAdapter GTest::gtest_main)
endif()

if(TARGET MwInfer::OpenCVCudaAdapter)
  add_executable(mw_infer_opencv_cuda_input_test opencv_cuda_input_test.cpp)
  target_link_libraries(mw_infer_opencv_cuda_input_test
                        PRIVATE MwInfer::OpenCVCudaAdapter GTest::gtest_main)

  add_executable(mw_infer_opencv_cuda_geometry_test
                 opencv_cuda_geometry_test.cpp)
  target_link_libraries(mw_infer_opencv_cuda_geometry_test
                        PRIVATE MwInfer::OpenCVCudaAdapter GTest::gtest_main)
endif()

if(TARGET MwInfer::OnnxRuntimeBackends)
  add_executable(mw_infer_onnx_runtime_backends_test
                 onnx_runtime_backend_test.cpp)
  target_link_libraries(mw_infer_onnx_runtime_backends_test
                        PRIVATE MwInfer::OnnxRuntimeBackends GTest::gtest_main)
endif()

if(TARGET MwInfer::CudaTensor)
  add_executable(mw_infer_cuda_tensor_test cuda_tensor_test.cpp)
  target_link_libraries(mw_infer_cuda_tensor_test
                        PRIVATE MwInfer::CudaTensor GTest::gtest_main)
endif()

include(GoogleTest)

set(_MW_INFER_GTEST_DISCOVERY_TIMEOUT 30)

gtest_discover_tests(mw_infer_input_test DISCOVERY_TIMEOUT
                     ${_MW_INFER_GTEST_DISCOVERY_TIMEOUT})
gtest_discover_tests(mw_infer_tensor_test DISCOVERY_TIMEOUT
                     ${_MW_INFER_GTEST_DISCOVERY_TIMEOUT})
gtest_discover_tests(mw_infer_geometry_test DISCOVERY_TIMEOUT
                     ${_MW_INFER_GTEST_DISCOVERY_TIMEOUT})

if(TARGET MwInfer::OpenCVAdapter)
  gtest_discover_tests(mw_infer_opencv_input_test DISCOVERY_TIMEOUT
                       ${_MW_INFER_GTEST_DISCOVERY_TIMEOUT})
  gtest_discover_tests(mw_infer_opencv_geometry_test DISCOVERY_TIMEOUT
                       ${_MW_INFER_GTEST_DISCOVERY_TIMEOUT})
endif()

if(TARGET MwInfer::OpenCVCudaAdapter)
  gtest_discover_tests(mw_infer_opencv_cuda_input_test DISCOVERY_TIMEOUT
                       ${_MW_INFER_GTEST_DISCOVERY_TIMEOUT})
  gtest_discover_tests(mw_infer_opencv_cuda_geometry_test DISCOVERY_TIMEOUT
                       ${_MW_INFER_GTEST_DISCOVERY_TIMEOUT})
endif()

if(TARGET MwInfer::OnnxRuntimeBackends)
  gtest_discover_tests(mw_infer_onnx_runtime_backends_test DISCOVERY_TIMEOUT
                       ${_MW_INFER_GTEST_DISCOVERY_TIMEOUT})
endif()

if(TARGET MwInfer::CudaTensor)
  gtest_discover_tests(mw_infer_cuda_tensor_test DISCOVERY_TIMEOUT
                       ${_MW_INFER_GTEST_DISCOVERY_TIMEOUT})
endif()
