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

add_executable(mw_infer_runtime_config_test runtime_config_test.cc)
target_link_libraries(mw_infer_runtime_config_test PRIVATE MwInfer::Runtime
                                                           GTest::gtest_main)

add_executable(mw_infer_pipeline_test pipeline_test.cc)
target_link_libraries(mw_infer_pipeline_test PRIVATE MwInfer::Runtime
                                                     GTest::gtest_main)

add_executable(mw_infer_blocks_test blocks_test.cc)
target_link_libraries(mw_infer_blocks_test PRIVATE MwInfer::Runtime
                                                   GTest::gtest_main)

add_executable(mw_infer_opencv_blocks_test opencv_blocks_test.cc)
target_link_libraries(mw_infer_opencv_blocks_test PRIVATE MwInfer::Runtime
                                                          GTest::gtest_main)

if(MW_INFER_WITH_ONNXRUNTIME)
  add_executable(mw_infer_onnx_infer_test onnx_infer_test.cc)
  target_link_libraries(mw_infer_onnx_infer_test PRIVATE MwInfer::Runtime
                                                         GTest::gtest_main)
endif()

include(GoogleTest)
gtest_discover_tests(mw_infer_runtime_config_test)
gtest_discover_tests(mw_infer_pipeline_test)
gtest_discover_tests(mw_infer_blocks_test)
gtest_discover_tests(mw_infer_opencv_blocks_test)

if(MW_INFER_WITH_ONNXRUNTIME)
  gtest_discover_tests(mw_infer_onnx_infer_test)
endif()
