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

include(GoogleTest)
gtest_discover_tests(mw_infer_runtime_config_test)
