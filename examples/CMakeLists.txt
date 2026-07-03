if(MW_INFER_FETCH_DEPS)
  CPMFindPackage(
    NAME CLI11
    VERSION 2.5.0
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG v2.5.0
    FIND_PACKAGE_ARGUMENTS "CONFIG"
    OPTIONS "CLI11_BUILD_DOCS OFF" "CLI11_BUILD_EXAMPLES OFF"
            "CLI11_BUILD_TESTS OFF" "CLI11_INSTALL OFF")
else()
  find_package(CLI11 2.5.0 CONFIG REQUIRED)
endif()

add_executable(mw_infer_inspect inspect_model.cc)
target_link_libraries(mw_infer_inspect PRIVATE MwInfer::Runtime CLI11::CLI11)
