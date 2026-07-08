cmake_minimum_required(VERSION 3.22)

project(MwInfer VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(MW_INFER_BUILD_RUNTIME "Build MwInfer runtime." ON)
option(MW_INFER_BUILD_TESTS "Build MwInfer tests." ON)
option(MW_INFER_BUILD_EXAMPLES "Build MwInfer examples." OFF)
option(MW_INFER_FETCH_DEPS
       "Fetch lightweight dependencies when find_package cannot locate them."
       ON)

if(MW_INFER_FETCH_DEPS)
  include("${PROJECT_SOURCE_DIR}/cmake/CPM.cmake")
endif()

set(MW_INFER_DEPENDENCY_OPENCV_STATUS "not checked")
set(MW_INFER_DEPENDENCY_OPENCV_CUDA_STATUS "not checked")
set(MW_INFER_DEPENDENCY_CUDATOOLKIT_STATUS "not checked")
set(MW_INFER_DEPENDENCY_ONNXRUNTIME_STATUS "not checked")
set(MW_INFER_DEPENDENCY_ONNXRUNTIME_CUDA_STATUS "not checked")
set(MW_INFER_DEPENDENCY_FMT_STATUS "not checked")
set(MW_INFER_DEPENDENCY_SPDLOG_STATUS "not checked")
set(MW_INFER_DEPENDENCY_GTEST_STATUS "not checked")
set(MW_INFER_DEPENDENCY_CLI11_STATUS "not checked")
set(MW_INFER_FEATURE_OPENCV_ADAPTER_STATUS "disabled")
set(MW_INFER_FEATURE_OPENCV_CUDA_ADAPTER_STATUS "disabled")
set(MW_INFER_FEATURE_CUDA_TENSOR_STATUS "disabled")
set(MW_INFER_FEATURE_CUDA_TENSOR_OPS_STATUS "disabled")
set(MW_INFER_FEATURE_CUDA_POSTPROCESS_STATUS "disabled")
set(MW_INFER_FEATURE_ONNXRUNTIME_BACKEND_STATUS "disabled")

if(MW_INFER_BUILD_RUNTIME)
  if(MW_INFER_FETCH_DEPS)
    CPMFindPackage(
      NAME fmt
      VERSION 11.2.0
      GIT_REPOSITORY https://github.com/fmtlib/fmt.git
      GIT_TAG 11.2.0
      EXCLUDE_FROM_ALL YES
      OPTIONS "FMT_DOC OFF" "FMT_TEST OFF" "FMT_INSTALL OFF")
    CPMFindPackage(
      NAME spdlog
      VERSION 1.15.3
      GIT_REPOSITORY https://github.com/gabime/spdlog.git
      GIT_TAG v1.15.3
      EXCLUDE_FROM_ALL YES
      OPTIONS "SPDLOG_BUILD_EXAMPLE OFF" "SPDLOG_BUILD_TESTS OFF"
              "SPDLOG_FMT_EXTERNAL ON" "SPDLOG_INSTALL OFF")
  else()
    find_package(fmt 11 CONFIG REQUIRED)
    find_package(spdlog 1.15 CONFIG REQUIRED)
  endif()

  if(TARGET fmt::fmt OR TARGET fmt::fmt-header-only)
    set(MW_INFER_DEPENDENCY_FMT_STATUS "found")
  endif()
  if(TARGET spdlog::spdlog OR TARGET spdlog::spdlog_header_only)
    set(MW_INFER_DEPENDENCY_SPDLOG_STATUS "found")
  endif()

  add_subdirectory(src)
endif()

if(MW_INFER_BUILD_EXAMPLES)
  if(NOT MW_INFER_BUILD_RUNTIME)
    message(FATAL_ERROR "MW_INFER_BUILD_EXAMPLES requires MW_INFER_BUILD_RUNTIME=ON.")
  endif()

  if(MW_INFER_FETCH_DEPS)
    CPMFindPackage(
      NAME CLI11
      VERSION 2.4.2
      GITHUB_REPOSITORY CLIUtils/CLI11
      GIT_TAG v2.4.2
      EXCLUDE_FROM_ALL YES
      OPTIONS "CLI11_BUILD_DOCS OFF" "CLI11_BUILD_EXAMPLES OFF"
              "CLI11_BUILD_TESTS OFF" "CLI11_INSTALL OFF")
  else()
    find_package(CLI11 2.4 CONFIG REQUIRED)
  endif()
  if(TARGET CLI11::CLI11)
    set(MW_INFER_DEPENDENCY_CLI11_STATUS "found")
  endif()

  add_subdirectory(examples)
endif()

if(MW_INFER_BUILD_TESTS)
  if(NOT MW_INFER_BUILD_RUNTIME)
    message(FATAL_ERROR "MW_INFER_BUILD_TESTS requires MW_INFER_BUILD_RUNTIME=ON.")
  endif()

  enable_testing()
  add_subdirectory(tests)
endif()

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
if(MW_INFER_BUILD_RUNTIME)
  install(EXPORT MwInferTargets
          NAMESPACE MwInfer::
          DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MwInfer)
endif()

message(STATUS "")
message(STATUS "MwInfer configuration summary")
message(STATUS "  Options:")
message(STATUS "    MW_INFER_BUILD_RUNTIME: ${MW_INFER_BUILD_RUNTIME}")
message(STATUS "    MW_INFER_BUILD_TESTS: ${MW_INFER_BUILD_TESTS}")
message(STATUS "    MW_INFER_BUILD_EXAMPLES: ${MW_INFER_BUILD_EXAMPLES}")
message(STATUS "    MW_INFER_FETCH_DEPS: ${MW_INFER_FETCH_DEPS}")
message(STATUS "  Dependencies:")
message(STATUS "    OpenCV core/imgproc: ${MW_INFER_DEPENDENCY_OPENCV_STATUS}")
message(STATUS "    OpenCV CUDA modules: ${MW_INFER_DEPENDENCY_OPENCV_CUDA_STATUS}")
message(STATUS "    CUDAToolkit: ${MW_INFER_DEPENDENCY_CUDATOOLKIT_STATUS}")
message(STATUS "    ONNXRuntime: ${MW_INFER_DEPENDENCY_ONNXRUNTIME_STATUS}")
message(STATUS
        "    ONNXRuntime CUDA provider: ${MW_INFER_DEPENDENCY_ONNXRUNTIME_CUDA_STATUS}")
message(STATUS "    fmt: ${MW_INFER_DEPENDENCY_FMT_STATUS}")
message(STATUS "    spdlog: ${MW_INFER_DEPENDENCY_SPDLOG_STATUS}")
message(STATUS "    GTest: ${MW_INFER_DEPENDENCY_GTEST_STATUS}")
message(STATUS "    CLI11: ${MW_INFER_DEPENDENCY_CLI11_STATUS}")
message(STATUS "  Runtime features:")
message(STATUS "    OpenCV adapter: ${MW_INFER_FEATURE_OPENCV_ADAPTER_STATUS}")
message(STATUS
        "    OpenCV CUDA adapter: ${MW_INFER_FEATURE_OPENCV_CUDA_ADAPTER_STATUS}")
message(STATUS "    CUDA tensor allocator: ${MW_INFER_FEATURE_CUDA_TENSOR_STATUS}")
message(STATUS "    CUDA tensor ops: ${MW_INFER_FEATURE_CUDA_TENSOR_OPS_STATUS}")
message(STATUS "    CUDA postprocess: ${MW_INFER_FEATURE_CUDA_POSTPROCESS_STATUS}")
message(STATUS
        "    ONNXRuntime backend: ${MW_INFER_FEATURE_ONNXRUNTIME_BACKEND_STATUS}")
message(STATUS "")
