cmake_minimum_required(VERSION 3.22)

project(MwInfer VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(_MW_INFER_DEFAULT_ONNXRUNTIME OFF)
if(ONNXRUNTIME_ROOT
   OR ONNXRUNTIME_DIR
   OR DEFINED ENV{ONNXRUNTIME_ROOT}
   OR DEFINED ENV{ONNXRUNTIME_DIR})
  set(_MW_INFER_DEFAULT_ONNXRUNTIME ON)
endif()

set(_MW_INFER_DEFAULT_TENSORRT OFF)
if(TENSORRT_ROOT
   OR TENSORRT_DIR
   OR DEFINED ENV{TENSORRT_ROOT}
   OR DEFINED ENV{TENSORRT_DIR})
  set(_MW_INFER_DEFAULT_TENSORRT ON)
endif()

option(MW_INFER_ENABLE_ONNXRUNTIME "Enable ONNX Runtime backends."
       ${_MW_INFER_DEFAULT_ONNXRUNTIME})
option(MW_INFER_ENABLE_TENSORRT "Enable TensorRT backend."
       ${_MW_INFER_DEFAULT_TENSORRT})
option(MW_INFER_BUILD_RUNTIME "Build MwInfer runtime." ON)
option(MW_INFER_BUILD_BUILDER "Build MwInfer builder." OFF)
option(MW_INFER_BUILD_TESTS "Build MwInfer tests." ON)
option(MW_INFER_BUILD_EXAMPLES "Build MwInfer examples." OFF)
option(MW_INFER_FETCH_DEPS
       "Fetch lightweight dependencies when find_package cannot locate them."
       ON)

if(MW_INFER_FETCH_DEPS)
  include("${PROJECT_SOURCE_DIR}/cmake/CPM.cmake")
endif()

add_subdirectory(src/common)

if(MW_INFER_BUILD_RUNTIME)
  add_subdirectory(src/runtime)
endif()

if(MW_INFER_BUILD_BUILDER)
  add_subdirectory(src/builder)
endif()

if(MW_INFER_BUILD_EXAMPLES)
  if(NOT MW_INFER_BUILD_RUNTIME)
    message(FATAL_ERROR "MW_INFER_BUILD_EXAMPLES requires MW_INFER_BUILD_RUNTIME=ON.")
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
install(EXPORT MwInferTargets
        NAMESPACE MwInfer::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MwInfer)
