add_library(MwInferRuntime STATIC runtime/model.cpp
                                 runtime/tensor/tensor_allocator.cpp
                                 runtime/process/geometry.cpp)
add_library(MwInfer::Runtime ALIAS MwInferRuntime)

set_target_properties(MwInferRuntime PROPERTIES EXPORT_NAME Runtime)

target_include_directories(
  MwInferRuntime PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

if(TARGET fmt::fmt-header-only)
  target_link_libraries(MwInferRuntime
                        PRIVATE $<BUILD_INTERFACE:fmt::fmt-header-only>)
else()
  target_link_libraries(MwInferRuntime PRIVATE $<BUILD_INTERFACE:fmt::fmt>)
endif()

if(TARGET spdlog::spdlog_header_only)
  target_link_libraries(MwInferRuntime
                        PRIVATE $<BUILD_INTERFACE:spdlog::spdlog_header_only>)
else()
  target_link_libraries(MwInferRuntime
                        PRIVATE $<BUILD_INTERFACE:spdlog::spdlog>)
endif()

find_package(OpenCV QUIET COMPONENTS core imgproc OPTIONAL_COMPONENTS
                                      cudaarithm cudawarping)

set(MW_INFER_DEPENDENCY_OPENCV_STATUS "not found" PARENT_SCOPE)
set(MW_INFER_DEPENDENCY_OPENCV_CUDA_STATUS "not found" PARENT_SCOPE)

if(TARGET opencv_core AND TARGET opencv_imgproc)
  set(MW_INFER_DEPENDENCY_OPENCV_STATUS "found" PARENT_SCOPE)
  set(MW_INFER_FEATURE_OPENCV_ADAPTER_STATUS "enabled" PARENT_SCOPE)
  target_sources(
    MwInferRuntime
    PRIVATE runtime/input/opencv_input.cpp
            runtime/process/opencv_geometry.cpp)
  target_compile_definitions(MwInferRuntime
                             PRIVATE MW_INFER_HAS_OPENCV_GEOMETRY_ADAPTER=1)
  target_link_libraries(MwInferRuntime PUBLIC opencv_core opencv_imgproc)

  if(TARGET opencv_cudaarithm AND TARGET opencv_cudawarping)
    set(MW_INFER_DEPENDENCY_OPENCV_CUDA_STATUS "found" PARENT_SCOPE)
    set(MW_INFER_FEATURE_OPENCV_CUDA_ADAPTER_STATUS "enabled" PARENT_SCOPE)
    target_sources(
      MwInferRuntime
      PRIVATE runtime/input/opencv_cuda_input.cpp
              runtime/process/opencv_cuda_geometry.cpp)
    target_compile_definitions(MwInferRuntime
                               PRIVATE MW_INFER_HAS_OPENCV_CUDA_ADAPTER=1)
    target_link_libraries(MwInferRuntime PUBLIC opencv_cudaarithm
                                                opencv_cudawarping)
  endif()
endif()

find_package(CUDAToolkit QUIET)

set(MW_INFER_DEPENDENCY_CUDATOOLKIT_STATUS "not found" PARENT_SCOPE)

if(TARGET CUDA::cudart)
  set(MW_INFER_DEPENDENCY_CUDATOOLKIT_STATUS "found" PARENT_SCOPE)
  set(MW_INFER_FEATURE_CUDA_TENSOR_STATUS "enabled" PARENT_SCOPE)
  target_sources(MwInferRuntime PRIVATE runtime/tensor/cuda_tensor.cpp)
  target_link_libraries(MwInferRuntime PUBLIC CUDA::cudart)
endif()

find_package(ONNXRuntime QUIET)

set(MW_INFER_DEPENDENCY_ONNXRUNTIME_STATUS "not found" PARENT_SCOPE)
set(MW_INFER_DEPENDENCY_ONNXRUNTIME_CUDA_STATUS "not found" PARENT_SCOPE)

if(TARGET ONNXRuntime::ONNXRuntime)
  set(MW_INFER_DEPENDENCY_ONNXRUNTIME_STATUS "found" PARENT_SCOPE)
  set(MW_INFER_FEATURE_ONNXRUNTIME_BACKEND_STATUS "enabled" PARENT_SCOPE)
  target_sources(MwInferRuntime PRIVATE runtime/backend/onnx_runtime_backend.cpp)
  target_link_libraries(MwInferRuntime PUBLIC ONNXRuntime::ONNXRuntime)

  if(ONNXRuntime_CUDA_PROVIDER_FOUND)
    set(MW_INFER_DEPENDENCY_ONNXRUNTIME_CUDA_STATUS "found" PARENT_SCOPE)
    target_compile_definitions(
      MwInferRuntime PRIVATE MW_INFER_HAS_ONNXRUNTIME_CUDA_PROVIDER=1)
    if(TARGET ONNXRuntime::CUDAProvider)
      target_link_libraries(MwInferRuntime PUBLIC ONNXRuntime::CUDAProvider)
    endif()
    if(TARGET ONNXRuntime::SharedProvider)
      target_link_libraries(MwInferRuntime PUBLIC ONNXRuntime::SharedProvider)
    endif()
  endif()
endif()

install(TARGETS MwInferRuntime EXPORT MwInferTargets)
