add_library(MwInferRuntime STATIC runtime/backend/backend.cpp
                                 runtime/backend/model.cpp
                                 runtime/tensor/gather.cpp
                                 runtime/tensor/host_tensor_allocator.cpp
                                 runtime/tensor/tensor_allocator.cpp
                                 runtime/process/channel.cpp
                                 runtime/process/geometry.cpp
                                 runtime/process/image_to_tensor.cpp
                                 runtime/process/normalize.cpp
                                 runtime/postprocess/nms.cpp
                                 runtime/postprocess/softmax.cpp
                                 runtime/postprocess/topk.cpp)
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
  target_link_libraries(MwInferRuntime PRIVATE $<BUILD_INTERFACE:spdlog::spdlog>)
endif()

include(CheckLanguage)

find_package(CUDAToolkit QUIET)

set(MW_INFER_DEPENDENCY_CUDATOOLKIT_STATUS "not found" PARENT_SCOPE)

set(MW_INFER_CUDA_LANGUAGE_AVAILABLE FALSE)
if(TARGET CUDA::cudart)
  if(WIN32 AND MSVC)
    set(MW_INFER_CUDA_MSVC_FLAGS "--allow-unsupported-compiler"
                                 "--forward-slash-prefix-opts")
    foreach(cuda_msvc_flag IN LISTS MW_INFER_CUDA_MSVC_FLAGS)
      foreach(flag_var CMAKE_CUDA_FLAGS CMAKE_CUDA_FLAGS_INIT)
        if(NOT "${${flag_var}}" MATCHES
               "(^|[ ;])${cuda_msvc_flag}([ ;]|$)")
          string(APPEND ${flag_var} " ${cuda_msvc_flag}")
        endif()
      endforeach()
    endforeach()
  endif()

  if(CUDAToolkit_NVCC_EXECUTABLE AND NOT CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER
        "${CUDAToolkit_NVCC_EXECUTABLE}"
        CACHE FILEPATH "CUDA compiler")
  endif()
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    if(NOT CMAKE_CUDA_ARCHITECTURES)
      set(MW_INFER_CUDA_ARCHITECTURES OFF)
    else()
      set(MW_INFER_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    endif()
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(MW_INFER_CUDA_LANGUAGE_AVAILABLE TRUE)
  endif()
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
            runtime/process/opencv_geometry.cpp
            runtime/process/opencv_image_to_tensor.cpp)
  target_compile_definitions(MwInferRuntime
                             PRIVATE MW_INFER_HAS_OPENCV_GEOMETRY_ADAPTER=1
                                     MW_INFER_HAS_OPENCV_IMAGE_TO_TENSOR_ADAPTER=1)
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

    if(TARGET CUDA::cudart)
      target_sources(MwInferRuntime
                     PRIVATE runtime/process/opencv_cuda_image_to_tensor.cpp)
      target_compile_definitions(
        MwInferRuntime
        PRIVATE MW_INFER_HAS_OPENCV_CUDA_IMAGE_TO_TENSOR_ADAPTER=1)
    endif()
  endif()
endif()

if(TARGET CUDA::cudart)
  set(MW_INFER_DEPENDENCY_CUDATOOLKIT_STATUS "found" PARENT_SCOPE)
  set(MW_INFER_FEATURE_CUDA_TENSOR_STATUS "enabled" PARENT_SCOPE)
  target_sources(MwInferRuntime
                 PRIVATE runtime/tensor/cuda_tensor_allocator.cpp)
  target_compile_definitions(MwInferRuntime
                             PUBLIC MW_INFER_HAS_CUDA_RUNTIME=1
                             PRIVATE MW_INFER_HAS_CUDA_TENSOR_ADAPTER=1)
  target_link_libraries(MwInferRuntime PUBLIC CUDA::cudart)

  if(MW_INFER_CUDA_LANGUAGE_AVAILABLE)
    set(MW_INFER_FEATURE_CUDA_TENSOR_OPS_STATUS "enabled" PARENT_SCOPE)
    set(MW_INFER_FEATURE_CUDA_PREPROCESS_STATUS "enabled" PARENT_SCOPE)
    set(MW_INFER_FEATURE_CUDA_POSTPROCESS_STATUS "enabled" PARENT_SCOPE)
    set_target_properties(
      MwInferRuntime PROPERTIES CUDA_ARCHITECTURES
                                "${MW_INFER_CUDA_ARCHITECTURES}")
    target_sources(MwInferRuntime PRIVATE runtime/tensor/gather.cu
                                          runtime/process/channel.cu
                                          runtime/process/normalize.cu
                                          runtime/postprocess/nms.cu
                                          runtime/postprocess/softmax.cu
                                          runtime/postprocess/topk.cu)
    target_compile_definitions(MwInferRuntime
                               PUBLIC MW_INFER_HAS_CUDA_PREPROCESS=1
                                      MW_INFER_HAS_CUDA_POSTPROCESS=1
                                      MW_INFER_HAS_CUDA_TENSOR_OPS=1)
    set_target_properties(MwInferRuntime
                          PROPERTIES CUDA_STANDARD 17
                                     CUDA_STANDARD_REQUIRED ON)
  endif()
endif()

find_package(ONNXRuntime QUIET)

set(MW_INFER_DEPENDENCY_ONNXRUNTIME_STATUS "not found" PARENT_SCOPE)
set(MW_INFER_DEPENDENCY_ONNXRUNTIME_CUDA_STATUS "not found" PARENT_SCOPE)

if(TARGET ONNXRuntime::ONNXRuntime)
  set(MW_INFER_DEPENDENCY_ONNXRUNTIME_STATUS "found" PARENT_SCOPE)
  set(MW_INFER_FEATURE_ONNXRUNTIME_BACKEND_STATUS "enabled" PARENT_SCOPE)
  target_sources(MwInferRuntime PRIVATE runtime/backend/onnx_backend.cpp)
  target_compile_definitions(MwInferRuntime
                             PRIVATE MW_INFER_HAS_ONNXRUNTIME_BACKEND=1)
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
