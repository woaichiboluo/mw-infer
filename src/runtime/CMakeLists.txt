add_library(MwInferRuntime INTERFACE)

add_library(MwInfer::Runtime ALIAS MwInferRuntime)
add_library(MwInfer ALIAS MwInferRuntime)
add_library(MwInfer::MwInfer ALIAS MwInferRuntime)

target_include_directories(
  MwInferRuntime INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                           $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_link_libraries(MwInferRuntime INTERFACE MwInfer::Common)

if(MW_INFER_BUILD_OPENCV_ADAPTER)
  set(_MW_INFER_OPENCV_COMPONENTS core imgproc)
  if(MW_INFER_BUILD_OPENCV_CUDA_ADAPTER)
    list(APPEND _MW_INFER_OPENCV_COMPONENTS cudaarithm cudawarping)
  endif()

  find_package(OpenCV REQUIRED COMPONENTS ${_MW_INFER_OPENCV_COMPONENTS})

  set(_MW_INFER_OPENCV_CORE_LIBS)
  foreach(_MW_INFER_OPENCV_COMPONENT core imgproc)
    if(TARGET opencv_${_MW_INFER_OPENCV_COMPONENT})
      list(APPEND _MW_INFER_OPENCV_CORE_LIBS
           opencv_${_MW_INFER_OPENCV_COMPONENT})
    else()
      set(_MW_INFER_OPENCV_CORE_LIBS ${OpenCV_LIBS})
      break()
    endif()
  endforeach()

  add_library(MwInferOpenCVAdapter INTERFACE)
  add_library(MwInfer::OpenCVAdapter ALIAS MwInferOpenCVAdapter)

  target_include_directories(
    MwInferOpenCVAdapter
    INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
              $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
              ${OpenCV_INCLUDE_DIRS})

  target_link_libraries(
    MwInferOpenCVAdapter INTERFACE MwInfer::Runtime
                                   ${_MW_INFER_OPENCV_CORE_LIBS})

  install(TARGETS MwInferOpenCVAdapter EXPORT MwInferTargets)

  if(MW_INFER_BUILD_OPENCV_CUDA_ADAPTER)
    set(_MW_INFER_OPENCV_CUDA_LIBS)
    foreach(_MW_INFER_OPENCV_CUDA_COMPONENT cudaarithm cudawarping)
      if(TARGET opencv_${_MW_INFER_OPENCV_CUDA_COMPONENT})
        list(APPEND _MW_INFER_OPENCV_CUDA_LIBS
             opencv_${_MW_INFER_OPENCV_CUDA_COMPONENT})
      else()
        set(_MW_INFER_OPENCV_CUDA_LIBS ${OpenCV_LIBS})
        break()
      endif()
    endforeach()

    add_library(MwInferOpenCVCudaAdapter INTERFACE)
    add_library(MwInfer::OpenCVCudaAdapter ALIAS MwInferOpenCVCudaAdapter)

    target_include_directories(
      MwInferOpenCVCudaAdapter
      INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
                ${OpenCV_INCLUDE_DIRS})

    target_compile_definitions(MwInferOpenCVCudaAdapter
                               INTERFACE MW_INFER_HAS_OPENCV_CUDA_ADAPTER=1)

    target_link_libraries(MwInferOpenCVCudaAdapter
                          INTERFACE MwInfer::OpenCVAdapter
                                    ${_MW_INFER_OPENCV_CUDA_LIBS})

    install(TARGETS MwInferOpenCVCudaAdapter EXPORT MwInferTargets)
  endif()
endif()

install(TARGETS MwInferRuntime EXPORT MwInferTargets)

if(MW_INFER_BUILD_CUDA_TENSOR)
  find_package(CUDAToolkit REQUIRED)

  add_library(MwInferCudaTensor STATIC
              allocator_adapters/cuda_tensor_allocator.cpp)
  add_library(MwInfer::CudaTensor ALIAS MwInferCudaTensor)

  target_include_directories(
    MwInferCudaTensor PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                             $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

  target_link_libraries(MwInferCudaTensor PUBLIC MwInfer::Runtime CUDA::cudart)

  install(TARGETS MwInferCudaTensor EXPORT MwInferTargets)
endif()

if(MW_INFER_ENABLE_ONNXRUNTIME)
  find_package(ONNXRuntime REQUIRED)

  add_library(MwInferOnnxRuntimeBackends STATIC
              backends/onnx_runtime_backend.cpp)
  add_library(MwInfer::OnnxRuntimeBackends ALIAS MwInferOnnxRuntimeBackends)

  target_include_directories(
    MwInferOnnxRuntimeBackends
    PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
           $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

  target_link_libraries(MwInferOnnxRuntimeBackends
                        PUBLIC MwInfer::Runtime ONNXRuntime::ONNXRuntime)

  if(ONNXRuntime_CUDA_PROVIDER_FOUND)
    target_compile_definitions(MwInferOnnxRuntimeBackends
                               PRIVATE MW_INFER_HAS_ONNXRUNTIME_CUDA_PROVIDER=1)
    if(TARGET ONNXRuntime::CUDAProvider)
      target_link_libraries(MwInferOnnxRuntimeBackends
                            PUBLIC ONNXRuntime::CUDAProvider)
    endif()
    if(TARGET ONNXRuntime::SharedProvider)
      target_link_libraries(MwInferOnnxRuntimeBackends
                            PUBLIC ONNXRuntime::SharedProvider)
    endif()
  endif()

  install(TARGETS MwInferOnnxRuntimeBackends EXPORT MwInferTargets)
endif()
