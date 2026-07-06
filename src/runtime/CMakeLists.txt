set(MW_INFER_RUNTIME_SOURCES
  backend_factory.cc
  backend.cc
  backends/onnx_cpu_backend.cc
  backends/onnx_gpu_backend.cc
  backends/onnx_runtime_session.cc
  backends/tensorrt_backend.cc
  blocks.cc
  onnx_infer.cc
  runtime_config.cc
  session.cc)

if(MW_INFER_WITH_OPENCV_CUDA)
  list(APPEND MW_INFER_RUNTIME_SOURCES cuda_blocks.cc)
endif()

add_library(MwInferRuntime ${MW_INFER_RUNTIME_SOURCES})

add_library(MwInfer::Runtime ALIAS MwInferRuntime)
add_library(MwInfer ALIAS MwInferRuntime)
add_library(MwInfer::MwInfer ALIAS MwInferRuntime)

target_include_directories(
  MwInferRuntime
  PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE ${PROJECT_SOURCE_DIR}/src)

set(MW_INFER_WITH_ONNXRUNTIME OFF)
set(MW_INFER_WITH_ONNXRUNTIME_CUDA OFF)
set(MW_INFER_WITH_TENSORRT OFF)

if(MW_INFER_ENABLE_ONNXRUNTIME)
  find_package(ONNXRuntime REQUIRED)
  set(MW_INFER_WITH_ONNXRUNTIME ON)

  if(ONNXRuntime_CUDA_PROVIDER_FOUND)
    set(MW_INFER_WITH_ONNXRUNTIME_CUDA ON)
  endif()
endif()

if(MW_INFER_ENABLE_TENSORRT)
  find_package(TensorRT REQUIRED)
  find_package(CUDAToolkit REQUIRED)
  find_package(CUDNN REQUIRED)
  set(MW_INFER_WITH_TENSORRT ON)
endif()

message(STATUS "MwInfer ONNX Runtime backend: ${MW_INFER_WITH_ONNXRUNTIME}")
message(STATUS "MwInfer ONNX Runtime CUDA backend: ${MW_INFER_WITH_ONNXRUNTIME_CUDA}")
message(STATUS "MwInfer TensorRT backend: ${MW_INFER_WITH_TENSORRT}")

set(MW_INFER_WITH_ONNXRUNTIME
    ${MW_INFER_WITH_ONNXRUNTIME}
    PARENT_SCOPE)
set(MW_INFER_WITH_ONNXRUNTIME_CUDA
    ${MW_INFER_WITH_ONNXRUNTIME_CUDA}
    PARENT_SCOPE)
set(MW_INFER_WITH_TENSORRT
    ${MW_INFER_WITH_TENSORRT}
    PARENT_SCOPE)

target_compile_definitions(
  MwInferRuntime
  PUBLIC MW_INFER_WITH_OPENCV_CUDA=$<BOOL:${MW_INFER_WITH_OPENCV_CUDA}>
  PRIVATE MW_INFER_WITH_ONNXRUNTIME=$<BOOL:${MW_INFER_WITH_ONNXRUNTIME}>
          MW_INFER_WITH_ONNXRUNTIME_CUDA=$<BOOL:${MW_INFER_WITH_ONNXRUNTIME_CUDA}>
          MW_INFER_WITH_TENSORRT=$<BOOL:${MW_INFER_WITH_TENSORRT}>)

target_include_directories(MwInferRuntime PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(MwInferRuntime PUBLIC MwInfer::Common ${OpenCV_LIBS})

if(MW_INFER_WITH_OPENCV_CUDA)
  target_link_libraries(MwInferRuntime PUBLIC ${MW_INFER_OPENCV_CUDA_LIBS})
endif()

if(MW_INFER_WITH_ONNXRUNTIME)
  target_link_libraries(MwInferRuntime PRIVATE ONNXRuntime::ONNXRuntime)
endif()

if(MW_INFER_WITH_ONNXRUNTIME_CUDA)
  if(TARGET ONNXRuntime::CUDAProvider)
    target_link_libraries(MwInferRuntime PRIVATE ONNXRuntime::CUDAProvider)
  endif()

  if(TARGET ONNXRuntime::SharedProvider)
    target_link_libraries(MwInferRuntime PRIVATE ONNXRuntime::SharedProvider)
  endif()
endif()

if(MW_INFER_WITH_TENSORRT)
  target_link_libraries(
    MwInferRuntime PRIVATE TensorRT::nvinfer TensorRT::nvinfer_plugin
                           CUDA::cudart CUDNN::cudnn)
endif()

install(TARGETS MwInferRuntime EXPORT MwInferTargets)
