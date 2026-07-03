add_library(
  MwInferRuntime
  backend_factory.cc
  backends/onnx_cpu_backend.cc
  backends/onnx_cuda_backend.cc
  backends/tensorrt_backend.cc
  runtime_config.cc
  session.cc)

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

target_compile_definitions(
  MwInferRuntime
  PRIVATE MW_INFER_WITH_ONNXRUNTIME=$<BOOL:${MW_INFER_WITH_ONNXRUNTIME}>
          MW_INFER_WITH_ONNXRUNTIME_CUDA=$<BOOL:${MW_INFER_WITH_ONNXRUNTIME_CUDA}>
          MW_INFER_WITH_TENSORRT=$<BOOL:${MW_INFER_WITH_TENSORRT}>)

target_link_libraries(MwInferRuntime PUBLIC MwInfer::Common)

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
