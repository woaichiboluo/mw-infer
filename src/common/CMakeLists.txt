add_library(MwInferCommon STATIC tensor_allocator.cpp)

add_library(MwInfer::Common ALIAS MwInferCommon)

target_include_directories(
  MwInferCommon PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                       $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

install(TARGETS MwInferCommon EXPORT MwInferTargets)
