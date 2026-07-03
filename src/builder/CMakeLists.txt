add_library(MwInferBuilder build_config.cc builder.cc)
add_library(MwInfer::Builder ALIAS MwInferBuilder)

target_include_directories(
  MwInferBuilder
  PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_link_libraries(MwInferBuilder PUBLIC MwInfer::Common)

install(TARGETS MwInferBuilder EXPORT MwInferTargets)
