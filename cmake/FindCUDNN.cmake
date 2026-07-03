# Copyright (c) OpenMMLab. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Based on MMDeploy's FindCUDNN.cmake and adapted for MwInfer.

set(CUDNN_ROOT "" CACHE PATH "Optional cuDNN installation prefix.")
set(CUDNN_DIR "" CACHE PATH "Optional cuDNN installation prefix.")

if(NOT CUDNN_DIR AND CUDNN_ROOT)
  set(CUDNN_DIR "${CUDNN_ROOT}")
endif()

if(NOT CUDNN_DIR AND DEFINED ENV{CUDNN_DIR})
  set(CUDNN_DIR "$ENV{CUDNN_DIR}")
endif()

if(NOT CUDNN_DIR AND DEFINED ENV{CUDNN_ROOT})
  set(CUDNN_DIR "$ENV{CUDNN_ROOT}")
endif()

find_path(
  CUDNN_INCLUDE_DIR
  NAMES cudnn.h
  HINTS ${CUDNN_DIR} ${CUDAToolkit_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES include
  NO_SYSTEM_ENVIRONMENT_PATH)

find_library(
  CUDNN_LIBRARY
  NAMES cudnn
  HINTS ${CUDNN_DIR} ${CUDAToolkit_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64
  NO_SYSTEM_ENVIRONMENT_PATH)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUDNN REQUIRED_VARS CUDNN_INCLUDE_DIR
                                                      CUDNN_LIBRARY)

if(CUDNN_FOUND AND NOT TARGET CUDNN::cudnn)
  add_library(CUDNN::cudnn UNKNOWN IMPORTED)
  set_target_properties(
    CUDNN::cudnn
    PROPERTIES IMPORTED_LOCATION "${CUDNN_LIBRARY}"
               INTERFACE_INCLUDE_DIRECTORIES "${CUDNN_INCLUDE_DIR}")
endif()

mark_as_advanced(CUDNN_INCLUDE_DIR CUDNN_LIBRARY)
