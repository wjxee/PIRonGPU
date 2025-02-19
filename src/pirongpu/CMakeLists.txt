# Copyright 2025 Alişah Özcan
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

file(GLOB_RECURSE LIB_SOURCE_FILES
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    CONFIGURE_DEPENDS
    lib/*.cu
    lib/*.cpp
)

file(GLOB_RECURSE LIB_HEADER_FILES
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    CONFIGURE_DEPENDS
    include/*.cuh
    include/*.h
)

add_library(pirongpu STATIC ${LIB_SOURCE_FILES})

target_link_libraries(pirongpu PUBLIC 
    heongpu
    OpenMP::OpenMP_CXX
)

target_compile_options(pirongpu PUBLIC $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler=-fopenmp>)

set_target_properties(pirongpu PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_RUNTIME_LIBRARY Static
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
)

target_include_directories(
    pirongpu
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)