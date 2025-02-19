# Copyright 2025 Alişah Özcan
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)
project(PIRonGPU VERSION 1.0 LANGUAGES C CXX CUDA ASM)

set(BINARY_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/bin)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()
message(STATUS "Setting CMAKE_CUDA_ARCHITECTURES to ${CMAKE_CUDA_ARCHITECTURES} for PIRonGPU")

find_package(HEonGPU QUIET)

if(NOT HEonGPU_FOUND)
    message(STATUS "HEonGPU not found, fetching...")
    include(FetchContent)
    FetchContent_Declare(
        heongpu
        GIT_REPOSITORY https://github.com/Alisah-Ozcan/HEonGPU.git
        GIT_TAG        v1.1.0
    )
    FetchContent_MakeAvailable(heongpu)
else()
    message(STATUS "HEonGPU found!")
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
add_subdirectory(src)

#########
# Tests #
#########

option(PIRonGPU_BUILD_TESTS "Build PIRonGPU Tests" OFF)
message(STATUS "PIRonGPU_BUILD_TESTS: ${PIRonGPU_BUILD_TESTS}")
if(PIRonGPU_BUILD_TESTS)
    add_subdirectory(test)
endif()

############
# Examples #
############

option(PIRonGPU_BUILD_EXAMPLES "Build PIRonGPU Examples" OFF)
message(STATUS "PIRonGPU_BUILD_EXAMPLES: ${PIRonGPU_BUILD_EXAMPLES}")
if(PIRonGPU_BUILD_EXAMPLES)
    add_subdirectory(example)
endif()

