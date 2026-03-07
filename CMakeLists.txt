cmake_minimum_required(VERSION 3.18)
project(PM LANGUAGES CXX )
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
find_package(OpenMP REQUIRED)

add_subdirectory(test)

# find_package(MPI REQUIRED)
# add_subdirectory(test_mpi)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_ARCHITECTURES native)
    add_subdirectory(test_cuda)
else()
    message(STATUS "CUDA not found, skipping test_cuda")
endif()

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION include)