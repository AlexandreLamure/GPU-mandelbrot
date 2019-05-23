cmake_minimum_required (VERSION 3.10)
project (GPGPU)

include(FetchContent)


FetchContent_Declare(
  GoogleBenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  )

FetchContent_Declare(
  CLI11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
  )

FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  )


if (NOT GoogleBenchmark_POPULATED)
  FetchContent_Populate(GoogleBenchmark)
  set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "From Gtest")
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "From Gtest")
  add_subdirectory(${googlebenchmark_SOURCE_DIR} ${googlebenchmark_BINARY_DIR})
endif()

if (NOT spdlog_POPULATED)
  FetchContent_Populate(spdlog)
  add_subdirectory(${spdlog_SOURCE_DIR} ${spdlog_BINARY_DIR})
endif()

if (NOT CLI11_POPULATED)
  FetchContent_Populate(CLI11)
  add_subdirectory(${cli11_SOURCE_DIR} ${cli11_BINARY_DIR})
endif()


find_package(PNG REQUIRED)
find_package(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-W -Wall -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -s")

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++17" )


include_directories(lib/include)

add_library(renderer src/render_cpu.hpp src/render_cpu.cpp src/render_gpu.hpp src/render_gpu.cpp)  # version etudiant
target_link_libraries(renderer)
set_target_properties(renderer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(renderer PRIVATE KERNEL_SOURCE_FILE=${CMAKE_SOURCE_DIR}/src/mandel_etu.cl)  # version etudiant

add_executable(mandel src/mandel.cpp)
target_link_libraries(mandel renderer PNG::PNG CLI11::CLI11)

add_executable(bench src/bench.cpp)
target_link_libraries(bench renderer benchmark)
