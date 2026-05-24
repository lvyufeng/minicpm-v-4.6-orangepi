cmake_minimum_required(VERSION 3.20)
project(minicpmv_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Ascend toolkit root. Defaults to the standard CANN install on Orange Pi
# AIPro; override with -DASCEND_TOOLKIT_ROOT=... for other paths.
set(ASCEND_TOOLKIT_ROOT "/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux"
    CACHE PATH "Ascend toolkit root")

# Where the custom_opp .run installer was extracted. Defaults to
# `${CMAKE_SOURCE_DIR}/custom_opp_install/vendors/customize` (the location
# scripts/install_custom_ops.sh uses); override with -DCUSTOM_OPP_VENDOR=...
set(CUSTOM_OPP_VENDOR "${CMAKE_SOURCE_DIR}/custom_opp_install/vendors/customize"
    CACHE PATH "Custom OPP vendor root (output of custom_opp .run installer)")

set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")

add_library(minicpmv_engine
  src/csrc/lib/acl_context.cpp
  src/csrc/lib/tensor.cpp
  src/csrc/lib/weights.cpp
  src/csrc/lib/ops.cpp
  src/csrc/lib/decoder_layer.cpp
  src/csrc/lib/language_model.cpp
)

target_include_directories(minicpmv_engine PUBLIC
  ${CMAKE_SOURCE_DIR}/src/csrc/include
  ${ASCEND_TOOLKIT_ROOT}/include
  ${CUSTOM_OPP_VENDOR}/op_api/include
)

target_link_directories(minicpmv_engine PUBLIC
  ${ASCEND_TOOLKIT_ROOT}/lib64
  ${CUSTOM_OPP_VENDOR}/op_api/lib
)

target_link_libraries(minicpmv_engine PUBLIC
  ascendcl
  nnopbase
  opapi
  cust_opapi
)

target_link_options(minicpmv_engine PUBLIC
  "-Wl,-rpath,${CUSTOM_OPP_VENDOR}/op_api/lib"
)

# User-runnable tools.
add_executable(minicpmv_hybrid_decode src/csrc/tools/minicpmv_hybrid_decode.cpp)
target_link_libraries(minicpmv_hybrid_decode PRIVATE minicpmv_engine)

# Per-file test / bench targets — globbed so adding a tests/test_foo.cpp or
# bench/bench_foo.cpp file picks up automatically on the next cmake re-config.
file(GLOB TEST_SRCS  CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/tests/*.cpp)
file(GLOB BENCH_SRCS CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/bench/*.cpp)

foreach(test_src ${TEST_SRCS})
  get_filename_component(name ${test_src} NAME_WE)
  add_executable(${name} ${test_src})
  target_link_libraries(${name} PRIVATE minicpmv_engine)
endforeach()

foreach(bench_src ${BENCH_SRCS})
  get_filename_component(name ${bench_src} NAME_WE)
  # bench_ddr_bandwidth is a host-only NEON benchmark that doesn't link the
  # engine — handled separately by bench/Makefile / its own build line.
  if(NOT name STREQUAL "bench_ddr_bandwidth")
    add_executable(${name} ${bench_src})
    target_link_libraries(${name} PRIVATE minicpmv_engine)
  endif()
endforeach()
