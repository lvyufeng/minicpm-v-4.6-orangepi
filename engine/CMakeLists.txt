cmake_minimum_required(VERSION 3.20)
project(minicpmv_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(ASCEND_TOOLKIT_ROOT "/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux" CACHE PATH "Ascend toolkit root")
set(CUSTOM_OPP_VENDOR "/mnt/data/minicpm/custom_opp_install/vendors/customize" CACHE PATH "Custom OPP vendor root")
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")

add_library(minicpmv_engine
  src/acl_context.cpp
  src/tensor.cpp
  src/weights.cpp
  src/ops.cpp
  src/decoder_layer.cpp
  src/language_model.cpp
)

target_include_directories(minicpmv_engine PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
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

add_executable(test_weights_load tests/test_weights_load.cpp)
target_link_libraries(test_weights_load PRIVATE minicpmv_engine)

target_include_directories(test_weights_load PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${ASCEND_TOOLKIT_ROOT}/include
)

add_executable(test_embedding tests/test_embedding.cpp)
target_link_libraries(test_embedding PRIVATE minicpmv_engine)
add_executable(test_matmul tests/test_matmul.cpp)
target_link_libraries(test_matmul PRIVATE minicpmv_engine)
add_executable(test_lm_head_pipeline tests/test_lm_head_pipeline.cpp)
target_link_libraries(test_lm_head_pipeline PRIVATE minicpmv_engine)
add_executable(test_elementwise tests/test_elementwise.cpp)
target_link_libraries(test_elementwise PRIVATE minicpmv_engine)
add_executable(test_rms_norm1024 tests/test_rms_norm1024.cpp)
target_link_libraries(test_rms_norm1024 PRIVATE minicpmv_engine)
add_executable(test_layer_norm tests/test_layer_norm.cpp)
target_link_libraries(test_layer_norm PRIVATE minicpmv_engine)
add_executable(test_token_forward tests/test_token_forward.cpp)
target_link_libraries(test_token_forward PRIVATE minicpmv_engine)
add_executable(test_cast tests/test_cast.cpp)
target_link_libraries(test_cast PRIVATE minicpmv_engine)
add_executable(test_rms_norm tests/test_rms_norm.cpp)
target_link_libraries(test_rms_norm PRIVATE minicpmv_engine)
add_executable(test_partial_rope_ref tests/test_partial_rope_ref.cpp)
target_link_libraries(test_partial_rope_ref PRIVATE minicpmv_engine)
add_executable(test_partial_rope tests/test_partial_rope.cpp)
target_link_libraries(test_partial_rope PRIVATE minicpmv_engine)
add_executable(test_layer3_full_attention_front tests/test_layer3_full_attention_front.cpp)
target_link_libraries(test_layer3_full_attention_front PRIVATE minicpmv_engine)
add_executable(test_layer3_full_attention tests/test_layer3_full_attention.cpp)
target_link_libraries(test_layer3_full_attention PRIVATE minicpmv_engine)
add_executable(test_layer3_decoder_layer tests/test_layer3_decoder_layer.cpp)
target_link_libraries(test_layer3_decoder_layer PRIVATE minicpmv_engine)
add_executable(test_decoder_stack tests/test_decoder_stack.cpp)
target_link_libraries(test_decoder_stack PRIVATE minicpmv_engine)
add_executable(test_linear_attention_projection tests/test_linear_attention_projection.cpp)
target_link_libraries(test_linear_attention_projection PRIVATE minicpmv_engine)
add_executable(test_linear_attention_conv tests/test_linear_attention_conv.cpp)
target_link_libraries(test_linear_attention_conv PRIVATE minicpmv_engine)
add_executable(test_linear_attention_gated_delta_cpu tests/test_linear_attention_gated_delta_cpu.cpp)
target_link_libraries(test_linear_attention_gated_delta_cpu PRIVATE minicpmv_engine)
add_executable(test_full_language_model tests/test_full_language_model.cpp)
target_link_libraries(test_full_language_model PRIVATE minicpmv_engine)
add_executable(test_autoregressive_decode tests/test_autoregressive_decode.cpp)
target_link_libraries(test_autoregressive_decode PRIVATE minicpmv_engine)
add_executable(test_lm_profile tests/test_lm_profile.cpp)
target_link_libraries(test_lm_profile PRIVATE minicpmv_engine)
add_executable(test_incremental_decode tests/test_incremental_decode.cpp)
target_link_libraries(test_incremental_decode PRIVATE minicpmv_engine)
add_executable(test_language_logits_dump tests/test_language_logits_dump.cpp)
target_link_libraries(test_language_logits_dump PRIVATE minicpmv_engine)
add_executable(test_prefill_from_embeddings tests/test_prefill_from_embeddings.cpp)
target_link_libraries(test_prefill_from_embeddings PRIVATE minicpmv_engine)
add_executable(minicpmv_hybrid_decode tools/minicpmv_hybrid_decode.cpp)
target_link_libraries(minicpmv_hybrid_decode PRIVATE minicpmv_engine)
add_executable(bench_matmul_throughput tests/bench_matmul_throughput.cpp)
target_link_libraries(bench_matmul_throughput PRIVATE minicpmv_engine)
add_executable(bench_npu_bandwidth tests/bench_npu_bandwidth.cpp)
target_link_libraries(bench_npu_bandwidth PRIVATE minicpmv_engine)
add_executable(bench_prefill tests/bench_prefill.cpp)
target_link_libraries(bench_prefill PRIVATE minicpmv_engine)
add_executable(bench_step_kernel tests/bench_step_kernel.cpp)
target_link_libraries(bench_step_kernel PRIVATE minicpmv_engine)
add_executable(bench_decode tests/bench_decode.cpp)
target_link_libraries(bench_decode PRIVATE minicpmv_engine)
add_executable(bench_matmul_vec tests/bench_matmul_vec.cpp)
target_link_libraries(bench_matmul_vec PRIVATE minicpmv_engine)
add_executable(bench_decode_subops tests/bench_decode_subops.cpp)
target_link_libraries(bench_decode_subops PRIVATE minicpmv_engine)
add_executable(test_matmul_cube tests/test_matmul_cube.cpp)
target_link_libraries(test_matmul_cube PRIVATE minicpmv_engine)
