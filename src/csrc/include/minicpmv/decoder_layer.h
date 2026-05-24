#pragma once

#include "minicpmv/tensor.h"

#include <acl/acl.h>

#include <cstdint>
#include <string>
#include <vector>

namespace minicpmv {

struct FullAttentionDecoderLayerConfig {
    int64_t num_q_heads;
    int64_t num_kv_heads;
    int64_t head_dim;
    int64_t rotary_dim;
    double rms_epsilon;
};

struct FullAttentionDecoderLayerWeights {
    const Tensor* input_norm_weight;
    const Tensor* post_attention_norm_weight;
    const Tensor* q_proj_weight;
    const Tensor* k_proj_weight;
    const Tensor* v_proj_weight;
    const Tensor* o_proj_weight;
    const Tensor* q_norm_weight;
    const Tensor* k_norm_weight;
    const Tensor* gate_proj_weight;
    const Tensor* up_proj_weight;
    const Tensor* down_proj_weight;
};

void full_attention_decoder_layer(const Tensor& hidden,
                                  const FullAttentionDecoderLayerWeights& weights,
                                  const Tensor& cos_table,
                                  const Tensor& sin_table,
                                  const std::vector<int32_t>& row_to_t,
                                  const FullAttentionDecoderLayerConfig& config,
                                  Tensor& out,
                                  aclrtStream stream);

struct FullAttentionLayerCache;

// Multi-token forward that also populates the layer's K/V cache rows [0, T).
// Equivalent to `full_attention_decoder_layer` for the hidden output. Caller is
// responsible for advancing DecodeState::seq_len after all layers have run.
void full_attention_decoder_layer_with_cache(const Tensor& hidden,
                                             const FullAttentionDecoderLayerWeights& weights,
                                             const Tensor& cos_table,
                                             const Tensor& sin_table,
                                             const std::vector<int32_t>& row_to_t,
                                             const FullAttentionDecoderLayerConfig& config,
                                             FullAttentionLayerCache& cache,
                                             Tensor& out,
                                             aclrtStream stream);

struct LinearAttentionDecoderLayerConfig {
    double rms_epsilon;
};

// Stub-only weights (residual + MLP skeleton). Real linear-attention path uses
// `LinearAttentionDecoderLayerWeights` below.
struct LinearAttentionDecoderLayerStubWeights {
    const Tensor* input_norm_weight;
    const Tensor* post_attention_norm_weight;
    const Tensor* gate_proj_weight;
    const Tensor* up_proj_weight;
    const Tensor* down_proj_weight;
};

void linear_attention_decoder_layer_stub(const Tensor& hidden,
                                         const LinearAttentionDecoderLayerStubWeights& weights,
                                         const LinearAttentionDecoderLayerConfig& config,
                                         Tensor& out,
                                         aclrtStream stream);

// Real NPU linear-attention layer. Currently fixed at:
// - hidden_size = 1024
// - num_heads   = 16
// - head_dim    = 128
// - conv kernel = 4
// host pre-computes beta = sigmoid(b) and decay = exp(-exp(A_log)*softplus(a + dt_bias)).
struct LinearAttentionDecoderLayerWeights {
    const Tensor* input_norm_weight;
    const Tensor* post_attention_norm_weight;
    const Tensor* in_proj_qkv_weight;
    const Tensor* in_proj_z_weight;
    const Tensor* in_proj_a_weight;
    const Tensor* in_proj_b_weight;
    const Tensor* conv1d_weight;
    const Tensor* dt_bias;
    const Tensor* a_log;
    const Tensor* gated_norm_weight;
    const Tensor* out_proj_weight;
    const Tensor* gate_proj_weight;
    const Tensor* up_proj_weight;
    const Tensor* down_proj_weight;
    // Optional pre-transposed [4, C] conv weight for the T=1 decode step. When
    // set, the step path uses the vectorized linear_causal_conv_step kernel
    // instead of the generic causal conv + last-row extract.
    const Tensor* conv1d_step_weight{nullptr};
};

void linear_attention_decoder_layer(const Tensor& hidden,
                                    const LinearAttentionDecoderLayerWeights& weights,
                                    const LinearAttentionDecoderLayerConfig& config,
                                    Tensor& out,
                                    aclrtStream stream);

// Single-token decode caches and entry points.
struct FullAttentionLayerCache {
    Tensor k_cache;  // [max_seq, num_kv_heads * head_dim] fp16, post-RoPE K rows
    Tensor v_cache;  // [max_seq, num_kv_heads * head_dim] fp16, V rows
};

struct LinearAttentionLayerCache {
    Tensor conv_buf;          // [3, 6144] fp16, last 3 conv inputs (pre-conv qkv projection)
    Tensor recurrent_state;   // [16, 128, 128] fp32, gated delta rule state
};

// Multi-token linear-attention forward that advances cache.conv_buf and
// cache.recurrent_state by T steps. Equivalent to T sequential
// `linear_attention_decoder_layer_step` calls in terms of cache state, but
// projections / conv1d / MLP run batched on [T, ...] inputs.
// Requires the cache to be in its initial (post-`make_decode_state`) state:
// both conv_buf and recurrent_state must be zero, since the multi-token
// conv1d assumes implicit zero history.
void linear_attention_decoder_layer_with_cache(const Tensor& hidden,
                                               const LinearAttentionDecoderLayerWeights& weights,
                                               const LinearAttentionDecoderLayerConfig& config,
                                               LinearAttentionLayerCache& cache,
                                               Tensor& out,
                                               aclrtStream stream);

struct DecodeState {
    int64_t max_seq_len{0};
    int64_t seq_len{0};
    std::vector<FullAttentionLayerCache> full;
    std::vector<LinearAttentionLayerCache> linear;
};

// Allocate caches for each layer in `layer_types` ("full_attention" or
// "linear_attention"), zero-initialize state buffers. Sized to `max_seq_len`
// tokens for full-attention K/V caches.
DecodeState make_decode_state(int64_t max_seq_len,
                              const std::vector<std::string>& layer_types,
                              const FullAttentionDecoderLayerConfig& full_config,
                              aclrtStream stream);

// Single-token full-attention step. `pos` is the absolute position of this
// token; `cache_len` is the number of valid tokens already in K/V (rows [0,
// cache_len) are valid; this call writes row `cache_len`). Caller bumps
// `seq_len` after the step.
void full_attention_decoder_layer_step(const Tensor& hidden_1xH,
                                       const FullAttentionDecoderLayerWeights& weights,
                                       const Tensor& cos_table,
                                       const Tensor& sin_table,
                                       int32_t pos,
                                       int64_t cache_len,
                                       const FullAttentionDecoderLayerConfig& config,
                                       FullAttentionLayerCache& cache,
                                       Tensor& out_1xH,
                                       aclrtStream stream);

// Single-token linear-attention step using device-resident conv buffer and
// recurrent state in `cache`.
void linear_attention_decoder_layer_step(const Tensor& hidden_1xH,
                                         const LinearAttentionDecoderLayerWeights& weights,
                                         const LinearAttentionDecoderLayerConfig& config,
                                         LinearAttentionLayerCache& cache,
                                         Tensor& out_1xH,
                                         aclrtStream stream);

}  // namespace minicpmv
