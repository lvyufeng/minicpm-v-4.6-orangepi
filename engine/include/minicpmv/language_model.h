#pragma once

#include "minicpmv/decoder_layer.h"
#include "minicpmv/tensor.h"
#include "minicpmv/weights.h"

#include <acl/acl.h>

#include <cstdint>
#include <string>
#include <vector>

namespace minicpmv {

struct LanguageModelConfig {
    int64_t hidden_size{1024};
    int64_t num_q_heads{8};
    int64_t num_kv_heads{2};
    int64_t head_dim{256};
    int64_t rotary_dim{64};
    double rope_theta{10000000.0};
    double rms_epsilon{1e-6};
    int64_t num_layers{24};
    int64_t vocab_size{248094};
    std::vector<std::string> layer_types;
};

LanguageModelConfig default_minicpmv46_lm_config();

struct LanguageModelLayerWeights {
    Tensor input_norm_w;
    Tensor post_norm_w;
    Tensor gate_w;
    Tensor up_w;
    Tensor down_w;
    // linear
    Tensor qkv_w;
    Tensor z_w;
    Tensor a_w;
    Tensor b_w;
    Tensor conv_w;
    Tensor dt_bias;
    Tensor a_log;
    Tensor gated_norm_w;
    Tensor out_proj_w;
    // full
    Tensor q_w;
    Tensor k_w;
    Tensor v_w;
    Tensor o_w;
    Tensor q_norm_w;
    Tensor k_norm_w;
};

struct LanguageModelWeights {
    Tensor embed;
    Tensor final_norm_w;
    std::vector<LanguageModelLayerWeights> layers;
};

LanguageModelWeights load_language_model_weights(WeightsIndex& index,
                                                 const LanguageModelConfig& cfg);

// Build [T, rotary_dim/2] cos/sin tables for positions [0, T).
void build_rope_tables(int64_t T,
                       const LanguageModelConfig& cfg,
                       Tensor& cos_table,
                       Tensor& sin_table);

// Prefill prompt embeddings (`prompt_hidden` shape [T, hidden_size] fp16),
// populating `state` so it is ready for single-token decode from position T.
// Returns the last-row hidden as a [1, hidden_size] fp16 tensor.
Tensor prefill_from_embeddings(const Tensor& prompt_hidden,
                               const LanguageModelWeights& w,
                               const LanguageModelConfig& cfg,
                               const Tensor& cos_table,
                               const Tensor& sin_table,
                               DecodeState& state,
                               aclrtStream stream);

// Apply final RMSNorm + tied-embedding LM head, return argmax token id.
int64_t lm_head_greedy(const Tensor& last_hidden_1xH,
                       const LanguageModelWeights& w,
                       const LanguageModelConfig& cfg,
                       aclrtStream stream);

// One decode step: embedding lookup, 24-layer step path, lm head, greedy
// argmax. `state.seq_len` is bumped by 1 on return.
int64_t decode_step_greedy(int32_t token_id,
                           const LanguageModelWeights& w,
                           const LanguageModelConfig& cfg,
                           const Tensor& cos_table,
                           const Tensor& sin_table,
                           DecodeState& state,
                           aclrtStream stream);

bool is_eos(int64_t token_id, const std::vector<int64_t>& eos_token_ids);

}  // namespace minicpmv
