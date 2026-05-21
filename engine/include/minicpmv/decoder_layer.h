#pragma once

#include "minicpmv/tensor.h"

#include <acl/acl.h>

#include <cstdint>
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

}  // namespace minicpmv
