#include "minicpmv/language_model.h"

#include "minicpmv/acl_context.h"
#include "minicpmv/ops.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace minicpmv {

namespace {

uint16_t f32_to_f16_bits(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;
    if (exp <= 0) return static_cast<uint16_t>(sign);
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}

Tensor load_weight(WeightsIndex& index, const std::string& name) {
    return index.load_to_device_as(name, DType::Float16);
}

Tensor load_layer_weight(WeightsIndex& index, int layer, const std::string& suffix) {
    return index.load_to_device_as("model.language_model.layers." + std::to_string(layer) + "." + suffix, DType::Float16);
}

void copy_row(const Tensor& src, int64_t src_row, Tensor& dst, int64_t dst_row, aclrtStream stream) {
    const size_t elem = dtype_size(src.dtype());
    const size_t row_bytes = static_cast<size_t>(src.shape()[1]) * elem;
    auto* s = static_cast<const uint8_t*>(src.data()) + static_cast<size_t>(src_row) * row_bytes;
    auto* d = static_cast<uint8_t*>(dst.data()) + static_cast<size_t>(dst_row) * row_bytes;
    check_acl(aclrtMemcpyAsync(d, row_bytes, s, row_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
              "lm copy_row");
    check_acl(aclrtSynchronizeStream(stream), "lm copy_row sync");
}

void copy_tensor(const Tensor& src, Tensor& dst, aclrtStream stream) {
    check_acl(aclrtMemcpyAsync(dst.data(), dst.size_bytes(), src.data(), src.size_bytes(),
                               ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
              "lm copy_tensor");
    check_acl(aclrtSynchronizeStream(stream), "lm copy_tensor sync");
}

}  // namespace

LanguageModelConfig default_minicpmv46_lm_config() {
    LanguageModelConfig cfg;
    cfg.layer_types = {
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
    };
    return cfg;
}

LanguageModelWeights load_language_model_weights(WeightsIndex& index,
                                                 const LanguageModelConfig& cfg) {
    LanguageModelWeights w;
    w.embed = load_weight(index, "model.language_model.embed_tokens.weight");
    w.final_norm_w = load_weight(index, "model.language_model.norm.weight");
    w.layers.resize(cfg.num_layers);
    for (int64_t layer = 0; layer < cfg.num_layers; ++layer) {
        auto& lw = w.layers[layer];
        lw.input_norm_w = load_layer_weight(index, static_cast<int>(layer), "input_layernorm.weight");
        lw.post_norm_w = load_layer_weight(index, static_cast<int>(layer), "post_attention_layernorm.weight");
        lw.gate_w = load_layer_weight(index, static_cast<int>(layer), "mlp.gate_proj.weight");
        lw.up_w = load_layer_weight(index, static_cast<int>(layer), "mlp.up_proj.weight");
        lw.down_w = load_layer_weight(index, static_cast<int>(layer), "mlp.down_proj.weight");
        if (cfg.layer_types[layer] == "linear_attention") {
            lw.qkv_w = load_layer_weight(index, static_cast<int>(layer), "linear_attn.in_proj_qkv.weight");
            lw.z_w = load_layer_weight(index, static_cast<int>(layer), "linear_attn.in_proj_z.weight");
            lw.a_w = load_layer_weight(index, static_cast<int>(layer), "linear_attn.in_proj_a.weight");
            lw.b_w = load_layer_weight(index, static_cast<int>(layer), "linear_attn.in_proj_b.weight");
            lw.conv_w = load_layer_weight(index, static_cast<int>(layer), "linear_attn.conv1d.weight");
            lw.dt_bias = load_layer_weight(index, static_cast<int>(layer), "linear_attn.dt_bias");
            lw.a_log = load_layer_weight(index, static_cast<int>(layer), "linear_attn.A_log");
            lw.gated_norm_w = load_layer_weight(index, static_cast<int>(layer), "linear_attn.norm.weight");
            lw.out_proj_w = load_layer_weight(index, static_cast<int>(layer), "linear_attn.out_proj.weight");
        } else {
            lw.q_w = load_layer_weight(index, static_cast<int>(layer), "self_attn.q_proj.weight");
            lw.k_w = load_layer_weight(index, static_cast<int>(layer), "self_attn.k_proj.weight");
            lw.v_w = load_layer_weight(index, static_cast<int>(layer), "self_attn.v_proj.weight");
            lw.o_w = load_layer_weight(index, static_cast<int>(layer), "self_attn.o_proj.weight");
            lw.q_norm_w = load_layer_weight(index, static_cast<int>(layer), "self_attn.q_norm.weight");
            lw.k_norm_w = load_layer_weight(index, static_cast<int>(layer), "self_attn.k_norm.weight");
        }
    }
    return w;
}

void build_rope_tables(int64_t T,
                       const LanguageModelConfig& cfg,
                       Tensor& cos_table,
                       Tensor& sin_table) {
    const int64_t half = cfg.rotary_dim / 2;
    std::vector<uint16_t> cos_host(static_cast<size_t>(T * half));
    std::vector<uint16_t> sin_host(static_cast<size_t>(T * half));
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t i = 0; i < half; ++i) {
            float inv = std::pow(static_cast<float>(cfg.rope_theta),
                                 -2.0f * static_cast<float>(i) / static_cast<float>(cfg.rotary_dim));
            float theta = static_cast<float>(t) * inv;
            cos_host[t * half + i] = f32_to_f16_bits(std::cos(theta));
            sin_host[t * half + i] = f32_to_f16_bits(std::sin(theta));
        }
    }
    cos_table = Tensor({T, half}, DType::Float16);
    sin_table = Tensor({T, half}, DType::Float16);
    cos_table.copy_from_host(cos_host.data(), cos_host.size() * sizeof(uint16_t));
    sin_table.copy_from_host(sin_host.data(), sin_host.size() * sizeof(uint16_t));
}

namespace {

void run_layer_step(int64_t layer,
                    const LanguageModelWeights& w,
                    const LanguageModelConfig& cfg,
                    const Tensor& cos_table,
                    const Tensor& sin_table,
                    int32_t pos,
                    int64_t cache_len,
                    DecodeState& state,
                    int& full_i,
                    int& linear_i,
                    const Tensor& hidden,
                    Tensor& next,
                    aclrtStream stream) {
    const auto& lw = w.layers[layer];
    if (cfg.layer_types[layer] == "linear_attention") {
        LinearAttentionDecoderLayerConfig lcfg{cfg.rms_epsilon};
        LinearAttentionDecoderLayerWeights ww{
            &lw.input_norm_w, &lw.post_norm_w, &lw.qkv_w, &lw.z_w, &lw.a_w,
            &lw.b_w, &lw.conv_w, &lw.dt_bias, &lw.a_log, &lw.gated_norm_w,
            &lw.out_proj_w, &lw.gate_w, &lw.up_w, &lw.down_w,
        };
        linear_attention_decoder_layer_step(hidden, ww, lcfg, state.linear[linear_i], next, stream);
        ++linear_i;
    } else {
        FullAttentionDecoderLayerConfig fcfg{cfg.num_q_heads, cfg.num_kv_heads,
                                             cfg.head_dim, cfg.rotary_dim, cfg.rms_epsilon};
        FullAttentionDecoderLayerWeights ww{
            &lw.input_norm_w, &lw.post_norm_w, &lw.q_w, &lw.k_w, &lw.v_w,
            &lw.o_w, &lw.q_norm_w, &lw.k_norm_w, &lw.gate_w, &lw.up_w, &lw.down_w,
        };
        full_attention_decoder_layer_step(hidden, ww, cos_table, sin_table,
                                          pos, cache_len, fcfg, state.full[full_i], next, stream);
        ++full_i;
    }
}

}  // namespace

Tensor prefill_from_embeddings(const Tensor& prompt_hidden,
                               const LanguageModelWeights& w,
                               const LanguageModelConfig& cfg,
                               const Tensor& cos_table,
                               const Tensor& sin_table,
                               DecodeState& state,
                               aclrtStream stream) {
    if (prompt_hidden.shape().size() != 2 || prompt_hidden.shape()[1] != cfg.hidden_size) {
        throw std::runtime_error("prefill_from_embeddings prompt_hidden must be [T, hidden_size]");
    }
    if (prompt_hidden.dtype() != DType::Float16) {
        throw std::runtime_error("prefill_from_embeddings prompt_hidden must be fp16");
    }
    const int64_t T = prompt_hidden.shape()[0];
    if (T <= 0) throw std::runtime_error("prefill_from_embeddings T must be > 0");
    if (state.seq_len != 0) {
        throw std::runtime_error("prefill_from_embeddings expects empty state");
    }
    if (T > state.max_seq_len) {
        throw std::runtime_error("prefill_from_embeddings T exceeds state.max_seq_len");
    }
    if (cos_table.shape()[0] < T) {
        throw std::runtime_error("prefill_from_embeddings cos_table too short");
    }

    Tensor hidden({T, cfg.hidden_size}, DType::Float16); hidden.allocate();
    Tensor next({T, cfg.hidden_size}, DType::Float16); next.allocate();
    copy_tensor(prompt_hidden, hidden, stream);

    std::vector<int32_t> row_to_t(static_cast<size_t>(T));
    for (int64_t t = 0; t < T; ++t) row_to_t[t] = static_cast<int32_t>(t);

    LinearAttentionDecoderLayerConfig lcfg{cfg.rms_epsilon};
    FullAttentionDecoderLayerConfig fcfg{cfg.num_q_heads, cfg.num_kv_heads,
                                         cfg.head_dim, cfg.rotary_dim, cfg.rms_epsilon};

    int full_i = 0;
    int linear_i = 0;
    for (int64_t layer = 0; layer < cfg.num_layers; ++layer) {
        const auto& lw = w.layers[layer];
        if (cfg.layer_types[layer] == "linear_attention") {
            LinearAttentionDecoderLayerWeights ww{
                &lw.input_norm_w, &lw.post_norm_w, &lw.qkv_w, &lw.z_w, &lw.a_w,
                &lw.b_w, &lw.conv_w, &lw.dt_bias, &lw.a_log, &lw.gated_norm_w,
                &lw.out_proj_w, &lw.gate_w, &lw.up_w, &lw.down_w,
            };
            linear_attention_decoder_layer_with_cache(hidden, ww, lcfg, state.linear[linear_i], next, stream);
            ++linear_i;
        } else {
            FullAttentionDecoderLayerWeights ww{
                &lw.input_norm_w, &lw.post_norm_w, &lw.q_w, &lw.k_w, &lw.v_w,
                &lw.o_w, &lw.q_norm_w, &lw.k_norm_w, &lw.gate_w, &lw.up_w, &lw.down_w,
            };
            full_attention_decoder_layer_with_cache(hidden, ww, cos_table, sin_table,
                                                    row_to_t, fcfg, state.full[full_i], next, stream);
            ++full_i;
        }
        copy_tensor(next, hidden, stream);
    }
    state.seq_len = T;

    Tensor last_hidden({1, cfg.hidden_size}, DType::Float16); last_hidden.allocate();
    copy_row(hidden, T - 1, last_hidden, 0, stream);
    return last_hidden;
}

int64_t lm_head_greedy(const Tensor& last_hidden_1xH,
                       const LanguageModelWeights& w,
                       const LanguageModelConfig& cfg,
                       aclrtStream stream) {
    if (last_hidden_1xH.shape() != std::vector<int64_t>{1, cfg.hidden_size}) {
        throw std::runtime_error("lm_head_greedy hidden must be [1, hidden_size]");
    }
    Tensor normed({1, cfg.hidden_size}, DType::Float16); normed.allocate();
    rms_norm(last_hidden_1xH, w.final_norm_w, normed, cfg.rms_epsilon, stream);
    Tensor logits({1, cfg.vocab_size}, DType::Float16); logits.allocate();
    matmul_b_transposed(normed, w.embed, logits, stream);
    Tensor pred({1}, DType::Int64); pred.allocate();
    argmax_last_dim(logits, pred, stream);
    int64_t out = 0;
    pred.copy_to_host(&out, sizeof(out));
    return out;
}

int64_t decode_step_greedy(int32_t token_id,
                           const LanguageModelWeights& w,
                           const LanguageModelConfig& cfg,
                           const Tensor& cos_table,
                           const Tensor& sin_table,
                           DecodeState& state,
                           aclrtStream stream) {
    if (state.seq_len >= state.max_seq_len) {
        throw std::runtime_error("decode_step_greedy state full");
    }
    Tensor hidden({1, cfg.hidden_size}, DType::Float16); hidden.allocate();
    Tensor next({1, cfg.hidden_size}, DType::Float16); next.allocate();
    embedding_lookup(w.embed, {token_id}, hidden, stream);

    int full_i = 0;
    int linear_i = 0;
    for (int64_t layer = 0; layer < cfg.num_layers; ++layer) {
        run_layer_step(layer, w, cfg, cos_table, sin_table,
                       static_cast<int32_t>(state.seq_len), state.seq_len,
                       state, full_i, linear_i, hidden, next, stream);
        copy_tensor(next, hidden, stream);
    }
    ++state.seq_len;
    return lm_head_greedy(hidden, w, cfg, stream);
}

bool is_eos(int64_t token_id, const std::vector<int64_t>& eos_token_ids) {
    return std::find(eos_token_ids.begin(), eos_token_ids.end(), token_id) != eos_token_ids.end();
}

}  // namespace minicpmv
