#include "minicpmv/acl_context.h"
#include "minicpmv/decoder_layer.h"
#include "minicpmv/language_model.h"
#include "minicpmv/ops.h"
#include "minicpmv/tensor.h"
#include "minicpmv/weights.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace minicpmv;

static float h2f(uint16_t h) {
    uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) { out = sign; }
        else {
            int32_t e = 1;
            while ((mant & 0x400u) == 0) { mant <<= 1; e--; }
            mant &= 0x3ffu;
            out = sign | (static_cast<uint32_t>(e + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7f800000u | (mant << 13);
    } else {
        out = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}

static float max_abs_diff(const Tensor& a, const Tensor& b) {
    std::vector<uint16_t> ha(a.numel()), hb(b.numel());
    a.copy_to_host(ha.data(), ha.size() * sizeof(uint16_t));
    b.copy_to_host(hb.data(), hb.size() * sizeof(uint16_t));
    float m = 0.0f;
    for (size_t i = 0; i < ha.size(); ++i) {
        float d = std::fabs(h2f(ha[i]) - h2f(hb[i]));
        if (d > m) m = d;
    }
    return m;
}

int main() {
    AclContext ctx(0);
    WeightsIndex index("/mnt/data/minicpm/MiniCPM-V-4.6/model.safetensors");

    LanguageModelConfig cfg = default_minicpmv46_lm_config();
    LanguageModelWeights w = load_language_model_weights(index, cfg);

    const std::vector<int32_t> tokens = {1, 2, 10, 100};
    const int64_t T = static_cast<int64_t>(tokens.size());

    Tensor prompt_hidden({T, cfg.hidden_size}, DType::Float16);
    prompt_hidden.allocate();
    embedding_lookup(w.embed, tokens, prompt_hidden, ctx.stream());

    Tensor cos_t, sin_t;
    build_rope_tables(T + 4, cfg, cos_t, sin_t);

    DecodeState state = make_decode_state(T + 4,
                                          cfg.layer_types,
                                          FullAttentionDecoderLayerConfig{cfg.num_q_heads, cfg.num_kv_heads,
                                                                          cfg.head_dim, cfg.rotary_dim, cfg.rms_epsilon},
                                          ctx.stream());

    Tensor last_hidden = prefill_from_embeddings(prompt_hidden, w, cfg, cos_t, sin_t, state, ctx.stream());
    int64_t next_tok = lm_head_greedy(last_hidden, w, cfg, ctx.stream());

    // Reference: run via existing decoder_stack-style full-rerun path on the
    // same prompt to capture the last-row hidden.
    Tensor hidden({T, cfg.hidden_size}, DType::Float16);
    Tensor next({T, cfg.hidden_size}, DType::Float16);
    hidden.allocate();
    next.allocate();
    embedding_lookup(w.embed, tokens, hidden, ctx.stream());

    Tensor cos_full({T, cfg.rotary_dim / 2}, DType::Float16);
    Tensor sin_full({T, cfg.rotary_dim / 2}, DType::Float16);
    {
        std::vector<uint16_t> cos_host(T * cfg.rotary_dim / 2);
        std::vector<uint16_t> sin_host(T * cfg.rotary_dim / 2);
        for (int64_t t = 0; t < T; ++t) {
            for (int64_t i = 0; i < cfg.rotary_dim / 2; ++i) {
                float inv = std::pow(static_cast<float>(cfg.rope_theta),
                                     -2.0f * static_cast<float>(i) / static_cast<float>(cfg.rotary_dim));
                float theta = static_cast<float>(t) * inv;
                uint32_t bits;
                {
                    float c = std::cos(theta);
                    uint32_t x;
                    std::memcpy(&x, &c, sizeof(x));
                    uint32_t sign = (x >> 16) & 0x8000u;
                    int32_t e = static_cast<int32_t>((x >> 23) & 0xff) - 127 + 15;
                    uint32_t mant = x & 0x7fffffu;
                    if (e <= 0) bits = sign;
                    else if (e >= 31) bits = sign | 0x7c00u;
                    else bits = sign | (static_cast<uint32_t>(e) << 10) | (mant >> 13);
                    cos_host[t * cfg.rotary_dim / 2 + i] = static_cast<uint16_t>(bits);
                }
                {
                    float s = std::sin(theta);
                    uint32_t x;
                    std::memcpy(&x, &s, sizeof(x));
                    uint32_t sign = (x >> 16) & 0x8000u;
                    int32_t e = static_cast<int32_t>((x >> 23) & 0xff) - 127 + 15;
                    uint32_t mant = x & 0x7fffffu;
                    if (e <= 0) bits = sign;
                    else if (e >= 31) bits = sign | 0x7c00u;
                    else bits = sign | (static_cast<uint32_t>(e) << 10) | (mant >> 13);
                    sin_host[t * cfg.rotary_dim / 2 + i] = static_cast<uint16_t>(bits);
                }
            }
        }
        cos_full.copy_from_host(cos_host.data(), cos_host.size() * sizeof(uint16_t));
        sin_full.copy_from_host(sin_host.data(), sin_host.size() * sizeof(uint16_t));
    }
    std::vector<int32_t> row_to_t(T);
    for (int64_t t = 0; t < T; ++t) row_to_t[t] = static_cast<int32_t>(t);

    LinearAttentionDecoderLayerConfig lcfg{cfg.rms_epsilon};
    FullAttentionDecoderLayerConfig fcfg{cfg.num_q_heads, cfg.num_kv_heads,
                                         cfg.head_dim, cfg.rotary_dim, cfg.rms_epsilon};
    for (int64_t layer = 0; layer < cfg.num_layers; ++layer) {
        const auto& lw = w.layers[layer];
        if (cfg.layer_types[layer] == "linear_attention") {
            LinearAttentionDecoderLayerWeights ww{
                &lw.input_norm_w, &lw.post_norm_w, &lw.qkv_w, &lw.z_w, &lw.a_w,
                &lw.b_w, &lw.conv_w, &lw.dt_bias, &lw.a_log, &lw.gated_norm_w,
                &lw.out_proj_w, &lw.gate_w, &lw.up_w, &lw.down_w,
            };
            linear_attention_decoder_layer(hidden, ww, lcfg, next, ctx.stream());
        } else {
            FullAttentionDecoderLayerWeights ww{
                &lw.input_norm_w, &lw.post_norm_w, &lw.q_w, &lw.k_w, &lw.v_w,
                &lw.o_w, &lw.q_norm_w, &lw.k_norm_w, &lw.gate_w, &lw.up_w, &lw.down_w,
            };
            full_attention_decoder_layer(hidden, ww, cos_full, sin_full, row_to_t, fcfg, next, ctx.stream());
        }
        check_acl(aclrtMemcpyAsync(hidden.data(), hidden.size_bytes(), next.data(), next.size_bytes(),
                                   ACL_MEMCPY_DEVICE_TO_DEVICE, ctx.stream()),
                  "ref copy");
        check_acl(aclrtSynchronizeStream(ctx.stream()), "ref copy sync");
    }

    Tensor ref_last({1, cfg.hidden_size}, DType::Float16); ref_last.allocate();
    {
        const size_t row_bytes = static_cast<size_t>(cfg.hidden_size) * dtype_size(hidden.dtype());
        auto* src = static_cast<const uint8_t*>(hidden.data()) + static_cast<size_t>(T - 1) * row_bytes;
        check_acl(aclrtMemcpyAsync(ref_last.data(), row_bytes, src, row_bytes,
                                   ACL_MEMCPY_DEVICE_TO_DEVICE, ctx.stream()),
                  "ref last");
        check_acl(aclrtSynchronizeStream(ctx.stream()), "ref last sync");
    }
    int64_t ref_tok = lm_head_greedy(ref_last, w, cfg, ctx.stream());

    float diff = max_abs_diff(last_hidden, ref_last);
    if (next_tok != ref_tok) {
        std::cerr << "prefill_from_embeddings next-token mismatch: "
                  << next_tok << " vs ref " << ref_tok
                  << " hidden_max_abs=" << diff << std::endl;
        return 1;
    }
    if (!std::isfinite(diff) || diff > 1.0f) {
        std::cerr << "prefill_from_embeddings hidden diff out of bounds: " << diff << std::endl;
        return 1;
    }

    std::cout << "prefill_from_embeddings smoke ok"
              << " T=" << T
              << " next_token=" << next_tok
              << " ref_token=" << ref_tok
              << " hidden_max_abs=" << diff
              << " state_seq_len=" << state.seq_len << std::endl;
    return 0;
}
