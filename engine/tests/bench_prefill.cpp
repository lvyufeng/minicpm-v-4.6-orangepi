// Compare batched prefill_from_embeddings against the old per-token loop that
// drove every layer's `_step` API once per prompt token. Same model weights,
// same prompt; report wall time and verify final next-token prediction matches.
#include "minicpmv/acl_context.h"
#include "minicpmv/decoder_layer.h"
#include "minicpmv/language_model.h"
#include "minicpmv/ops.h"
#include "minicpmv/tensor.h"
#include "minicpmv/weights.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace minicpmv;
using clk = std::chrono::steady_clock;

static double ms(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

namespace {

// Same shape as the now-deleted `run_layer_step` helper in language_model.cpp:
// runs one layer's `_step` for a single token using public APIs.
int64_t per_token_prefill(const Tensor& prompt_hidden,
                          const LanguageModelWeights& w,
                          const LanguageModelConfig& cfg,
                          const Tensor& cos_table,
                          const Tensor& sin_table,
                          DecodeState& state,
                          aclrtStream stream) {
    const int64_t T = prompt_hidden.shape()[0];
    const int64_t H = cfg.hidden_size;
    Tensor hidden({1, H}, DType::Float16); hidden.allocate();
    Tensor next({1, H}, DType::Float16); next.allocate();
    const size_t row_bytes = static_cast<size_t>(H) * dtype_size(prompt_hidden.dtype());

    LinearAttentionDecoderLayerConfig lcfg{cfg.rms_epsilon};
    FullAttentionDecoderLayerConfig fcfg{cfg.num_q_heads, cfg.num_kv_heads,
                                         cfg.head_dim, cfg.rotary_dim, cfg.rms_epsilon};

    for (int64_t t = 0; t < T; ++t) {
        auto* src = static_cast<const uint8_t*>(prompt_hidden.data()) + static_cast<size_t>(t) * row_bytes;
        check_acl(aclrtMemcpyAsync(hidden.data(), row_bytes, src, row_bytes,
                                   ACL_MEMCPY_DEVICE_TO_DEVICE, stream), "per_tok copy_row");
        check_acl(aclrtSynchronizeStream(stream), "per_tok copy_row sync");

        int full_i = 0, linear_i = 0;
        for (int64_t layer = 0; layer < cfg.num_layers; ++layer) {
            const auto& lw = w.layers[layer];
            if (cfg.layer_types[layer] == "linear_attention") {
                LinearAttentionDecoderLayerWeights ww{
                    &lw.input_norm_w, &lw.post_norm_w, &lw.qkv_w, &lw.z_w, &lw.a_w,
                    &lw.b_w, &lw.conv_w, &lw.dt_bias, &lw.a_log, &lw.gated_norm_w,
                    &lw.out_proj_w, &lw.gate_w, &lw.up_w, &lw.down_w,
                };
                linear_attention_decoder_layer_step(hidden, ww, lcfg, state.linear[linear_i], next, stream);
                ++linear_i;
            } else {
                FullAttentionDecoderLayerWeights ww{
                    &lw.input_norm_w, &lw.post_norm_w, &lw.q_w, &lw.k_w, &lw.v_w,
                    &lw.o_w, &lw.q_norm_w, &lw.k_norm_w, &lw.gate_w, &lw.up_w, &lw.down_w,
                };
                full_attention_decoder_layer_step(hidden, ww, cos_table, sin_table,
                                                  static_cast<int32_t>(t), state.seq_len,
                                                  fcfg, state.full[full_i], next, stream);
                ++full_i;
            }
            check_acl(aclrtMemcpyAsync(hidden.data(), hidden.size_bytes(), next.data(), next.size_bytes(),
                                       ACL_MEMCPY_DEVICE_TO_DEVICE, stream), "per_tok layer copy");
            check_acl(aclrtSynchronizeStream(stream), "per_tok layer copy sync");
        }
        ++state.seq_len;
    }
    return lm_head_greedy(hidden, w, cfg, stream);
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<int64_t> Ts;
    if (argc > 1) {
        for (int i = 1; i < argc; ++i) Ts.push_back(std::atoll(argv[i]));
    } else {
        Ts = {8, 32, 64, 128};
    }

    AclContext ctx(0);
    WeightsIndex index("/mnt/data/minicpm/MiniCPM-V-4.6/model.safetensors");
    LanguageModelConfig cfg = default_minicpmv46_lm_config();
    LanguageModelWeights w = load_language_model_weights(index, cfg);

    FullAttentionDecoderLayerConfig fcfg{cfg.num_q_heads, cfg.num_kv_heads,
                                         cfg.head_dim, cfg.rotary_dim, cfg.rms_epsilon};

    std::printf("%-6s %-12s %-12s %-8s %-10s %-10s\n",
                "T", "batched_ms", "per_token_ms", "speedup", "batch_tok", "ref_tok");

    for (int64_t T : Ts) {
        std::vector<int32_t> tokens(static_cast<size_t>(T));
        for (int64_t t = 0; t < T; ++t) tokens[t] = static_cast<int32_t>(1 + (t * 137) % 30000);
        Tensor prompt_hidden({T, cfg.hidden_size}, DType::Float16);
        prompt_hidden.allocate();
        embedding_lookup(w.embed, tokens, prompt_hidden, ctx.stream());

        Tensor cos_t, sin_t;
        build_rope_tables(T + 8, cfg, cos_t, sin_t);

        // Warmup batched (JIT/page-in).
        {
            DecodeState st = make_decode_state(T + 8, cfg.layer_types, fcfg, ctx.stream());
            Tensor lh = prefill_from_embeddings(prompt_hidden, w, cfg, cos_t, sin_t, st, ctx.stream());
            (void)lm_head_greedy(lh, w, cfg, ctx.stream());
        }

        const int iters = 2;
        double batched_ms = 0;
        int64_t batched_tok = 0;
        for (int i = 0; i < iters; ++i) {
            DecodeState st = make_decode_state(T + 8, cfg.layer_types, fcfg, ctx.stream());
            check_acl(aclrtSynchronizeStream(ctx.stream()), "pre-batched sync");
            auto t0 = clk::now();
            Tensor lh = prefill_from_embeddings(prompt_hidden, w, cfg, cos_t, sin_t, st, ctx.stream());
            check_acl(aclrtSynchronizeStream(ctx.stream()), "batched sync");
            auto t1 = clk::now();
            batched_ms += ms(t0, t1);
            batched_tok = lm_head_greedy(lh, w, cfg, ctx.stream());
        }
        batched_ms /= iters;

        // Warmup per-token.
        {
            DecodeState st = make_decode_state(T + 8, cfg.layer_types, fcfg, ctx.stream());
            (void)per_token_prefill(prompt_hidden, w, cfg, cos_t, sin_t, st, ctx.stream());
        }

        double per_tok_ms = 0;
        int64_t per_tok_tok = 0;
        for (int i = 0; i < iters; ++i) {
            DecodeState st = make_decode_state(T + 8, cfg.layer_types, fcfg, ctx.stream());
            check_acl(aclrtSynchronizeStream(ctx.stream()), "pre-pertok sync");
            auto t0 = clk::now();
            int64_t tok = per_token_prefill(prompt_hidden, w, cfg, cos_t, sin_t, st, ctx.stream());
            check_acl(aclrtSynchronizeStream(ctx.stream()), "pertok sync");
            auto t1 = clk::now();
            per_tok_ms += ms(t0, t1);
            per_tok_tok = tok;
        }
        per_tok_ms /= iters;

        std::printf("%-6lld %-12.1f %-12.1f %-8.2fx %-10lld %-10lld\n",
                    static_cast<long long>(T), batched_ms, per_tok_ms,
                    per_tok_ms / batched_ms,
                    static_cast<long long>(batched_tok),
                    static_cast<long long>(per_tok_tok));
    }
    return 0;
}
