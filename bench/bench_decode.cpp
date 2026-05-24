// Per-token decode latency from a primed DecodeState (prompt prefilled).
// Useful for understanding single-batch / interactive inference cost.
#include "minicpmv/acl_context.h"
#include "minicpmv/decoder_layer.h"
#include "minicpmv/language_model.h"
#include "minicpmv/ops.h"
#include "minicpmv/tensor.h"
#include "minicpmv/weights.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace minicpmv;
using clk = std::chrono::steady_clock;

static double ms(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

int main(int argc, char** argv) {
    const int prompt_T = (argc > 1) ? std::atoi(argv[1]) : 8;
    const int decode_N = (argc > 2) ? std::atoi(argv[2]) : 32;
    const int warmup   = 4;

    AclContext ctx(0);
    WeightsIndex index(default_safetensors_path());
    LanguageModelConfig cfg = default_minicpmv46_lm_config();
    LanguageModelWeights w = load_language_model_weights(index, cfg);

    std::vector<int32_t> tokens(prompt_T);
    for (int t = 0; t < prompt_T; ++t) tokens[t] = 1 + (t * 137) % 30000;
    Tensor prompt_h({prompt_T, cfg.hidden_size}, DType::Float16); prompt_h.allocate();
    embedding_lookup(w.embed, tokens, prompt_h, ctx.stream());

    Tensor cos_t, sin_t;
    build_rope_tables(prompt_T + decode_N + warmup + 8, cfg, cos_t, sin_t);

    FullAttentionDecoderLayerConfig fcfg{cfg.num_q_heads, cfg.num_kv_heads,
                                         cfg.head_dim, cfg.rotary_dim, cfg.rms_epsilon};
    DecodeState state = make_decode_state(prompt_T + decode_N + warmup + 8,
                                          cfg.layer_types, fcfg, ctx.stream());

    // Prefill.
    auto p0 = clk::now();
    Tensor last_h = prefill_from_embeddings(prompt_h, w, cfg, cos_t, sin_t, state, ctx.stream());
    int64_t cur_tok = lm_head_greedy(last_h, w, cfg, ctx.stream());
    auto p1 = clk::now();
    const double prefill_ms = ms(p0, p1);

    // Warm-up decode steps (excluded from timed region).
    for (int i = 0; i < warmup && state.seq_len < state.max_seq_len; ++i) {
        cur_tok = decode_step_greedy(static_cast<int32_t>(cur_tok), w, cfg,
                                     cos_t, sin_t, state, ctx.stream());
    }

    // Timed decode.
    auto d0 = clk::now();
    int decoded = 0;
    for (int i = 0; i < decode_N && state.seq_len < state.max_seq_len; ++i) {
        cur_tok = decode_step_greedy(static_cast<int32_t>(cur_tok), w, cfg,
                                     cos_t, sin_t, state, ctx.stream());
        ++decoded;
    }
    auto d1 = clk::now();
    const double decode_ms = ms(d0, d1);
    const double per_step_ms = decoded > 0 ? decode_ms / decoded : 0.0;
    const double tokens_per_s = per_step_ms > 0 ? 1000.0 / per_step_ms : 0.0;

    std::printf("prompt_T=%d  prefill_ms=%.1f\n", prompt_T, prefill_ms);
    std::printf("decode steps=%d  total_ms=%.1f  per_step_ms=%.2f  tokens/s=%.2f\n",
                decoded, decode_ms, per_step_ms, tokens_per_s);
    return 0;
}
