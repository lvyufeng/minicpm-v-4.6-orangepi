#include "minicpmv/language_model.h"

#include "minicpmv/acl_context.h"
#include "minicpmv/ops.h"
#include "minicpmv/quantized_weight.h"

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

// Load a 2D matmul weight stored as [N, K] in safetensors and return it as
// [K, N] (natural layout for our cube matmul fast path). Cube kernel beats
// aclnnMm 2-7x at M=1 for the shapes we hit at decode time; the host wrapper
// `matmul_b_transposed` falls back to aclnnMm for shapes the cube can't take.
// Keeps original layout for shapes the cube fast-path can't accelerate so
// memory and model-load time aren't wasted on a transpose that won't help.
Tensor load_matmul_weight_transposed(WeightsIndex& index, int layer, const std::string& suffix) {
    Tensor src = load_layer_weight(index, layer, suffix);
    if (src.shape().size() != 2) {
        throw std::runtime_error("matmul weight " + suffix + " expected 2D");
    }
    const int64_t N = src.shape()[0];
    const int64_t K = src.shape()[1];
    if (N > 16384 || (N % 128) != 0) {
        return src;  // cube fast-path doesn't apply; keep [N, K]
    }
    std::vector<uint16_t> hostNK(static_cast<size_t>(N) * K);
    src.copy_to_host(hostNK.data(), hostNK.size() * sizeof(uint16_t));
    std::vector<uint16_t> hostKN(static_cast<size_t>(K) * N);
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t k = 0; k < K; ++k) {
            hostKN[k * N + n] = hostNK[n * K + k];
        }
    }
    Tensor dst({K, N}, DType::Float16);
    dst.copy_from_host(hostKN.data(), hostKN.size() * sizeof(uint16_t));
    return dst;
}

std::string layer_base(int layer, const std::string& suffix) {
    return "model.language_model.layers." + std::to_string(layer) + "." + suffix;
}

W4A16QuantizedWeight load_layer_w4a16_if_present(WeightsIndex& index, int layer, const std::string& suffix) {
    const std::string base = layer_base(layer, suffix);
    if (!has_w4a16_quantized_weight(index, base)) {
        return {};
    }
    return load_w4a16_quantized_weight(index, base);
}

const W4A16QuantizedWeight* quant_ptr(const W4A16QuantizedWeight& w) {
    return w.w_int8.data() == nullptr ? nullptr : &w;
}

// Transpose a causal-conv weight from canonical [C, K=4] (or [C, 1, K]) layout
// into [K=4, C], so each of the 4 rows holds one tap's weights across all
// channels — what `linear_causal_conv_step` expects.
Tensor build_conv_step_weight(const Tensor& conv_w) {
    const auto& s = conv_w.shape();
    int64_t C = 0;
    int64_t K = 0;
    if (s.size() == 2) { C = s[0]; K = s[1]; }
    else if (s.size() == 3 && s[1] == 1) { C = s[0]; K = s[2]; }
    else { throw std::runtime_error("conv weight must be [C, 4] or [C, 1, 4]"); }
    if (K != 4) throw std::runtime_error("conv step weight expects K=4");

    std::vector<uint16_t> host_CK(C * K);
    conv_w.copy_to_host(host_CK.data(), host_CK.size() * sizeof(uint16_t));
    std::vector<uint16_t> host_KC(K * C);
    for (int64_t c = 0; c < C; ++c) {
        for (int64_t k = 0; k < K; ++k) {
            host_KC[k * C + c] = host_CK[c * K + k];
        }
    }
    Tensor dst({K, C}, DType::Float16);
    dst.copy_from_host(host_KC.data(), host_KC.size() * sizeof(uint16_t));
    return dst;
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
        lw.gate_w = load_matmul_weight_transposed(index, static_cast<int>(layer), "mlp.gate_proj.weight");
        lw.up_w   = load_matmul_weight_transposed(index, static_cast<int>(layer), "mlp.up_proj.weight");
        lw.down_w = load_matmul_weight_transposed(index, static_cast<int>(layer), "mlp.down_proj.weight");
        lw.gate_q = load_layer_w4a16_if_present(index, static_cast<int>(layer), "mlp.gate_proj");
        lw.up_q = load_layer_w4a16_if_present(index, static_cast<int>(layer), "mlp.up_proj");
        lw.down_q = load_layer_w4a16_if_present(index, static_cast<int>(layer), "mlp.down_proj");
        if (cfg.layer_types[layer] == "linear_attention") {
            lw.qkv_w = load_matmul_weight_transposed(index, static_cast<int>(layer), "linear_attn.in_proj_qkv.weight");
            lw.z_w   = load_matmul_weight_transposed(index, static_cast<int>(layer), "linear_attn.in_proj_z.weight");
            lw.qkv_q = load_layer_w4a16_if_present(index, static_cast<int>(layer), "linear_attn.in_proj_qkv");
            lw.z_q = load_layer_w4a16_if_present(index, static_cast<int>(layer), "linear_attn.in_proj_z");
            lw.a_w = load_layer_weight(index, static_cast<int>(layer), "linear_attn.in_proj_a.weight");
            lw.b_w = load_layer_weight(index, static_cast<int>(layer), "linear_attn.in_proj_b.weight");
            lw.conv_w = load_layer_weight(index, static_cast<int>(layer), "linear_attn.conv1d.weight");
            lw.dt_bias = load_layer_weight(index, static_cast<int>(layer), "linear_attn.dt_bias");
            lw.a_log = load_layer_weight(index, static_cast<int>(layer), "linear_attn.A_log");
            lw.gated_norm_w = load_layer_weight(index, static_cast<int>(layer), "linear_attn.norm.weight");
            lw.out_proj_w = load_matmul_weight_transposed(index, static_cast<int>(layer), "linear_attn.out_proj.weight");
            lw.out_proj_q = load_layer_w4a16_if_present(index, static_cast<int>(layer), "linear_attn.out_proj");
            lw.conv_w_step_t = build_conv_step_weight(lw.conv_w);
        } else {
            lw.q_w = load_matmul_weight_transposed(index, static_cast<int>(layer), "self_attn.q_proj.weight");
            lw.k_w = load_matmul_weight_transposed(index, static_cast<int>(layer), "self_attn.k_proj.weight");
            lw.v_w = load_matmul_weight_transposed(index, static_cast<int>(layer), "self_attn.v_proj.weight");
            lw.o_w = load_matmul_weight_transposed(index, static_cast<int>(layer), "self_attn.o_proj.weight");
            lw.q_q = load_layer_w4a16_if_present(index, static_cast<int>(layer), "self_attn.q_proj");
            lw.k_q = load_layer_w4a16_if_present(index, static_cast<int>(layer), "self_attn.k_proj");
            lw.v_q = load_layer_w4a16_if_present(index, static_cast<int>(layer), "self_attn.v_proj");
            lw.o_q = load_layer_w4a16_if_present(index, static_cast<int>(layer), "self_attn.o_proj");
            lw.q_norm_w = load_layer_weight(index, static_cast<int>(layer), "self_attn.q_norm.weight");
            lw.k_norm_w = load_layer_weight(index, static_cast<int>(layer), "self_attn.k_norm.weight");
        }
    }

    // Pre-build cube-friendly lm_head chunks: [K=hidden, N=16384] per slice,
    // padded with zero columns if the tail vocab piece is smaller. Cube path
    // requires N <= 16384 and N % 128 == 0, so a uniform 16384-wide chunk
    // keeps every slice on the fast path. Padded columns produce a 0 logit
    // that lm_head_greedy ignores by clipping to start_vocab + valid_n.
    constexpr int64_t kChunkN = 16384;
    const int64_t H = cfg.hidden_size;
    const int64_t V = cfg.vocab_size;
    if (w.embed.shape().size() != 2 || w.embed.shape()[0] != V || w.embed.shape()[1] != H) {
        throw std::runtime_error("embed weight shape unexpected for lm_head chunking");
    }
    std::vector<uint16_t> embed_host(static_cast<size_t>(V) * H);
    w.embed.copy_to_host(embed_host.data(), embed_host.size() * sizeof(uint16_t));
    std::vector<uint16_t> chunk_host(static_cast<size_t>(H) * kChunkN);
    for (int64_t start = 0; start < V; start += kChunkN) {
        const int64_t valid = std::min<int64_t>(kChunkN, V - start);
        std::fill(chunk_host.begin(), chunk_host.end(), uint16_t{0});
        for (int64_t n = 0; n < valid; ++n) {
            const int64_t row = start + n;
            for (int64_t k = 0; k < H; ++k) {
                chunk_host[static_cast<size_t>(k) * kChunkN + n] =
                    embed_host[static_cast<size_t>(row) * H + k];
            }
        }
        LmHeadChunk chunk;
        chunk.start_vocab = start;
        chunk.weight_kn = Tensor({H, kChunkN}, DType::Float16);
        chunk.weight_kn.copy_from_host(chunk_host.data(), chunk_host.size() * sizeof(uint16_t));
        w.lm_head_chunks.push_back(std::move(chunk));
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
            &lw.conv_w_step_t,
            quant_ptr(lw.qkv_q), quant_ptr(lw.z_q), quant_ptr(lw.out_proj_q),
            quant_ptr(lw.gate_q), quant_ptr(lw.up_q), quant_ptr(lw.down_q),
        };
        linear_attention_decoder_layer_step(hidden, ww, lcfg, state.linear[linear_i], next, stream);
        ++linear_i;
    } else {
        FullAttentionDecoderLayerConfig fcfg{cfg.num_q_heads, cfg.num_kv_heads,
                                             cfg.head_dim, cfg.rotary_dim, cfg.rms_epsilon};
        FullAttentionDecoderLayerWeights ww{
            &lw.input_norm_w, &lw.post_norm_w, &lw.q_w, &lw.k_w, &lw.v_w,
            &lw.o_w, &lw.q_norm_w, &lw.k_norm_w, &lw.gate_w, &lw.up_w, &lw.down_w,
            quant_ptr(lw.q_q), quant_ptr(lw.k_q), quant_ptr(lw.v_q), quant_ptr(lw.o_q),
            quant_ptr(lw.gate_q), quant_ptr(lw.up_q), quant_ptr(lw.down_q),
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
    if (w.lm_head_chunks.empty()) {
        throw std::runtime_error("lm_head_greedy missing pre-built chunks");
    }
    Tensor normed({1, cfg.hidden_size}, DType::Float16); normed.allocate();
    rms_norm(last_hidden_1xH, w.final_norm_w, normed, cfg.rms_epsilon, stream);

    // Reusable per-chunk logits buffer and host scratch for the D2H read.
    const int64_t kChunkN = w.lm_head_chunks.front().weight_kn.shape()[1];
    Tensor logits({1, kChunkN}, DType::Float16); logits.allocate();
    std::vector<uint16_t> logits_host(static_cast<size_t>(kChunkN));

    auto h2f = [](uint16_t h) -> float {
        uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
        uint32_t exp = (h >> 10) & 0x1fu;
        uint32_t mant = h & 0x03ffu;
        uint32_t out;
        if (exp == 0) {
            out = sign;
        } else if (exp == 31) {
            out = sign | 0x7f800000u | (mant << 13);
        } else {
            out = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
        float f;
        std::memcpy(&f, &out, sizeof(f));
        return f;
    };

    int64_t best_token = 0;
    float best_logit = -std::numeric_limits<float>::infinity();

    for (const auto& chunk : w.lm_head_chunks) {
        matmul_b_transposed(normed, chunk.weight_kn, logits, stream);
        logits.copy_to_host(logits_host.data(), logits_host.size() * sizeof(uint16_t));
        const int64_t valid = std::min<int64_t>(kChunkN, cfg.vocab_size - chunk.start_vocab);
        for (int64_t i = 0; i < valid; ++i) {
            float v = h2f(logits_host[static_cast<size_t>(i)]);
            if (v > best_logit) {
                best_logit = v;
                best_token = chunk.start_vocab + i;
            }
        }
    }
    return best_token;
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
