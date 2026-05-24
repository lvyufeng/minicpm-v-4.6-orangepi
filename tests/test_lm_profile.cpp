#include "minicpmv/acl_context.h"
#include "minicpmv/decoder_layer.h"
#include "minicpmv/ops.h"
#include "minicpmv/tensor.h"
#include "minicpmv/weights.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace minicpmv;

using clk = std::chrono::steady_clock;

static double ms_between(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

static uint16_t f2h(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;
    if (exp <= 0) return static_cast<uint16_t>(sign);
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}

static Tensor load_layer_weight(WeightsIndex& index, int layer, const std::string& suffix) {
    return index.load_to_device_as("model.language_model.layers." + std::to_string(layer) + "." + suffix, DType::Float16);
}

static void copy_tensor(const Tensor& src, Tensor& dst, aclrtStream stream) {
    check_acl(aclrtMemcpyAsync(dst.data(), dst.size_bytes(), src.data(), src.size_bytes(),
                               ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
              "copy_tensor");
    check_acl(aclrtSynchronizeStream(stream), "copy_tensor sync");
}

struct LayerWeights {
    Tensor input_norm_w;
    Tensor post_norm_w;
    Tensor gate_w;
    Tensor up_w;
    Tensor down_w;
    Tensor qkv_w;
    Tensor z_w;
    Tensor a_w;
    Tensor b_w;
    Tensor conv_w;
    Tensor dt_bias;
    Tensor a_log;
    Tensor gated_norm_w;
    Tensor out_proj_w;
    Tensor q_w;
    Tensor k_w;
    Tensor v_w;
    Tensor o_w;
    Tensor q_norm_w;
    Tensor k_norm_w;
};

int main() {
    AclContext ctx(0);
    WeightsIndex index(default_safetensors_path());

    constexpr int64_t Hidden = 1024;
    constexpr int64_t NumQHeads = 8;
    constexpr int64_t NumKVHeads = 2;
    constexpr int64_t HeadDim = 256;
    constexpr int64_t Rot = 64;
    constexpr int64_t HalfRot = Rot / 2;
    constexpr int64_t NumLayers = 24;
    constexpr int64_t NumGenerate = 8;

    const std::vector<std::string> layer_types = {
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
    };

    auto t_load_start = clk::now();
    Tensor embed = index.load_to_device_as("model.language_model.embed_tokens.weight", DType::Float16);
    Tensor final_norm_w = index.load_to_device_as("model.language_model.norm.weight", DType::Float16);
    const int64_t vocab = embed.shape()[0];

    std::vector<LayerWeights> layers(NumLayers);
    for (int layer = 0; layer < NumLayers; ++layer) {
        const std::string& ltype = layer_types[layer];
        layers[layer].input_norm_w = load_layer_weight(index, layer, "input_layernorm.weight");
        layers[layer].post_norm_w = load_layer_weight(index, layer, "post_attention_layernorm.weight");
        layers[layer].gate_w = load_layer_weight(index, layer, "mlp.gate_proj.weight");
        layers[layer].up_w = load_layer_weight(index, layer, "mlp.up_proj.weight");
        layers[layer].down_w = load_layer_weight(index, layer, "mlp.down_proj.weight");
        if (ltype == "linear_attention") {
            layers[layer].qkv_w = load_layer_weight(index, layer, "linear_attn.in_proj_qkv.weight");
            layers[layer].z_w = load_layer_weight(index, layer, "linear_attn.in_proj_z.weight");
            layers[layer].a_w = load_layer_weight(index, layer, "linear_attn.in_proj_a.weight");
            layers[layer].b_w = load_layer_weight(index, layer, "linear_attn.in_proj_b.weight");
            layers[layer].conv_w = load_layer_weight(index, layer, "linear_attn.conv1d.weight");
            layers[layer].dt_bias = load_layer_weight(index, layer, "linear_attn.dt_bias");
            layers[layer].a_log = load_layer_weight(index, layer, "linear_attn.A_log");
            layers[layer].gated_norm_w = load_layer_weight(index, layer, "linear_attn.norm.weight");
            layers[layer].out_proj_w = load_layer_weight(index, layer, "linear_attn.out_proj.weight");
        } else {
            layers[layer].q_w = load_layer_weight(index, layer, "self_attn.q_proj.weight");
            layers[layer].k_w = load_layer_weight(index, layer, "self_attn.k_proj.weight");
            layers[layer].v_w = load_layer_weight(index, layer, "self_attn.v_proj.weight");
            layers[layer].o_w = load_layer_weight(index, layer, "self_attn.o_proj.weight");
            layers[layer].q_norm_w = load_layer_weight(index, layer, "self_attn.q_norm.weight");
            layers[layer].k_norm_w = load_layer_weight(index, layer, "self_attn.k_norm.weight");
        }
    }
    auto t_load_end = clk::now();
    std::cout << "weight_load_ms=" << ms_between(t_load_start, t_load_end) << std::endl;

    LinearAttentionDecoderLayerConfig linear_config{1e-6};
    FullAttentionDecoderLayerConfig full_config{NumQHeads, NumKVHeads, HeadDim, Rot, 1e-6};
    constexpr float RopeTheta = 10000000.0f;

    auto build_rope = [&](int64_t T, std::vector<uint16_t>& cos_host, std::vector<uint16_t>& sin_host) {
        cos_host.assign(T * HalfRot, 0);
        sin_host.assign(T * HalfRot, 0);
        for (int64_t t = 0; t < T; ++t) {
            for (int64_t i = 0; i < HalfRot; ++i) {
                float inv = std::pow(RopeTheta, -2.0f * static_cast<float>(i) / static_cast<float>(Rot));
                float theta = static_cast<float>(t) * inv;
                cos_host[t * HalfRot + i] = f2h(std::cos(theta));
                sin_host[t * HalfRot + i] = f2h(std::sin(theta));
            }
        }
    };

    auto run_head = [&](const Tensor& hidden_last, double& head_ms) -> int32_t {
        auto th0 = clk::now();
        Tensor normed({1, Hidden}, DType::Float16); normed.allocate();
        rms_norm(hidden_last, final_norm_w, normed, 1e-6, ctx.stream());
        Tensor logits({1, vocab}, DType::Float16); logits.allocate();
        matmul_b_transposed(normed, embed, logits, ctx.stream());
        Tensor pred({1}, DType::Int64); pred.allocate();
        argmax_last_dim(logits, pred, ctx.stream());
        check_acl(aclrtSynchronizeStream(ctx.stream()), "head sync");
        auto th1 = clk::now();
        head_ms = ms_between(th0, th1);
        int64_t host_pred = 0;
        pred.copy_to_host(&host_pred, sizeof(host_pred));
        return static_cast<int32_t>(host_pred);
    };

    auto run_prefill = [&](const std::vector<int32_t>& tokens,
                           std::vector<double>& per_layer_ms,
                           double& head_ms,
                           Tensor& last_hidden_out) -> int32_t {
        const int64_t T = static_cast<int64_t>(tokens.size());
        Tensor hidden({T, Hidden}, DType::Float16);
        Tensor next({T, Hidden}, DType::Float16);
        hidden.allocate();
        next.allocate();
        embedding_lookup(embed, tokens, hidden, ctx.stream());

        std::vector<uint16_t> cos_host, sin_host;
        build_rope(T, cos_host, sin_host);
        Tensor cos_t({T, HalfRot}, DType::Float16);
        Tensor sin_t({T, HalfRot}, DType::Float16);
        cos_t.copy_from_host(cos_host.data(), cos_host.size() * sizeof(uint16_t));
        sin_t.copy_from_host(sin_host.data(), sin_host.size() * sizeof(uint16_t));

        std::vector<int32_t> row_to_t(T);
        for (int64_t t = 0; t < T; ++t) row_to_t[t] = static_cast<int32_t>(t);

        per_layer_ms.assign(NumLayers, 0.0);
        for (int layer = 0; layer < NumLayers; ++layer) {
            const std::string& ltype = layer_types[layer];
            auto t0 = clk::now();
            if (ltype == "linear_attention") {
                LinearAttentionDecoderLayerWeights w{
                    &layers[layer].input_norm_w,
                    &layers[layer].post_norm_w,
                    &layers[layer].qkv_w,
                    &layers[layer].z_w,
                    &layers[layer].a_w,
                    &layers[layer].b_w,
                    &layers[layer].conv_w,
                    &layers[layer].dt_bias,
                    &layers[layer].a_log,
                    &layers[layer].gated_norm_w,
                    &layers[layer].out_proj_w,
                    &layers[layer].gate_w,
                    &layers[layer].up_w,
                    &layers[layer].down_w,
                };
                linear_attention_decoder_layer(hidden, w, linear_config, next, ctx.stream());
            } else {
                FullAttentionDecoderLayerWeights w{
                    &layers[layer].input_norm_w,
                    &layers[layer].post_norm_w,
                    &layers[layer].q_w,
                    &layers[layer].k_w,
                    &layers[layer].v_w,
                    &layers[layer].o_w,
                    &layers[layer].q_norm_w,
                    &layers[layer].k_norm_w,
                    &layers[layer].gate_w,
                    &layers[layer].up_w,
                    &layers[layer].down_w,
                };
                full_attention_decoder_layer(hidden, w, cos_t, sin_t, row_to_t, full_config, next, ctx.stream());
            }
            copy_tensor(next, hidden, ctx.stream());
            check_acl(aclrtSynchronizeStream(ctx.stream()), "profile layer sync");
            auto t1 = clk::now();
            per_layer_ms[layer] = ms_between(t0, t1);
        }

        const size_t row_bytes = static_cast<size_t>(Hidden) * dtype_size(hidden.dtype());
        auto* src = static_cast<const uint8_t*>(hidden.data()) + static_cast<size_t>(T - 1) * row_bytes;
        check_acl(aclrtMemcpyAsync(last_hidden_out.data(), row_bytes, src, row_bytes,
                                   ACL_MEMCPY_DEVICE_TO_DEVICE, ctx.stream()),
                  "copy last hidden");
        check_acl(aclrtSynchronizeStream(ctx.stream()), "copy last hidden sync");
        return run_head(last_hidden_out, head_ms);
    };

    auto run_step_decode = [&](const std::vector<int32_t>& prompt_tokens,
                               int32_t next_token,
                               int64_t steps,
                               double& avg_step_ms,
                               double& avg_head_ms,
                               int32_t& last_tok) {
        DecodeState state = make_decode_state(static_cast<int64_t>(prompt_tokens.size()) + steps + 4,
                                              layer_types, full_config, ctx.stream());
        Tensor hidden_last({1, Hidden}, DType::Float16); hidden_last.allocate();

        // warm cache with prompt via real step path
        std::vector<uint16_t> cos_host, sin_host;
        build_rope(static_cast<int64_t>(prompt_tokens.size() + steps + 4), cos_host, sin_host);
        Tensor cos_t({static_cast<int64_t>(prompt_tokens.size() + steps + 4), HalfRot}, DType::Float16);
        Tensor sin_t({static_cast<int64_t>(prompt_tokens.size() + steps + 4), HalfRot}, DType::Float16);
        cos_t.copy_from_host(cos_host.data(), cos_host.size() * sizeof(uint16_t));
        sin_t.copy_from_host(sin_host.data(), sin_host.size() * sizeof(uint16_t));

        Tensor hidden({1, Hidden}, DType::Float16); hidden.allocate();
        Tensor next({1, Hidden}, DType::Float16); next.allocate();
        for (size_t step = 0; step < prompt_tokens.size(); ++step) {
            embedding_lookup(embed, {prompt_tokens[step]}, hidden, ctx.stream());
            int full_i = 0;
            int linear_i = 0;
            for (int layer = 0; layer < NumLayers; ++layer) {
                const std::string& ltype = layer_types[layer];
                if (ltype == "linear_attention") {
                    LinearAttentionDecoderLayerWeights w{
                        &layers[layer].input_norm_w,
                        &layers[layer].post_norm_w,
                        &layers[layer].qkv_w,
                        &layers[layer].z_w,
                        &layers[layer].a_w,
                        &layers[layer].b_w,
                        &layers[layer].conv_w,
                        &layers[layer].dt_bias,
                        &layers[layer].a_log,
                        &layers[layer].gated_norm_w,
                        &layers[layer].out_proj_w,
                        &layers[layer].gate_w,
                        &layers[layer].up_w,
                        &layers[layer].down_w,
                    };
                    linear_attention_decoder_layer_step(hidden, w, linear_config, state.linear[linear_i], next, ctx.stream());
                    ++linear_i;
                } else {
                    FullAttentionDecoderLayerWeights w{
                        &layers[layer].input_norm_w,
                        &layers[layer].post_norm_w,
                        &layers[layer].q_w,
                        &layers[layer].k_w,
                        &layers[layer].v_w,
                        &layers[layer].o_w,
                        &layers[layer].q_norm_w,
                        &layers[layer].k_norm_w,
                        &layers[layer].gate_w,
                        &layers[layer].up_w,
                        &layers[layer].down_w,
                    };
                    full_attention_decoder_layer_step(hidden, w, cos_t, sin_t,
                                                      static_cast<int32_t>(step), state.seq_len,
                                                      full_config, state.full[full_i], next, ctx.stream());
                    ++full_i;
                }
                copy_tensor(next, hidden, ctx.stream());
            }
            ++state.seq_len;
        }

        double total_step_ms = 0.0;
        double total_head_ms = 0.0;
        last_tok = next_token;
        for (int64_t step = 0; step < steps; ++step) {
            auto t0 = clk::now();
            embedding_lookup(embed, {last_tok}, hidden, ctx.stream());
            int full_i = 0;
            int linear_i = 0;
            for (int layer = 0; layer < NumLayers; ++layer) {
                const std::string& ltype = layer_types[layer];
                if (ltype == "linear_attention") {
                    LinearAttentionDecoderLayerWeights w{
                        &layers[layer].input_norm_w,
                        &layers[layer].post_norm_w,
                        &layers[layer].qkv_w,
                        &layers[layer].z_w,
                        &layers[layer].a_w,
                        &layers[layer].b_w,
                        &layers[layer].conv_w,
                        &layers[layer].dt_bias,
                        &layers[layer].a_log,
                        &layers[layer].gated_norm_w,
                        &layers[layer].out_proj_w,
                        &layers[layer].gate_w,
                        &layers[layer].up_w,
                        &layers[layer].down_w,
                    };
                    linear_attention_decoder_layer_step(hidden, w, linear_config, state.linear[linear_i], next, ctx.stream());
                    ++linear_i;
                } else {
                    FullAttentionDecoderLayerWeights w{
                        &layers[layer].input_norm_w,
                        &layers[layer].post_norm_w,
                        &layers[layer].q_w,
                        &layers[layer].k_w,
                        &layers[layer].v_w,
                        &layers[layer].o_w,
                        &layers[layer].q_norm_w,
                        &layers[layer].k_norm_w,
                        &layers[layer].gate_w,
                        &layers[layer].up_w,
                        &layers[layer].down_w,
                    };
                    full_attention_decoder_layer_step(hidden, w, cos_t, sin_t,
                                                      static_cast<int32_t>(state.seq_len), state.seq_len,
                                                      full_config, state.full[full_i], next, ctx.stream());
                    ++full_i;
                }
                copy_tensor(next, hidden, ctx.stream());
            }
            ++state.seq_len;
            double head_ms = 0.0;
            last_tok = run_head(hidden, head_ms);
            auto t1 = clk::now();
            total_step_ms += ms_between(t0, t1);
            total_head_ms += head_ms;
        }
        avg_step_ms = total_step_ms / static_cast<double>(steps);
        avg_head_ms = total_head_ms / static_cast<double>(steps);
    };

    std::vector<int32_t> tokens = {1, 2, 10, 100};
    std::vector<double> per_layer_ms;
    double head_ms = 0.0;
    Tensor last_hidden({1, Hidden}, DType::Float16); last_hidden.allocate();

    auto t_prefill_start = clk::now();
    int32_t first_tok = run_prefill(tokens, per_layer_ms, head_ms, last_hidden);
    auto t_prefill_end = clk::now();
    double total_ms = ms_between(t_prefill_start, t_prefill_end);

    double linear_total = 0.0;
    double full_total = 0.0;
    int linear_count = 0;
    int full_count = 0;
    for (int layer = 0; layer < NumLayers; ++layer) {
        if (layer_types[layer] == "linear_attention") {
            linear_total += per_layer_ms[layer];
            ++linear_count;
        } else {
            full_total += per_layer_ms[layer];
            ++full_count;
        }
    }
    std::cout << "prefill_ms=" << total_ms
              << " head_ms=" << head_ms
              << " linear_layer_avg_ms=" << (linear_total / linear_count)
              << " full_layer_avg_ms=" << (full_total / full_count)
              << " first_tok=" << first_tok << std::endl;
    for (int layer = 0; layer < NumLayers; ++layer) {
        std::cout << "  layer " << layer << " " << layer_types[layer]
                  << " ms=" << per_layer_ms[layer] << std::endl;
    }

    double avg_step_ms = 0.0;
    double avg_step_head_ms = 0.0;
    int32_t last_tok = first_tok;
    run_step_decode(tokens, first_tok, NumGenerate - 1, avg_step_ms, avg_step_head_ms, last_tok);
    std::cout << "decode_step_avg_ms=" << avg_step_ms
              << " head_avg_ms=" << avg_step_head_ms
              << " steps=" << (NumGenerate - 1)
              << " last_tok=" << last_tok << std::endl;
    std::cout << "lm profile smoke ok generated_seq=";
    std::vector<int32_t> shown = tokens;
    shown.push_back(first_tok);
    for (auto t : shown) std::cout << t << ",";
    std::cout << std::endl;
    return 0;
}
