#include "minicpmv/acl_context.h"
#include "minicpmv/decoder_layer.h"
#include "minicpmv/ops.h"
#include "minicpmv/tensor.h"
#include "minicpmv/weights.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace minicpmv;

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

static Tensor load_layer_weight(WeightsIndex& index, int layer, const std::string& suffix) {
    return index.load_to_device_as("model.language_model.layers." + std::to_string(layer) + "." + suffix, DType::Float16);
}

static void copy_tensor(const Tensor& src, Tensor& dst, aclrtStream stream) {
    check_acl(aclrtMemcpyAsync(dst.data(), dst.size_bytes(), src.data(), src.size_bytes(),
                               ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
              "copy_tensor");
    check_acl(aclrtSynchronizeStream(stream), "copy_tensor sync");
}

static void copy_last_logits_row(const Tensor& logits, std::vector<uint16_t>& host, aclrtStream stream) {
    const int64_t vocab = logits.shape()[1];
    const int64_t last = logits.shape()[0] - 1;
    const size_t row_bytes = static_cast<size_t>(vocab) * dtype_size(logits.dtype());
    host.resize(static_cast<size_t>(vocab));
    auto* src = static_cast<const uint8_t*>(logits.data()) + static_cast<size_t>(last) * row_bytes;
    check_acl(aclrtMemcpyAsync(host.data(), row_bytes, src, row_bytes,
                               ACL_MEMCPY_DEVICE_TO_HOST, stream),
              "copy_last_logits_row");
    check_acl(aclrtSynchronizeStream(stream), "copy_last_logits_row sync");
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

int main(int argc, char** argv) {
    try {
        std::vector<int32_t> tokens;
        std::string mode = "prefill";
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--mode") {
                if (i + 1 >= argc) throw std::runtime_error("--mode requires a value");
                mode = argv[++i];
                if (mode != "prefill" && mode != "step") {
                    throw std::runtime_error("--mode must be prefill or step");
                }
                continue;
            }
            tokens.push_back(std::stoi(arg));
        }
        if (tokens.empty()) tokens = {1, 2, 10, 100};

        AclContext ctx(0);
        WeightsIndex index("/mnt/data/minicpm/MiniCPM-V-4.6/model.safetensors");

        constexpr int64_t Hidden = 1024;
        constexpr int64_t NumQHeads = 8;
        constexpr int64_t NumKVHeads = 2;
        constexpr int64_t HeadDim = 256;
        constexpr int64_t Rot = 64;
        constexpr int64_t HalfRot = Rot / 2;
        constexpr int64_t NumLayers = 24;
        constexpr int64_t TopK = 10;

        const std::vector<std::string> layer_types = {
            "linear_attention", "linear_attention", "linear_attention", "full_attention",
            "linear_attention", "linear_attention", "linear_attention", "full_attention",
            "linear_attention", "linear_attention", "linear_attention", "full_attention",
            "linear_attention", "linear_attention", "linear_attention", "full_attention",
            "linear_attention", "linear_attention", "linear_attention", "full_attention",
            "linear_attention", "linear_attention", "linear_attention", "full_attention",
        };

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

        const int64_t T = static_cast<int64_t>(tokens.size());

        LinearAttentionDecoderLayerConfig linear_config{1e-6};
        FullAttentionDecoderLayerConfig full_config{NumQHeads, NumKVHeads, HeadDim, Rot, 1e-6};

        Tensor last_hidden({1, Hidden}, DType::Float16);
        last_hidden.allocate();

        if (mode == "prefill") {
            Tensor hidden({T, Hidden}, DType::Float16);
            Tensor next({T, Hidden}, DType::Float16);
            hidden.allocate();
            next.allocate();
            embedding_lookup(embed, tokens, hidden, ctx.stream());

            std::vector<uint16_t> cos_host(T * HalfRot), sin_host(T * HalfRot);
            constexpr float RopeTheta = 10000000.0f;
            for (int64_t t = 0; t < T; ++t) {
                for (int64_t i = 0; i < HalfRot; ++i) {
                    float inv = std::pow(RopeTheta, -2.0f * static_cast<float>(i) / static_cast<float>(Rot));
                    float theta = static_cast<float>(t) * inv;
                    cos_host[t * HalfRot + i] = f2h(std::cos(theta));
                    sin_host[t * HalfRot + i] = f2h(std::sin(theta));
                }
            }
            Tensor cos_t({T, HalfRot}, DType::Float16);
            Tensor sin_t({T, HalfRot}, DType::Float16);
            cos_t.copy_from_host(cos_host.data(), cos_host.size() * sizeof(uint16_t));
            sin_t.copy_from_host(sin_host.data(), sin_host.size() * sizeof(uint16_t));

            std::vector<int32_t> row_to_t(T);
            for (int64_t t = 0; t < T; ++t) row_to_t[t] = static_cast<int32_t>(t);

            for (int layer = 0; layer < NumLayers; ++layer) {
                LayerWeights& lw = layers[layer];
                if (layer_types[layer] == "linear_attention") {
                    LinearAttentionDecoderLayerWeights w{
                        &lw.input_norm_w, &lw.post_norm_w, &lw.qkv_w, &lw.z_w, &lw.a_w,
                        &lw.b_w, &lw.conv_w, &lw.dt_bias, &lw.a_log, &lw.gated_norm_w,
                        &lw.out_proj_w, &lw.gate_w, &lw.up_w, &lw.down_w,
                    };
                    linear_attention_decoder_layer(hidden, w, linear_config, next, ctx.stream());
                } else {
                    FullAttentionDecoderLayerWeights w{
                        &lw.input_norm_w, &lw.post_norm_w, &lw.q_w, &lw.k_w, &lw.v_w,
                        &lw.o_w, &lw.q_norm_w, &lw.k_norm_w, &lw.gate_w, &lw.up_w,
                        &lw.down_w,
                    };
                    full_attention_decoder_layer(hidden, w, cos_t, sin_t, row_to_t, full_config, next, ctx.stream());
                }
                copy_tensor(next, hidden, ctx.stream());
            }

            const size_t row_bytes = static_cast<size_t>(Hidden) * dtype_size(hidden.dtype());
            auto* src = static_cast<const uint8_t*>(hidden.data()) + static_cast<size_t>(T - 1) * row_bytes;
            check_acl(aclrtMemcpyAsync(last_hidden.data(), row_bytes, src, row_bytes,
                                       ACL_MEMCPY_DEVICE_TO_DEVICE, ctx.stream()),
                      "copy last hidden prefill");
            check_acl(aclrtSynchronizeStream(ctx.stream()), "copy last hidden prefill sync");
        } else {
            DecodeState state = make_decode_state(static_cast<int64_t>(tokens.size()) + 4, layer_types, full_config, ctx.stream());

            const int64_t HalfRotLocal = HalfRot;
            Tensor cos_t({T, HalfRotLocal}, DType::Float16);
            Tensor sin_t({T, HalfRotLocal}, DType::Float16);
            std::vector<uint16_t> cos_host(T * HalfRotLocal), sin_host(T * HalfRotLocal);
            constexpr float RopeTheta = 10000000.0f;
            for (int64_t t = 0; t < T; ++t) {
                for (int64_t i = 0; i < HalfRotLocal; ++i) {
                    float inv = std::pow(RopeTheta, -2.0f * static_cast<float>(i) / static_cast<float>(Rot));
                    float theta = static_cast<float>(t) * inv;
                    cos_host[t * HalfRotLocal + i] = f2h(std::cos(theta));
                    sin_host[t * HalfRotLocal + i] = f2h(std::sin(theta));
                }
            }
            cos_t.copy_from_host(cos_host.data(), cos_host.size() * sizeof(uint16_t));
            sin_t.copy_from_host(sin_host.data(), sin_host.size() * sizeof(uint16_t));

            Tensor hidden({1, Hidden}, DType::Float16);
            Tensor next({1, Hidden}, DType::Float16);
            hidden.allocate();
            next.allocate();
            for (int64_t step = 0; step < T; ++step) {
                std::vector<int32_t> one = {tokens[step]};
                embedding_lookup(embed, one, hidden, ctx.stream());
                int full_i = 0;
                int linear_i = 0;
                for (int layer = 0; layer < NumLayers; ++layer) {
                    LayerWeights& lw = layers[layer];
                    if (layer_types[layer] == "linear_attention") {
                        LinearAttentionDecoderLayerWeights w{
                            &lw.input_norm_w, &lw.post_norm_w, &lw.qkv_w, &lw.z_w, &lw.a_w,
                            &lw.b_w, &lw.conv_w, &lw.dt_bias, &lw.a_log, &lw.gated_norm_w,
                            &lw.out_proj_w, &lw.gate_w, &lw.up_w, &lw.down_w,
                        };
                        linear_attention_decoder_layer_step(hidden, w, linear_config, state.linear[linear_i], next, ctx.stream());
                        ++linear_i;
                    } else {
                        FullAttentionDecoderLayerWeights w{
                            &lw.input_norm_w, &lw.post_norm_w, &lw.q_w, &lw.k_w, &lw.v_w,
                            &lw.o_w, &lw.q_norm_w, &lw.k_norm_w, &lw.gate_w, &lw.up_w,
                            &lw.down_w,
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
            copy_tensor(hidden, last_hidden, ctx.stream());
        }

        Tensor normed({1, Hidden}, DType::Float16);
        normed.allocate();
        rms_norm(last_hidden, final_norm_w, normed, 1e-6, ctx.stream());
        Tensor logits({1, vocab}, DType::Float16);
        logits.allocate();
        matmul_b_transposed(normed, embed, logits, ctx.stream());

        std::vector<uint16_t> last_logits_h;
        copy_last_logits_row(logits, last_logits_h, ctx.stream());
        std::vector<std::pair<float, int64_t>> scored;
        scored.reserve(last_logits_h.size());
        for (int64_t i = 0; i < static_cast<int64_t>(last_logits_h.size()); ++i) {
            scored.emplace_back(h2f(last_logits_h[static_cast<size_t>(i)]), i);
        }
        std::partial_sort(scored.begin(), scored.begin() + TopK, scored.end(),
                          [](const auto& a, const auto& b) {
                              if (a.first == b.first) return a.second < b.second;
                              return a.first > b.first;
                          });

        std::cout << "{\"mode\":\"" << mode << "\",\"tokens\":[";
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (i) std::cout << ",";
            std::cout << tokens[i];
        }
        std::cout << "],\"next_token\":" << scored[0].second << ",\"topk_ids\":[";
        for (int64_t i = 0; i < TopK; ++i) {
            if (i) std::cout << ",";
            std::cout << scored[static_cast<size_t>(i)].second;
        }
        std::cout << "],\"topk_logits\":[";
        for (int64_t i = 0; i < TopK; ++i) {
            if (i) std::cout << ",";
            std::cout << scored[static_cast<size_t>(i)].first;
        }
        std::cout << "]}" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "test_language_logits_dump failed: " << e.what() << std::endl;
        return 1;
    }
}
