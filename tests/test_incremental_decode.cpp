#include "minicpmv/acl_context.h"
#include "minicpmv/decoder_layer.h"
#include "minicpmv/ops.h"
#include "minicpmv/tensor.h"
#include "minicpmv/weights.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
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

static void copy_last_row(const Tensor& src, Tensor& dst, aclrtStream stream) {
    const int64_t last = src.shape()[0] - 1;
    const size_t row_bytes = static_cast<size_t>(src.shape()[1]) * dtype_size(src.dtype());
    auto* s = static_cast<const uint8_t*>(src.data()) + static_cast<size_t>(last) * row_bytes;
    check_acl(aclrtMemcpyAsync(dst.data(), row_bytes, s, row_bytes,
                               ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
              "copy_last_row");
    check_acl(aclrtSynchronizeStream(stream), "copy_last_row sync");
}

static float max_abs_diff(const Tensor& a, const Tensor& b) {
    std::vector<uint16_t> ha(a.numel()), hb(b.numel());
    a.copy_to_host(ha.data(), ha.size() * sizeof(uint16_t));
    b.copy_to_host(hb.data(), hb.size() * sizeof(uint16_t));
    float max_abs = 0.0f;
    for (size_t i = 0; i < ha.size(); ++i) {
        float d = std::fabs(h2f(ha[i]) - h2f(hb[i]));
        if (d > max_abs) max_abs = d;
    }
    return max_abs;
}

static int64_t argmax_token(const Tensor& hidden,
                           const Tensor& final_norm_w,
                           const Tensor& embed,
                           aclrtStream stream) {
    Tensor normed({1, hidden.shape()[1]}, DType::Float16);
    normed.allocate();
    rms_norm(hidden, final_norm_w, normed, 1e-6, stream);
    Tensor logits({1, embed.shape()[0]}, DType::Float16);
    logits.allocate();
    matmul_b_transposed(normed, embed, logits, stream);
    Tensor pred({1}, DType::Int64);
    pred.allocate();
    argmax_last_dim(logits, pred, stream);
    int64_t out = 0;
    pred.copy_to_host(&out, sizeof(out));
    return out;
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

static void build_rope(int64_t T, Tensor& cos_t, Tensor& sin_t) {
    constexpr int64_t Rot = 64;
    constexpr int64_t HalfRot = Rot / 2;
    constexpr float RopeTheta = 10000000.0f;
    std::vector<uint16_t> cos_host(T * HalfRot), sin_host(T * HalfRot);
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t i = 0; i < HalfRot; ++i) {
            float inv = std::pow(RopeTheta, -2.0f * static_cast<float>(i) / static_cast<float>(Rot));
            float theta = static_cast<float>(t) * inv;
            cos_host[t * HalfRot + i] = f2h(std::cos(theta));
            sin_host[t * HalfRot + i] = f2h(std::sin(theta));
        }
    }
    cos_t = Tensor({T, HalfRot}, DType::Float16);
    sin_t = Tensor({T, HalfRot}, DType::Float16);
    cos_t.copy_from_host(cos_host.data(), cos_host.size() * sizeof(uint16_t));
    sin_t.copy_from_host(sin_host.data(), sin_host.size() * sizeof(uint16_t));
}

static void run_full_sequence(const std::vector<int32_t>& tokens,
                              Tensor& embed,
                              const std::vector<std::string>& layer_types,
                              std::vector<LayerWeights>& layers,
                              const LinearAttentionDecoderLayerConfig& linear_config,
                              const FullAttentionDecoderLayerConfig& full_config,
                              Tensor& out_hidden,
                              aclrtStream stream) {
    const int64_t T = static_cast<int64_t>(tokens.size());
    constexpr int64_t Hidden = 1024;
    Tensor hidden({T, Hidden}, DType::Float16);
    Tensor next({T, Hidden}, DType::Float16);
    hidden.allocate();
    next.allocate();
    embedding_lookup(embed, tokens, hidden, stream);

    Tensor cos_t, sin_t;
    build_rope(T, cos_t, sin_t);
    std::vector<int32_t> row_to_t(T);
    for (int64_t t = 0; t < T; ++t) row_to_t[t] = static_cast<int32_t>(t);

    int full_i = 0;
    int linear_i = 0;
    (void)full_i;
    (void)linear_i;
    for (size_t layer = 0; layer < layer_types.size(); ++layer) {
        LayerWeights& lw = layers[layer];
        if (layer_types[layer] == "linear_attention") {
            LinearAttentionDecoderLayerWeights w{
                &lw.input_norm_w, &lw.post_norm_w, &lw.qkv_w, &lw.z_w, &lw.a_w,
                &lw.b_w, &lw.conv_w, &lw.dt_bias, &lw.a_log, &lw.gated_norm_w,
                &lw.out_proj_w, &lw.gate_w, &lw.up_w, &lw.down_w,
            };
            linear_attention_decoder_layer(hidden, w, linear_config, next, stream);
            ++linear_i;
        } else {
            FullAttentionDecoderLayerWeights w{
                &lw.input_norm_w, &lw.post_norm_w, &lw.q_w, &lw.k_w, &lw.v_w,
                &lw.o_w, &lw.q_norm_w, &lw.k_norm_w, &lw.gate_w, &lw.up_w,
                &lw.down_w,
            };
            full_attention_decoder_layer(hidden, w, cos_t, sin_t, row_to_t, full_config, next, stream);
            ++full_i;
        }
        copy_tensor(next, hidden, stream);
    }
    copy_last_row(hidden, out_hidden, stream);
}

static void run_step_sequence(const std::vector<int32_t>& tokens,
                              Tensor& embed,
                              const std::vector<std::string>& layer_types,
                              std::vector<LayerWeights>& layers,
                              const LinearAttentionDecoderLayerConfig& linear_config,
                              const FullAttentionDecoderLayerConfig& full_config,
                              DecodeState& state,
                              Tensor& out_hidden,
                              aclrtStream stream) {
    constexpr int64_t Hidden = 1024;
    const int64_t pos = state.seq_len;
    Tensor hidden({1, Hidden}, DType::Float16);
    Tensor next({1, Hidden}, DType::Float16);
    hidden.allocate();
    next.allocate();
    embedding_lookup(embed, {tokens.back()}, hidden, stream);

    Tensor cos_t, sin_t;
    build_rope(static_cast<int64_t>(tokens.size()), cos_t, sin_t);

    int full_i = 0;
    int linear_i = 0;
    for (size_t layer = 0; layer < layer_types.size(); ++layer) {
        LayerWeights& lw = layers[layer];
        if (layer_types[layer] == "linear_attention") {
            LinearAttentionDecoderLayerWeights w{
                &lw.input_norm_w, &lw.post_norm_w, &lw.qkv_w, &lw.z_w, &lw.a_w,
                &lw.b_w, &lw.conv_w, &lw.dt_bias, &lw.a_log, &lw.gated_norm_w,
                &lw.out_proj_w, &lw.gate_w, &lw.up_w, &lw.down_w,
            };
            linear_attention_decoder_layer_step(hidden, w, linear_config, state.linear[linear_i], next, stream);
            ++linear_i;
        } else {
            FullAttentionDecoderLayerWeights w{
                &lw.input_norm_w, &lw.post_norm_w, &lw.q_w, &lw.k_w, &lw.v_w,
                &lw.o_w, &lw.q_norm_w, &lw.k_norm_w, &lw.gate_w, &lw.up_w,
                &lw.down_w,
            };
            full_attention_decoder_layer_step(hidden, w, cos_t, sin_t, static_cast<int32_t>(pos),
                                              state.seq_len, full_config, state.full[full_i], next, stream);
            ++full_i;
        }
        copy_tensor(next, hidden, stream);
    }
    ++state.seq_len;
    copy_tensor(hidden, out_hidden, stream);
}

int main() {
    AclContext ctx(0);
    WeightsIndex index(default_safetensors_path());

    constexpr int64_t NumLayers = 24;
    constexpr int64_t Hidden = 1024;
    constexpr int64_t NumQHeads = 8;
    constexpr int64_t NumKVHeads = 2;
    constexpr int64_t HeadDim = 256;
    constexpr int64_t Rot = 64;

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

    LinearAttentionDecoderLayerConfig linear_config{1e-6};
    FullAttentionDecoderLayerConfig full_config{NumQHeads, NumKVHeads, HeadDim, Rot, 1e-6};
    DecodeState state = make_decode_state(8, layer_types, full_config, ctx.stream());

    std::vector<int32_t> tokens = {1, 2, 10, 100};
    Tensor ref_last({1, Hidden}, DType::Float16); ref_last.allocate();
    Tensor step_last({1, Hidden}, DType::Float16); step_last.allocate();

    float max_seen = 0.0f;
    for (size_t i = 0; i < tokens.size(); ++i) {
        std::vector<int32_t> prefix(tokens.begin(), tokens.begin() + static_cast<int64_t>(i) + 1);
        run_full_sequence(prefix, embed, layer_types, layers, linear_config, full_config, ref_last, ctx.stream());
        run_step_sequence(prefix, embed, layer_types, layers, linear_config, full_config, state, step_last, ctx.stream());
        float diff = max_abs_diff(ref_last, step_last);
        if (diff > max_seen) max_seen = diff;
        int64_t ref_tok = argmax_token(ref_last, final_norm_w, embed, ctx.stream());
        int64_t step_tok = argmax_token(step_last, final_norm_w, embed, ctx.stream());
        if (ref_tok != step_tok) {
            std::cerr << "incremental decode token mismatch step=" << i
                      << " ref=" << ref_tok << " step=" << step_tok << std::endl;
            return 1;
        }
        if (!std::isfinite(diff) || diff > 1.0f) {
            std::cerr << "incremental decode hidden mismatch step=" << i
                      << " max_abs=" << diff << std::endl;
            return 1;
        }
    }

    std::cout << "incremental decode smoke ok steps=" << tokens.size()
              << " max_abs=" << max_seen
              << " seq_len=" << state.seq_len << std::endl;
    return 0;
}
