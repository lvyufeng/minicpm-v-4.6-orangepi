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

static bool all_finite_nonzero(const Tensor& t, float* first) {
    std::vector<uint16_t> host(t.numel());
    t.copy_to_host(host.data(), host.size() * sizeof(uint16_t));
    bool nonzero = false;
    for (size_t i = 0; i < host.size(); ++i) {
        float v = h2f(host[i]);
        if (i == 0 && first) *first = v;
        if (!std::isfinite(v)) return false;
        if (std::fabs(v) > 0.0f) nonzero = true;
    }
    return nonzero;
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

int main() {
    AclContext ctx(0);
    WeightsIndex index("/mnt/data/minicpm/MiniCPM-V-4.6/model.safetensors");

    constexpr int64_t T = 4;
    constexpr int64_t Hidden = 1024;
    constexpr int64_t NumQHeads = 8;
    constexpr int64_t NumKVHeads = 2;
    constexpr int64_t HeadDim = 256;
    constexpr int64_t Rot = 64;
    constexpr int64_t HalfRot = Rot / 2;

    Tensor embed = index.load_to_device_as("model.language_model.embed_tokens.weight", DType::Float16);
    std::vector<int32_t> ids = {1, 2, 10, 100};
    Tensor hidden({T, Hidden}, DType::Float16);
    Tensor next({T, Hidden}, DType::Float16);
    hidden.allocate();
    next.allocate();
    embedding_lookup(embed, ids, hidden, ctx.stream());

    LinearAttentionDecoderLayerConfig linear_config{1e-6};
    for (int layer = 0; layer < 3; ++layer) {
        Tensor input_norm_w = load_layer_weight(index, layer, "input_layernorm.weight");
        Tensor post_norm_w = load_layer_weight(index, layer, "post_attention_layernorm.weight");
        Tensor qkv_w = load_layer_weight(index, layer, "linear_attn.in_proj_qkv.weight");
        Tensor z_w = load_layer_weight(index, layer, "linear_attn.in_proj_z.weight");
        Tensor a_w = load_layer_weight(index, layer, "linear_attn.in_proj_a.weight");
        Tensor b_w = load_layer_weight(index, layer, "linear_attn.in_proj_b.weight");
        Tensor conv_w = load_layer_weight(index, layer, "linear_attn.conv1d.weight");
        Tensor dt_bias_t = load_layer_weight(index, layer, "linear_attn.dt_bias");
        Tensor a_log_t = load_layer_weight(index, layer, "linear_attn.A_log");
        Tensor gated_norm_w = load_layer_weight(index, layer, "linear_attn.norm.weight");
        Tensor out_proj_w = load_layer_weight(index, layer, "linear_attn.out_proj.weight");
        Tensor gate_w = load_layer_weight(index, layer, "mlp.gate_proj.weight");
        Tensor up_w = load_layer_weight(index, layer, "mlp.up_proj.weight");
        Tensor down_w = load_layer_weight(index, layer, "mlp.down_proj.weight");

        LinearAttentionDecoderLayerWeights weights{
            &input_norm_w,
            &post_norm_w,
            &qkv_w,
            &z_w,
            &a_w,
            &b_w,
            &conv_w,
            &dt_bias_t,
            &a_log_t,
            &gated_norm_w,
            &out_proj_w,
            &gate_w,
            &up_w,
            &down_w,
        };
        linear_attention_decoder_layer(hidden, weights, linear_config, next, ctx.stream());
        copy_tensor(next, hidden, ctx.stream());
    }

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

    Tensor input_norm_w = load_layer_weight(index, 3, "input_layernorm.weight");
    Tensor post_norm_w = load_layer_weight(index, 3, "post_attention_layernorm.weight");
    Tensor q_w = load_layer_weight(index, 3, "self_attn.q_proj.weight");
    Tensor k_w = load_layer_weight(index, 3, "self_attn.k_proj.weight");
    Tensor v_w = load_layer_weight(index, 3, "self_attn.v_proj.weight");
    Tensor o_w = load_layer_weight(index, 3, "self_attn.o_proj.weight");
    Tensor q_norm_w = load_layer_weight(index, 3, "self_attn.q_norm.weight");
    Tensor k_norm_w = load_layer_weight(index, 3, "self_attn.k_norm.weight");
    Tensor gate_w = load_layer_weight(index, 3, "mlp.gate_proj.weight");
    Tensor up_w = load_layer_weight(index, 3, "mlp.up_proj.weight");
    Tensor down_w = load_layer_weight(index, 3, "mlp.down_proj.weight");

    FullAttentionDecoderLayerWeights full_weights{
        &input_norm_w,
        &post_norm_w,
        &q_w,
        &k_w,
        &v_w,
        &o_w,
        &q_norm_w,
        &k_norm_w,
        &gate_w,
        &up_w,
        &down_w,
    };
    FullAttentionDecoderLayerConfig full_config{
        NumQHeads,
        NumKVHeads,
        HeadDim,
        Rot,
        1e-6,
    };
    std::vector<int32_t> row_to_t = {0, 1, 2, 3};
    full_attention_decoder_layer(hidden, full_weights, cos_t, sin_t, row_to_t, full_config, next, ctx.stream());

    float first = 0.0f;
    if (!all_finite_nonzero(next, &first)) {
        std::cerr << "decoder stack smoke produced invalid output" << std::endl;
        return 1;
    }

    std::cout << "decoder stack smoke ok layers=0,1,2(linear_npu),3(full)"
              << " final=[" << next.shape()[0] << "," << next.shape()[1] << "]"
              << " first=" << first << std::endl;
    return 0;
}
