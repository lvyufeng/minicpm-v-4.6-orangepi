#include "minicpmv/acl_context.h"
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

static bool all_finite_nonzero_sample(const Tensor& t, float* first) {
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

static Tensor load_layer_weight(WeightsIndex& index, const std::string& suffix) {
    return index.load_to_device_as("model.language_model.layers.0." + suffix, DType::Float16);
}

int main() {
    AclContext ctx(0);
    WeightsIndex index("/mnt/data/minicpm/MiniCPM-V-4.6/model.safetensors");

    constexpr int64_t T = 4;
    constexpr int64_t Hidden = 1024;
    constexpr int64_t QKV = 6144;
    constexpr int64_t Z = 2048;
    constexpr int64_t Heads = 16;

    Tensor embed = index.load_to_device_as("model.language_model.embed_tokens.weight", DType::Float16);
    Tensor norm_w = load_layer_weight(index, "input_layernorm.weight");
    Tensor qkv_w = load_layer_weight(index, "linear_attn.in_proj_qkv.weight");
    Tensor z_w = load_layer_weight(index, "linear_attn.in_proj_z.weight");
    Tensor a_w = load_layer_weight(index, "linear_attn.in_proj_a.weight");
    Tensor b_w = load_layer_weight(index, "linear_attn.in_proj_b.weight");
    Tensor conv_w = load_layer_weight(index, "linear_attn.conv1d.weight");
    Tensor dt_bias = load_layer_weight(index, "linear_attn.dt_bias");
    Tensor a_log = load_layer_weight(index, "linear_attn.A_log");
    Tensor out_w = load_layer_weight(index, "linear_attn.out_proj.weight");

    if (embed.shape() != std::vector<int64_t>{248094, Hidden} ||
        norm_w.shape() != std::vector<int64_t>{Hidden} ||
        qkv_w.shape() != std::vector<int64_t>{QKV, Hidden} ||
        z_w.shape() != std::vector<int64_t>{Z, Hidden} ||
        a_w.shape() != std::vector<int64_t>{Heads, Hidden} ||
        b_w.shape() != std::vector<int64_t>{Heads, Hidden} ||
        conv_w.shape() != std::vector<int64_t>{QKV, 1, 4} ||
        dt_bias.shape() != std::vector<int64_t>{Heads} ||
        a_log.shape() != std::vector<int64_t>{Heads} ||
        out_w.shape() != std::vector<int64_t>{Hidden, Z}) {
        std::cerr << "unexpected layer0 linear-attention weight shape" << std::endl;
        return 1;
    }

    std::vector<int32_t> ids = {1, 2, 10, 100};
    Tensor hidden({T, Hidden}, DType::Float16);
    hidden.allocate();
    embedding_lookup(embed, ids, hidden, ctx.stream());

    Tensor normed({T, Hidden}, DType::Float16);
    normed.allocate();
    rms_norm(hidden, norm_w, normed, 1e-6, ctx.stream());

    Tensor qkv({T, QKV}, DType::Float16);
    Tensor z({T, Z}, DType::Float16);
    Tensor a({T, Heads}, DType::Float16);
    Tensor b({T, Heads}, DType::Float16);
    qkv.allocate();
    z.allocate();
    a.allocate();
    b.allocate();

    matmul_b_transposed(normed, qkv_w, qkv, ctx.stream());
    matmul_b_transposed(normed, z_w, z, ctx.stream());
    matmul_b_transposed(normed, a_w, a, ctx.stream());
    matmul_b_transposed(normed, b_w, b, ctx.stream());

    float qkv0 = 0.0f, z0 = 0.0f, a0 = 0.0f, b0 = 0.0f;
    if (!all_finite_nonzero_sample(qkv, &qkv0) ||
        !all_finite_nonzero_sample(z, &z0) ||
        !all_finite_nonzero_sample(a, &a0) ||
        !all_finite_nonzero_sample(b, &b0)) {
        std::cerr << "linear attention projection smoke produced invalid output" << std::endl;
        return 2;
    }

    std::cout << "linear attention projection smoke ok"
              << " qkv=[" << qkv.shape()[0] << "," << qkv.shape()[1] << "]"
              << " z=[" << z.shape()[0] << "," << z.shape()[1] << "]"
              << " a=[" << a.shape()[0] << "," << a.shape()[1] << "]"
              << " b=[" << b.shape()[0] << "," << b.shape()[1] << "]"
              << " first=" << qkv0 << "," << z0 << "," << a0 << "," << b0 << std::endl;
    return 0;
}
