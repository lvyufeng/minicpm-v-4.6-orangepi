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

static Tensor load_layer_weight(WeightsIndex& index, const std::string& suffix) {
    return index.load_to_device_as("model.language_model.layers.0." + suffix, DType::Float16);
}

static bool all_finite_nonzero(const std::vector<uint16_t>& host, float* first) {
    bool nonzero = false;
    for (size_t i = 0; i < host.size(); ++i) {
        float v = h2f(host[i]);
        if (i == 0 && first) *first = v;
        if (!std::isfinite(v)) return false;
        if (std::fabs(v) > 0.0f) nonzero = true;
    }
    return nonzero;
}

int main() {
    AclContext ctx(0);
    WeightsIndex index(default_safetensors_path());

    constexpr int64_t T = 4;
    constexpr int64_t Hidden = 1024;
    constexpr int64_t KeyDim = 2048;
    constexpr int64_t ValueDim = 2048;
    constexpr int64_t ConvDim = KeyDim * 2 + ValueDim;
    constexpr int64_t Kernel = 4;

    Tensor embed = index.load_to_device_as("model.language_model.embed_tokens.weight", DType::Float16);
    Tensor norm_w = load_layer_weight(index, "input_layernorm.weight");
    Tensor qkv_w = load_layer_weight(index, "linear_attn.in_proj_qkv.weight");
    Tensor conv_w = load_layer_weight(index, "linear_attn.conv1d.weight");

    std::vector<int32_t> ids = {1, 2, 10, 100};
    Tensor hidden({T, Hidden}, DType::Float16);
    hidden.allocate();
    embedding_lookup(embed, ids, hidden, ctx.stream());

    Tensor normed({T, Hidden}, DType::Float16);
    normed.allocate();
    rms_norm(hidden, norm_w, normed, 1e-6, ctx.stream());

    Tensor qkv({T, ConvDim}, DType::Float16);
    qkv.allocate();
    matmul_b_transposed(normed, qkv_w, qkv, ctx.stream());

    std::vector<uint16_t> qkv_host(T * ConvDim);
    std::vector<uint16_t> conv_w_host(ConvDim * Kernel);
    qkv.copy_to_host(qkv_host.data(), qkv_host.size() * sizeof(uint16_t));
    conv_w.copy_to_host(conv_w_host.data(), conv_w_host.size() * sizeof(uint16_t));

    std::vector<uint16_t> conv_host(T * ConvDim);
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t c = 0; c < ConvDim; ++c) {
            float acc = 0.0f;
            for (int64_t k = 0; k < Kernel; ++k) {
                const int64_t src_t = t - (Kernel - 1) + k;
                if (src_t >= 0) {
                    acc += h2f(qkv_host[src_t * ConvDim + c]) * h2f(conv_w_host[c * Kernel + k]);
                }
            }
            conv_host[t * ConvDim + c] = f2h(acc);
        }
    }

    Tensor conv({T, ConvDim}, DType::Float16);
    Tensor mixed({T, ConvDim}, DType::Float16);
    conv.allocate();
    mixed.allocate();
    linear_causal_conv(qkv, conv_w, conv, ctx.stream());
    silu(conv, mixed, ctx.stream());

    std::vector<uint16_t> conv_got(T * ConvDim);
    std::vector<uint16_t> mixed_host(T * ConvDim);
    conv.copy_to_host(conv_got.data(), conv_got.size() * sizeof(uint16_t));
    mixed.copy_to_host(mixed_host.data(), mixed_host.size() * sizeof(uint16_t));

    int conv_errors = 0;
    float max_abs = 0.0f;
    for (size_t i = 0; i < conv_host.size(); ++i) {
        float d = std::fabs(h2f(conv_got[i]) - h2f(conv_host[i]));
        if (d > max_abs) max_abs = d;
        if (d > 2e-3f) ++conv_errors;
    }

    float conv0 = 0.0f, mixed0 = 0.0f;
    if (conv_errors || !all_finite_nonzero(conv_got, &conv0) || !all_finite_nonzero(mixed_host, &mixed0)) {
        std::cerr << "linear attention conv smoke produced invalid output errors=" << conv_errors
                  << " max_abs=" << max_abs << std::endl;
        return 1;
    }

    std::cout << "linear attention conv smoke ok"
              << " q=[" << T << "," << KeyDim << "]"
              << " k=[" << T << "," << KeyDim << "]"
              << " v=[" << T << "," << ValueDim << "]"
              << " conv=[" << T << "," << ConvDim << "]"
              << " max_abs=" << max_abs
              << " first=" << conv0 << "," << mixed0 << std::endl;
    return 0;
}
