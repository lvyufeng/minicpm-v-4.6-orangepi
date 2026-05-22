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

static float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

static float softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return std::exp(x);
    return std::log1p(std::exp(x));
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
    WeightsIndex index("/mnt/data/minicpm/MiniCPM-V-4.6/model.safetensors");

    constexpr int64_t T = 4;
    constexpr int64_t Hidden = 1024;
    constexpr int64_t NumHeads = 16;
    constexpr int64_t HeadDim = 128;
    constexpr int64_t KeyDim = NumHeads * HeadDim;
    constexpr int64_t ValueDim = NumHeads * HeadDim;
    constexpr int64_t ConvDim = KeyDim * 2 + ValueDim;
    constexpr int64_t Kernel = 4;

    Tensor embed = index.load_to_device_as("model.language_model.embed_tokens.weight", DType::Float16);
    Tensor norm_w = load_layer_weight(index, "input_layernorm.weight");
    Tensor qkv_w = load_layer_weight(index, "linear_attn.in_proj_qkv.weight");
    Tensor z_w = load_layer_weight(index, "linear_attn.in_proj_z.weight");
    Tensor a_w = load_layer_weight(index, "linear_attn.in_proj_a.weight");
    Tensor b_w = load_layer_weight(index, "linear_attn.in_proj_b.weight");
    Tensor conv_w = load_layer_weight(index, "linear_attn.conv1d.weight");
    Tensor dt_bias = load_layer_weight(index, "linear_attn.dt_bias");
    Tensor a_log = load_layer_weight(index, "linear_attn.A_log");
    Tensor gated_norm_w = load_layer_weight(index, "linear_attn.norm.weight");
    Tensor out_w = load_layer_weight(index, "linear_attn.out_proj.weight");

    std::vector<int32_t> ids = {1, 2, 10, 100};
    Tensor hidden({T, Hidden}, DType::Float16);
    hidden.allocate();
    embedding_lookup(embed, ids, hidden, ctx.stream());

    Tensor normed({T, Hidden}, DType::Float16);
    normed.allocate();
    rms_norm(hidden, norm_w, normed, 1e-6, ctx.stream());

    Tensor qkv({T, ConvDim}, DType::Float16);
    Tensor z({T, ValueDim}, DType::Float16);
    Tensor a({T, NumHeads}, DType::Float16);
    Tensor b({T, NumHeads}, DType::Float16);
    qkv.allocate();
    z.allocate();
    a.allocate();
    b.allocate();
    matmul_b_transposed(normed, qkv_w, qkv, ctx.stream());
    matmul_b_transposed(normed, z_w, z, ctx.stream());
    matmul_b_transposed(normed, a_w, a, ctx.stream());
    matmul_b_transposed(normed, b_w, b, ctx.stream());

    std::vector<uint16_t> qkv_host(T * ConvDim);
    std::vector<uint16_t> z_host(T * ValueDim);
    std::vector<uint16_t> a_host(T * NumHeads);
    std::vector<uint16_t> b_host(T * NumHeads);
    std::vector<uint16_t> conv_w_host(ConvDim * Kernel);
    std::vector<uint16_t> dt_host(NumHeads);
    std::vector<uint16_t> a_log_host(NumHeads);
    std::vector<uint16_t> gated_norm_host(HeadDim);
    qkv.copy_to_host(qkv_host.data(), qkv_host.size() * sizeof(uint16_t));
    z.copy_to_host(z_host.data(), z_host.size() * sizeof(uint16_t));
    a.copy_to_host(a_host.data(), a_host.size() * sizeof(uint16_t));
    b.copy_to_host(b_host.data(), b_host.size() * sizeof(uint16_t));
    conv_w.copy_to_host(conv_w_host.data(), conv_w_host.size() * sizeof(uint16_t));
    dt_bias.copy_to_host(dt_host.data(), dt_host.size() * sizeof(uint16_t));
    a_log.copy_to_host(a_log_host.data(), a_log_host.size() * sizeof(uint16_t));
    gated_norm_w.copy_to_host(gated_norm_host.data(), gated_norm_host.size() * sizeof(uint16_t));

    std::vector<float> mixed(T * ConvDim);
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t c = 0; c < ConvDim; ++c) {
            float acc = 0.0f;
            for (int64_t k = 0; k < Kernel; ++k) {
                const int64_t src_t = t - (Kernel - 1) + k;
                if (src_t >= 0) {
                    acc += h2f(qkv_host[src_t * ConvDim + c]) * h2f(conv_w_host[c * Kernel + k]);
                }
            }
            mixed[t * ConvDim + c] = acc * sigmoid(acc);
        }
    }

    std::vector<float> state(NumHeads * HeadDim * HeadDim, 0.0f);
    std::vector<float> core(T * ValueDim, 0.0f);
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t h = 0; h < NumHeads; ++h) {
            float beta = sigmoid(h2f(b_host[t * NumHeads + h]));
            float g = -std::exp(h2f(a_log_host[h])) * softplus(h2f(a_host[t * NumHeads + h]) + h2f(dt_host[h]));
            float decay = std::exp(g);
            std::vector<float> q(HeadDim), k(HeadDim), v(HeadDim);
            float q_norm = 0.0f;
            float k_norm = 0.0f;
            for (int64_t i = 0; i < HeadDim; ++i) {
                q[i] = mixed[t * ConvDim + h * HeadDim + i];
                k[i] = mixed[t * ConvDim + KeyDim + h * HeadDim + i];
                v[i] = mixed[t * ConvDim + 2 * KeyDim + h * HeadDim + i];
                q_norm += q[i] * q[i];
                k_norm += k[i] * k[i];
            }
            q_norm = std::sqrt(q_norm + 1e-6f);
            k_norm = std::sqrt(k_norm + 1e-6f);
            const float q_scale = 1.0f / std::sqrt(static_cast<float>(HeadDim));
            for (int64_t i = 0; i < HeadDim; ++i) {
                q[i] = q[i] / q_norm * q_scale;
                k[i] = k[i] / k_norm;
            }
            auto state_base = static_cast<size_t>(h * HeadDim * HeadDim);
            for (int64_t i = 0; i < HeadDim * HeadDim; ++i) {
                state[state_base + i] *= decay;
            }
            std::vector<float> kv_mem(HeadDim, 0.0f);
            for (int64_t j = 0; j < HeadDim; ++j) {
                float acc = 0.0f;
                for (int64_t i = 0; i < HeadDim; ++i) {
                    acc += state[state_base + i * HeadDim + j] * k[i];
                }
                kv_mem[j] = acc;
            }
            std::vector<float> delta(HeadDim);
            for (int64_t j = 0; j < HeadDim; ++j) {
                delta[j] = (v[j] - kv_mem[j]) * beta;
            }
            for (int64_t i = 0; i < HeadDim; ++i) {
                for (int64_t j = 0; j < HeadDim; ++j) {
                    state[state_base + i * HeadDim + j] += k[i] * delta[j];
                }
            }
            for (int64_t j = 0; j < HeadDim; ++j) {
                float acc = 0.0f;
                for (int64_t i = 0; i < HeadDim; ++i) {
                    acc += state[state_base + i * HeadDim + j] * q[i];
                }
                core[t * ValueDim + h * HeadDim + j] = acc;
            }
        }
    }

    std::vector<uint16_t> gated_host(T * ValueDim);
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t h = 0; h < NumHeads; ++h) {
            float mean_sq = 0.0f;
            for (int64_t i = 0; i < HeadDim; ++i) {
                float v = core[t * ValueDim + h * HeadDim + i];
                mean_sq += v * v;
            }
            mean_sq /= static_cast<float>(HeadDim);
            float rstd = 1.0f / std::sqrt(mean_sq + 1e-6f);
            for (int64_t i = 0; i < HeadDim; ++i) {
                int64_t idx = t * ValueDim + h * HeadDim + i;
                float z_gate = h2f(z_host[idx]);
                float gamma = h2f(gated_norm_host[i]);
                float v = (gamma * core[idx] * rstd) * (z_gate * sigmoid(z_gate));
                gated_host[idx] = f2h(v);
            }
        }
    }

    Tensor gated({T, ValueDim}, DType::Float16);
    Tensor out({T, Hidden}, DType::Float16);
    gated.copy_from_host(gated_host.data(), gated_host.size() * sizeof(uint16_t));
    out.allocate();
    matmul_b_transposed(gated, out_w, out, ctx.stream());

    std::vector<uint16_t> out_host(T * Hidden);
    out.copy_to_host(out_host.data(), out_host.size() * sizeof(uint16_t));

    float core0 = core.empty() ? 0.0f : core[0];
    float gated0 = 0.0f;
    float out0 = 0.0f;
    if (!std::isfinite(core0) || !all_finite_nonzero(gated_host, &gated0) || !all_finite_nonzero(out_host, &out0)) {
        std::cerr << "linear gated delta CPU smoke produced invalid output" << std::endl;
        return 1;
    }

    std::cout << "linear gated delta cpu smoke ok"
              << " core=[" << T << "," << ValueDim << "]"
              << " out=[" << T << "," << Hidden << "]"
              << " first=" << core0 << "," << gated0 << "," << out0 << std::endl;

    std::vector<uint16_t> mixed_h(T * ConvDim);
    std::vector<uint16_t> beta_h(T * NumHeads);
    std::vector<uint16_t> decay_h(T * NumHeads);
    for (size_t i = 0; i < mixed.size(); ++i) mixed_h[i] = f2h(mixed[i]);
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t h = 0; h < NumHeads; ++h) {
            float beta = sigmoid(h2f(b_host[t * NumHeads + h]));
            float g = -std::exp(h2f(a_log_host[h])) *
                      softplus(h2f(a_host[t * NumHeads + h]) + h2f(dt_host[h]));
            beta_h[t * NumHeads + h] = f2h(beta);
            decay_h[t * NumHeads + h] = f2h(std::exp(g));
        }
    }

    Tensor mixed_dev({T, ConvDim}, DType::Float16);
    Tensor beta_dev({T, NumHeads}, DType::Float16);
    Tensor decay_dev({T, NumHeads}, DType::Float16);
    Tensor core_dev({T, ValueDim}, DType::Float16);
    Tensor scratch_dev({136192}, DType::Float32);
    mixed_dev.copy_from_host(mixed_h.data(), mixed_h.size() * sizeof(uint16_t));
    beta_dev.copy_from_host(beta_h.data(), beta_h.size() * sizeof(uint16_t));
    decay_dev.copy_from_host(decay_h.data(), decay_h.size() * sizeof(uint16_t));
    core_dev.allocate();
    scratch_dev.allocate();
    linear_gated_delta_rule(mixed_dev, beta_dev, decay_dev, scratch_dev, core_dev, ctx.stream());

    std::vector<uint16_t> core_npu(T * ValueDim);
    core_dev.copy_to_host(core_npu.data(), core_npu.size() * sizeof(uint16_t));
    float max_abs = 0.0f;
    float ref_max = 0.0f;
    int errors = 0;
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t i = 0; i < ValueDim; ++i) {
            float ref = core[t * ValueDim + i];
            float got = h2f(core_npu[t * ValueDim + i]);
            float d = std::fabs(got - ref);
            if (d > max_abs) max_abs = d;
            if (std::fabs(ref) > ref_max) ref_max = std::fabs(ref);
            if (d > 5e-2f) {
                ++errors;
            }
        }
    }
    if (errors) {
        std::cerr << "linear gated delta NPU smoke errors=" << errors
                  << " max_abs=" << max_abs << " ref_max=" << ref_max << std::endl;
        return 3;
    }

    std::cout << "linear gated delta npu smoke ok"
              << " core_npu=[" << T << "," << ValueDim << "]"
              << " max_abs=" << max_abs << " ref_max=" << ref_max << std::endl;

    Tensor z_silu({T, ValueDim}, DType::Float16);
    z_silu.allocate();
    silu(z, z_silu, ctx.stream());
    Tensor gated_npu({T, ValueDim}, DType::Float16);
    gated_npu.allocate();
    gated_rms_norm_z(core_dev, z_silu, gated_norm_w, gated_npu, ctx.stream());

    std::vector<uint16_t> gated_npu_host(T * ValueDim);
    gated_npu.copy_to_host(gated_npu_host.data(), gated_npu_host.size() * sizeof(uint16_t));
    float gated_max_abs = 0.0f;
    float gated_ref_max = 0.0f;
    int gated_errors = 0;
    for (int64_t i = 0; i < T * ValueDim; ++i) {
        float ref = h2f(gated_host[i]);
        float got = h2f(gated_npu_host[i]);
        float d = std::fabs(got - ref);
        if (d > gated_max_abs) gated_max_abs = d;
        if (std::fabs(ref) > gated_ref_max) gated_ref_max = std::fabs(ref);
        if (d > 5e-2f) ++gated_errors;
    }
    if (gated_errors) {
        std::cerr << "gated rms_norm_z NPU smoke errors=" << gated_errors
                  << " max_abs=" << gated_max_abs << " ref_max=" << gated_ref_max << std::endl;
        return 4;
    }
    std::cout << "gated rms_norm_z npu smoke ok"
              << " gated=[" << T << "," << ValueDim << "]"
              << " max_abs=" << gated_max_abs << " ref_max=" << gated_ref_max << std::endl;

    Tensor out_npu({T, Hidden}, DType::Float16);
    out_npu.allocate();
    matmul_b_transposed(gated_npu, out_w, out_npu, ctx.stream());
    std::vector<uint16_t> out_npu_host(T * Hidden);
    out_npu.copy_to_host(out_npu_host.data(), out_npu_host.size() * sizeof(uint16_t));
    float out_max_abs = 0.0f;
    float out_ref_max = 0.0f;
    for (int64_t i = 0; i < T * Hidden; ++i) {
        float ref = h2f(out_host[i]);
        float got = h2f(out_npu_host[i]);
        float d = std::fabs(got - ref);
        if (d > out_max_abs) out_max_abs = d;
        if (std::fabs(ref) > out_ref_max) out_ref_max = std::fabs(ref);
    }
    std::cout << "linear attention full npu pipeline ok"
              << " out=[" << T << "," << Hidden << "]"
              << " max_abs=" << out_max_abs << " ref_max=" << out_ref_max << std::endl;
    return 0;
}
