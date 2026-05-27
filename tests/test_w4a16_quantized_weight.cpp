#include "minicpmv/acl_context.h"
#include "minicpmv/ops.h"
#include "minicpmv/quantized_weight.h"
#include "minicpmv/tensor.h"
#include "minicpmv/weights.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
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
            while ((mant & 0x400u) == 0) { mant <<= 1; --e; }
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

static void run_case(AclContext& ctx, const std::string& path, const char* label) {
    WeightsIndex index(path);
    const std::string base = "model.language_model.layers.0.mlp.gate_proj";
    if (!has_w4a16_quantized_weight(index, base)) {
        throw std::runtime_error(std::string(label) + " missing quantized gate_proj tensors");
    }
    W4A16QuantizedWeight q = load_w4a16_quantized_weight(index, base);
    if (q.K != 1024 || q.N != 3584 || q.group_size != 128 ||
        q.w_int8.shape().empty() || q.scales.shape() != std::vector<int64_t>{8, 3584}) {
        throw std::runtime_error(std::string(label) + " unexpected W4A16 helper output shape");
    }

    std::vector<uint16_t> hx(q.K, f2h(0.05f));
    Tensor x({1, q.K}, DType::Float16);
    x.copy_from_host(hx.data(), hx.size() * sizeof(uint16_t));
    Tensor out({1, q.N}, DType::Float16);
    out.allocate();
    matmul_w4a16(x, q.w_int8, q.scales, out, ctx.stream());

    std::vector<uint16_t> ho(q.N);
    out.copy_to_host(ho.data(), ho.size() * sizeof(uint16_t));
    float max_abs = 0.0f;
    float sum_abs = 0.0f;
    for (uint16_t h : ho) {
        const float v = h2f(h);
        if (!std::isfinite(v)) {
            throw std::runtime_error(std::string(label) + " produced non-finite output");
        }
        const float a = std::fabs(v);
        max_abs = std::max(max_abs, a);
        sum_abs += a;
    }
    if (max_abs == 0.0f || sum_abs == 0.0f) {
        throw std::runtime_error(std::string(label) + " produced all-zero output");
    }
    std::printf("%s production W4A16 helper ok K=%ld N=%ld max_abs=%.4f mean_abs=%.4f\n",
                label, q.K, q.N, max_abs, sum_abs / q.N);
}

int main() {
    AclContext ctx(0);
    run_case(ctx, "MiniCPM-V-4.6-GPTQ/model.safetensors", "GPTQ");
    run_case(ctx, "MiniCPM-V-4.6-AWQ/model.safetensors", "AWQ");
    return 0;
}
