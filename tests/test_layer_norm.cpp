#include "minicpmv/acl_context.h"
#include "minicpmv/ops.h"
#include "minicpmv/tensor.h"

#include <acl/acl.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
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

int main() {
    AclContext ctx(0);

    constexpr int64_t N = 4;
    constexpr int64_t H = 1152;
    constexpr float eps = 1e-6f;

    std::vector<uint16_t> x_host(N * H);
    std::vector<uint16_t> w_host(H);
    std::vector<uint16_t> b_host(H);
    std::vector<float> x_f(N * H);
    std::vector<float> w_f(H);
    std::vector<float> b_f(H);

    for (int64_t i = 0; i < N * H; ++i) {
        float v = static_cast<float>((i * 13 + 7) % 47) / 47.0f - 0.5f;
        x_f[i] = v * 0.5f;
        x_host[i] = f2h(x_f[i]);
    }
    for (int64_t i = 0; i < H; ++i) {
        w_f[i] = 1.0f + 0.01f * static_cast<float>((i % 9) - 4);
        b_f[i] = 0.01f * static_cast<float>((i % 5) - 2);
        w_host[i] = f2h(w_f[i]);
        b_host[i] = f2h(b_f[i]);
    }

    Tensor X({N, H}, DType::Float16);
    Tensor W({H}, DType::Float16);
    Tensor B({H}, DType::Float16);
    Tensor Y({N, H}, DType::Float16);
    X.copy_from_host(x_host.data(), x_host.size() * sizeof(uint16_t));
    W.copy_from_host(w_host.data(), w_host.size() * sizeof(uint16_t));
    B.copy_from_host(b_host.data(), b_host.size() * sizeof(uint16_t));
    Y.allocate();

    try {
        layer_norm(X, W, B, Y, static_cast<double>(eps), ctx.stream());
    } catch (const std::exception& e) {
        std::cerr << "layer_norm failed: " << e.what()
                  << "\nrecent: " << aclGetRecentErrMsg() << std::endl;
        return 1;
    }

    std::vector<uint16_t> got(N * H);
    Y.copy_to_host(got.data(), got.size() * sizeof(uint16_t));

    int errors = 0;
    float max_abs = 0.0f;
    for (int64_t n = 0; n < N; ++n) {
        double mean = 0.0;
        for (int64_t h = 0; h < H; ++h) mean += static_cast<double>(x_f[n * H + h]);
        mean /= static_cast<double>(H);
        double var = 0.0;
        for (int64_t h = 0; h < H; ++h) {
            double d = static_cast<double>(x_f[n * H + h]) - mean;
            var += d * d;
        }
        var /= static_cast<double>(H);
        double inv_std = 1.0 / std::sqrt(var + static_cast<double>(eps));
        for (int64_t h = 0; h < H; ++h) {
            double norm = (static_cast<double>(x_f[n * H + h]) - mean) * inv_std;
            float ref = static_cast<float>(norm * static_cast<double>(w_f[h]) +
                                           static_cast<double>(b_f[h]));
            float g = h2f(got[n * H + h]);
            float d = std::fabs(g - ref);
            if (d > max_abs) max_abs = d;
            if (d > 5e-3f) {
                if (errors < 3) {
                    std::cerr << "row=" << n << " col=" << h
                              << " got=" << g << " ref=" << ref
                              << " diff=" << d << std::endl;
                }
                ++errors;
            }
        }
    }

    std::cout << "layer_norm errors=" << errors
              << " max_abs=" << max_abs
              << " (N=" << N << " H=" << H << ")" << std::endl;
    if (errors) return 2;
    std::cout << "layer_norm ok" << std::endl;
    return 0;
}
