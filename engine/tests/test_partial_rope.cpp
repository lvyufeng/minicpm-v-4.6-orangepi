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
    constexpr int64_t N = 40;   // B*Hh*T collapsed
    constexpr int64_t T = 5;
    constexpr int64_t D = 256;
    constexpr int64_t Rot = 64;
    constexpr int64_t HalfRot = Rot / 2;

    std::vector<uint16_t> x_host(N * D);
    std::vector<uint16_t> cos_host(T * HalfRot);
    std::vector<uint16_t> sin_host(T * HalfRot);
    std::vector<float> x_f(N * D), cos_f(T * HalfRot), sin_f(T * HalfRot);
    std::vector<int32_t> row_to_t(N);

    for (int64_t i = 0; i < N * D; ++i) {
        float v = (static_cast<float>((i * 7 + 3) % 29) / 29.0f - 0.5f) * 0.5f;
        x_f[i] = v;
        x_host[i] = f2h(v);
    }
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t i = 0; i < HalfRot; ++i) {
            float theta = static_cast<float>(t + 1) * 0.01f * static_cast<float>(i + 1);
            cos_f[t * HalfRot + i] = std::cos(theta);
            sin_f[t * HalfRot + i] = std::sin(theta);
            cos_host[t * HalfRot + i] = f2h(cos_f[t * HalfRot + i]);
            sin_host[t * HalfRot + i] = f2h(sin_f[t * HalfRot + i]);
        }
    }
    for (int64_t n = 0; n < N; ++n) row_to_t[n] = static_cast<int32_t>(n % T);

    Tensor X({N, D}, DType::Float16);
    Tensor C({T, HalfRot}, DType::Float16);
    Tensor S({T, HalfRot}, DType::Float16);
    Tensor O({N, D}, DType::Float16);
    X.copy_from_host(x_host.data(), x_host.size() * sizeof(uint16_t));
    C.copy_from_host(cos_host.data(), cos_host.size() * sizeof(uint16_t));
    S.copy_from_host(sin_host.data(), sin_host.size() * sizeof(uint16_t));
    O.allocate();

    try {
        apply_rope_partial(X, C, S, row_to_t, Rot, O, ctx.stream());
    } catch (const std::exception& e) {
        std::cerr << "apply_rope_partial failed: " << e.what()
                  << "\nrecent: " << aclGetRecentErrMsg() << std::endl;
        return 1;
    }

    std::vector<uint16_t> got(N * D);
    O.copy_to_host(got.data(), got.size() * sizeof(uint16_t));

    int errors = 0;
    float max_abs = 0.0f;
    for (int64_t n = 0; n < N; ++n) {
        int64_t t = row_to_t[n];
        int64_t base = n * D;
        for (int64_t i = 0; i < HalfRot; ++i) {
            float x1 = x_f[base + i];
            float x2 = x_f[base + HalfRot + i];
            float c = cos_f[t * HalfRot + i];
            float s = sin_f[t * HalfRot + i];
            float ref1 = x1 * c - x2 * s;
            float ref2 = x2 * c + x1 * s;
            float g1 = h2f(got[base + i]);
            float g2 = h2f(got[base + HalfRot + i]);
            float d1 = std::fabs(g1 - ref1);
            float d2 = std::fabs(g2 - ref2);
            if (d1 > max_abs) max_abs = d1;
            if (d2 > max_abs) max_abs = d2;
            if (d1 > 5e-3f || d2 > 5e-3f) {
                if (errors < 3) {
                    std::cerr << "row=" << n << " i=" << i
                              << " g1=" << g1 << " ref1=" << ref1
                              << " g2=" << g2 << " ref2=" << ref2 << std::endl;
                }
                ++errors;
            }
        }
        for (int64_t i = Rot; i < D; ++i) {
            float g = h2f(got[base + i]);
            float r = x_f[base + i];
            float d = std::fabs(g - r);
            if (d > max_abs) max_abs = d;
            if (d > 5e-3f) {
                if (errors < 3) {
                    std::cerr << "tail mismatch row=" << n << " i=" << i
                              << " g=" << g << " ref=" << r << std::endl;
                }
                ++errors;
            }
        }
    }

    std::cout << "partial_rope errors=" << errors
              << " max_abs=" << max_abs
              << " (N=" << N << " D=" << D << " rot=" << Rot << ")" << std::endl;
    if (errors) return 2;
    std::cout << "partial_rope ok" << std::endl;
    return 0;
}
