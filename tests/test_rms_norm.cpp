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

static int run_case(int64_t N, int64_t H) {
    AclContext ctx(0);
    constexpr float eps = 1e-6f;

    std::vector<uint16_t> x_host(N * H);
    std::vector<uint16_t> g_host(H);
    std::vector<float> x_f(N * H);
    std::vector<float> g_f(H);

    for (int64_t i = 0; i < N * H; ++i) {
        float v = (static_cast<float>((i * 17 + 5) % 41) / 41.0f - 0.5f) * 0.5f;
        x_f[i] = v;
        x_host[i] = f2h(v);
    }
    for (int64_t i = 0; i < H; ++i) {
        float v = 1.0f + 0.01f * static_cast<float>((i % 7) - 3);
        g_f[i] = v;
        g_host[i] = f2h(v);
    }

    Tensor X({N, H}, DType::Float16);
    Tensor G({H}, DType::Float16);
    Tensor Y({N, H}, DType::Float16);
    X.copy_from_host(x_host.data(), x_host.size() * sizeof(uint16_t));
    G.copy_from_host(g_host.data(), g_host.size() * sizeof(uint16_t));
    Y.allocate();

    try {
        rms_norm(X, G, Y, static_cast<double>(eps), ctx.stream());
    } catch (const std::exception& e) {
        std::cerr << "rms_norm H=" << H << " failed: " << e.what()
                  << "\nrecent: " << aclGetRecentErrMsg() << std::endl;
        return 1;
    }

    std::vector<uint16_t> got(N * H);
    Y.copy_to_host(got.data(), got.size() * sizeof(uint16_t));

    int errors = 0;
    float max_abs = 0.0f;
    for (int64_t n = 0; n < N; ++n) {
        double sumsq = 0.0;
        for (int64_t h = 0; h < H; ++h) {
            double v = static_cast<double>(x_f[n * H + h]);
            sumsq += v * v;
        }
        double rms = std::sqrt(sumsq / static_cast<double>(H) + static_cast<double>(eps));
        for (int64_t h = 0; h < H; ++h) {
            float ref = static_cast<float>(static_cast<double>(x_f[n * H + h]) / rms *
                                           static_cast<double>(g_f[h]));
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
    std::cout << "rms_norm H=" << H << " errors=" << errors
              << " max_abs=" << max_abs
              << " (N=" << N << ")" << std::endl;
    return errors ? 2 : 0;
}

int main() {
    int rc = 0;
    rc |= run_case(4, 256);
    rc |= run_case(4, 128);
    rc |= run_case(8, 1024);
    if (rc == 0) std::cout << "rms_norm ok" << std::endl;
    return rc;
}
