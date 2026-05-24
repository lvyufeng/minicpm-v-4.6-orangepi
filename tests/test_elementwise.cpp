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

static int check_close(const std::vector<uint16_t>& got, const std::vector<float>& ref,
                       float tol, const char* name) {
    int errors = 0;
    float max_abs = 0.0f;
    for (size_t i = 0; i < got.size(); ++i) {
        float g = h2f(got[i]);
        float r = ref[i];
        float d = std::fabs(g - r);
        if (d > max_abs) max_abs = d;
        if (d > tol) {
            if (errors < 3) {
                std::cerr << name << " mismatch i=" << i
                          << " got=" << g << " ref=" << r << " diff=" << d << "\n";
            }
            ++errors;
        }
    }
    std::cout << name << " errors=" << errors << " max_abs=" << max_abs << "\n";
    return errors;
}

int main() {
    AclContext ctx(0);

    constexpr int64_t N = 64;
    std::vector<uint16_t> a_host(N), b_host(N);
    std::vector<float> a_f(N), b_f(N);
    for (int64_t i = 0; i < N; ++i) {
        float av = static_cast<float>((i * 11 + 5) % 17) / 17.0f - 0.5f;
        float bv = static_cast<float>((i * 7 + 3) % 13) / 13.0f - 0.5f;
        a_f[i] = av;
        b_f[i] = bv;
        a_host[i] = f2h(av);
        b_host[i] = f2h(bv);
    }

    Tensor A({N}, DType::Float16);
    Tensor B({N}, DType::Float16);
    Tensor C({N}, DType::Float16);
    A.copy_from_host(a_host.data(), a_host.size() * sizeof(uint16_t));
    B.copy_from_host(b_host.data(), b_host.size() * sizeof(uint16_t));
    C.allocate();

    auto try_op = [&](const char* name, auto fn) -> int {
        try {
            fn();
        } catch (const std::exception& e) {
            std::cerr << name << " failed: " << e.what()
                      << "\nrecent: " << aclGetRecentErrMsg() << std::endl;
            return -1;
        }
        return 0;
    };

    int rc = 0;

    if (try_op("add", [&] { add(A, B, C, ctx.stream()); }) < 0) return 1;
    {
        std::vector<uint16_t> got(N);
        C.copy_to_host(got.data(), got.size() * sizeof(uint16_t));
        std::vector<float> ref(N);
        for (int64_t i = 0; i < N; ++i) ref[i] = a_f[i] + b_f[i];
        rc |= check_close(got, ref, 5e-3f, "add");
    }

    if (try_op("mul", [&] { mul(A, B, C, ctx.stream()); }) < 0) return 1;
    {
        std::vector<uint16_t> got(N);
        C.copy_to_host(got.data(), got.size() * sizeof(uint16_t));
        std::vector<float> ref(N);
        for (int64_t i = 0; i < N; ++i) ref[i] = a_f[i] * b_f[i];
        rc |= check_close(got, ref, 5e-3f, "mul");
    }

    if (try_op("silu", [&] { silu(A, C, ctx.stream()); }) < 0) return 1;
    {
        std::vector<uint16_t> got(N);
        C.copy_to_host(got.data(), got.size() * sizeof(uint16_t));
        std::vector<float> ref(N);
        for (int64_t i = 0; i < N; ++i) {
            float x = a_f[i];
            ref[i] = x / (1.0f + std::exp(-x));
        }
        rc |= check_close(got, ref, 5e-3f, "silu");
    }

    {
        constexpr int64_t M = 4;
        constexpr int64_t K = 16;
        std::vector<uint16_t> sm_host(M * K);
        std::vector<float> sm_f(M * K);
        for (int64_t i = 0; i < M * K; ++i) {
            float v = static_cast<float>((i * 5 + 1) % 11) / 11.0f - 0.5f;
            sm_f[i] = v * 2.0f;
            sm_host[i] = f2h(sm_f[i]);
        }
        Tensor S({M, K}, DType::Float16);
        Tensor SO({M, K}, DType::Float16);
        S.copy_from_host(sm_host.data(), sm_host.size() * sizeof(uint16_t));
        SO.allocate();
        if (try_op("softmax", [&] { softmax_last_dim(S, SO, ctx.stream()); }) < 0) return 1;
        std::vector<uint16_t> got(M * K);
        SO.copy_to_host(got.data(), got.size() * sizeof(uint16_t));
        std::vector<float> ref(M * K);
        for (int64_t m = 0; m < M; ++m) {
            float maxv = -INFINITY;
            for (int64_t k = 0; k < K; ++k) maxv = std::max(maxv, sm_f[m * K + k]);
            float sum = 0;
            for (int64_t k = 0; k < K; ++k) {
                float e = std::exp(sm_f[m * K + k] - maxv);
                ref[m * K + k] = e;
                sum += e;
            }
            for (int64_t k = 0; k < K; ++k) ref[m * K + k] /= sum;
        }
        rc |= check_close(got, ref, 5e-3f, "softmax");
    }

    if (rc) {
        std::cerr << "elementwise smoke failed rc=" << rc << "\n";
        return 2;
    }
    std::cout << "elementwise ok" << std::endl;
    return 0;
}
