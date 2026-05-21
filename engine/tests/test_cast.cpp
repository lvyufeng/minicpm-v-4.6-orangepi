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

static uint16_t f2bf(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    // round-to-nearest-even
    uint32_t lsb = (bits >> 16) & 1u;
    uint32_t round_bias = 0x7fffu + lsb;
    bits += round_bias;
    return static_cast<uint16_t>(bits >> 16);
}

static float bf2f(uint16_t b) {
    uint32_t bits = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

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
    constexpr int64_t N = 32;

    std::vector<float> ref(N);
    std::vector<uint16_t> bf_host(N), fp16_host(N);
    for (int64_t i = 0; i < N; ++i) {
        float v = (static_cast<float>((i * 7 + 3) % 23) / 23.0f - 0.5f) * 4.0f;
        ref[i] = v;
        bf_host[i] = f2bf(v);
        fp16_host[i] = f2h(v);
    }

    int rc = 0;

    // BF16 -> FP16
    {
        Tensor src({N}, DType::BFloat16);
        Tensor dst({N}, DType::Float16);
        src.copy_from_host(bf_host.data(), bf_host.size() * sizeof(uint16_t));
        dst.allocate();
        try {
            cast(src, dst, ctx.stream());
        } catch (const std::exception& e) {
            std::cerr << "bf16->fp16 cast failed: " << e.what()
                      << "\nrecent: " << aclGetRecentErrMsg() << std::endl;
            return 1;
        }
        std::vector<uint16_t> got(N);
        dst.copy_to_host(got.data(), got.size() * sizeof(uint16_t));
        float max_abs = 0.0f;
        int errors = 0;
        for (int64_t i = 0; i < N; ++i) {
            float g = h2f(got[i]);
            float r = bf2f(bf_host[i]);  // ground truth: original BF16-decoded value
            float d = std::fabs(g - r);
            if (d > max_abs) max_abs = d;
            if (d > 5e-3f) ++errors;
        }
        std::cout << "bf16->fp16 errors=" << errors << " max_abs=" << max_abs << "\n";
        if (errors) rc = 1;
    }

    // FP16 -> BF16
    {
        Tensor src({N}, DType::Float16);
        Tensor dst({N}, DType::BFloat16);
        src.copy_from_host(fp16_host.data(), fp16_host.size() * sizeof(uint16_t));
        dst.allocate();
        try {
            cast(src, dst, ctx.stream());
        } catch (const std::exception& e) {
            std::cerr << "fp16->bf16 cast failed: " << e.what()
                      << "\nrecent: " << aclGetRecentErrMsg() << std::endl;
            return 2;
        }
        std::vector<uint16_t> got(N);
        dst.copy_to_host(got.data(), got.size() * sizeof(uint16_t));
        float max_abs = 0.0f;
        int errors = 0;
        for (int64_t i = 0; i < N; ++i) {
            float g = bf2f(got[i]);
            float r = h2f(fp16_host[i]);
            float d = std::fabs(g - r);
            if (d > max_abs) max_abs = d;
            if (d > 5e-3f) ++errors;
        }
        std::cout << "fp16->bf16 errors=" << errors << " max_abs=" << max_abs << "\n";
        if (errors) rc = 2;
    }

    if (rc == 0) std::cout << "cast ok" << std::endl;
    return rc;
}
