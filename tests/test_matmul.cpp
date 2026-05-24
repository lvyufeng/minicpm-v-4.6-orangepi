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

static int run_case(int64_t M, int64_t K, int64_t N) {
    AclContext ctx(0);
    std::vector<uint16_t> a_host(M * K), b_host(K * N);
    for (int64_t i = 0; i < M * K; ++i) {
        a_host[i] = f2h(static_cast<float>((i % 7) - 3) * 0.125f);
    }
    for (int64_t i = 0; i < K * N; ++i) {
        b_host[i] = f2h(static_cast<float>((i % 5) - 2) * 0.0625f);
    }
    Tensor A({M, K}, DType::Float16);
    Tensor B({K, N}, DType::Float16);
    Tensor C({M, N}, DType::Float16);
    A.copy_from_host(a_host.data(), a_host.size() * sizeof(uint16_t));
    B.copy_from_host(b_host.data(), b_host.size() * sizeof(uint16_t));
    C.allocate();

    try {
        matmul(A, B, C, ctx.stream());
    } catch (const std::exception& e) {
        std::cerr << "matmul M=" << M << " K=" << K << " N=" << N
                  << " failed: " << e.what()
                  << "\nrecent: " << aclGetRecentErrMsg() << std::endl;
        return 1;
    }

    std::vector<uint16_t> got_bits(M * N);
    C.copy_to_host(got_bits.data(), got_bits.size() * sizeof(uint16_t));

    int errors = 0;
    float max_abs = 0.0f;
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            float ref = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                ref += h2f(a_host[m * K + k]) * h2f(b_host[k * N + n]);
            }
            float got = h2f(got_bits[m * N + n]);
            float diff = std::fabs(got - ref);
            if (diff > max_abs) max_abs = diff;
            float tol = 1e-2f * static_cast<float>(K) + 1e-3f;
            if (diff > tol) {
                if (errors < 3) {
                    std::cerr << "mismatch m=" << m << " n=" << n
                              << " got=" << got << " ref=" << ref
                              << " diff=" << diff << std::endl;
                }
                ++errors;
            }
        }
    }
    std::cout << "matmul M=" << M << " K=" << K << " N=" << N
              << " errors=" << errors << " max_abs=" << max_abs << std::endl;
    return errors ? 2 : 0;
}

int main() {
    int rc = 0;
    rc |= run_case(4, 128, 64);
    rc |= run_case(89, 1024, 1024);
    rc |= run_case(1, 1024, 4096);
    return rc;
}
