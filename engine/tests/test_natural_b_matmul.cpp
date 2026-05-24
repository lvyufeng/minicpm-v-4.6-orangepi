// Test: aclnnMm with NATURAL [K, N] B layout must match aclnnMm with
// transposed view of [N, K] B layout for the same input data. Compares
// matmul_b_transposed(a, b_legacy) vs matmul_b_transposed(a, b_transposed)
// where b_transposed is the [K, N] permutation of b_legacy.
#include "minicpmv/acl_context.h"
#include "minicpmv/ops.h"
#include "minicpmv/tensor.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
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
    if (exp == 0) { out = sign; }
    else if (exp == 31) { out = sign | 0x7f800000u | (mant << 13); }
    else { out = sign | ((exp + 127 - 15) << 23) | (mant << 13); }
    float f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}

int main(int argc, char** argv) {
    AclContext ctx(0);
    const int M = argc > 1 ? std::atoi(argv[1]) : 64;
    const int K = argc > 2 ? std::atoi(argv[2]) : 1024;
    const int N = argc > 3 ? std::atoi(argv[3]) : 3584;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.3f, 0.3f);

    std::vector<uint16_t> hA(M * K);
    std::vector<uint16_t> hB_NK(N * K);  // [N, K] layout
    std::vector<uint16_t> hB_KN(K * N);  // [K, N] layout
    for (auto& v : hA) v = f2h(dist(rng));
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k) {
            uint16_t v = f2h(dist(rng));
            hB_NK[n * K + k] = v;
            hB_KN[k * N + n] = v;
        }

    Tensor a({M, K}, DType::Float16); a.allocate(); a.copy_from_host(hA.data(), hA.size() * 2);
    Tensor b_legacy({N, K}, DType::Float16); b_legacy.allocate(); b_legacy.copy_from_host(hB_NK.data(), hB_NK.size() * 2);
    Tensor b_natural({K, N}, DType::Float16); b_natural.allocate(); b_natural.copy_from_host(hB_KN.data(), hB_KN.size() * 2);
    Tensor o_legacy({M, N}, DType::Float16); o_legacy.allocate();
    Tensor o_natural({M, N}, DType::Float16); o_natural.allocate();

    matmul_b_transposed(a, b_legacy, o_legacy, ctx.stream());
    matmul_b_transposed(a, b_natural, o_natural, ctx.stream());

    std::vector<uint16_t> hL(M * N), hNat(M * N);
    o_legacy.copy_to_host(hL.data(), hL.size() * 2);
    o_natural.copy_to_host(hNat.data(), hNat.size() * 2);

    float maxAbs = 0.0f;
    int big = 0;
    for (size_t i = 0; i < hL.size(); ++i) {
        float d = std::fabs(h2f(hL[i]) - h2f(hNat[i]));
        if (d > maxAbs) maxAbs = d;
        if (d > 0.05f) ++big;
    }
    std::printf("M=%d K=%d N=%d  max_abs_diff=%.6f  large_diffs(>0.05)=%d/%zu\n",
                M, K, N, maxAbs, big, hL.size());
    return 0;
}
