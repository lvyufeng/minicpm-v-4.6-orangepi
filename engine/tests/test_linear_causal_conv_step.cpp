// Compares linear_causal_conv_step (vectorized last-row kernel) against the
// generic linear_causal_conv for the [4, C] decode-step shape. The two should
// agree to within fp16 precision on the last row of the generic kernel's
// [4, C] output.
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
    const int C = argc > 1 ? std::atoi(argv[1]) : 6144;
    const int K = 4;

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> xd(-1.0f, 1.0f);
    std::uniform_real_distribution<float> wd(-0.2f, 0.2f);

    std::vector<uint16_t> hX(K * C), hW_CK(C * K), hW_KC(K * C);
    for (auto& v : hX) v = f2h(xd(rng));
    for (int c = 0; c < C; ++c)
        for (int k = 0; k < K; ++k) {
            uint16_t v = f2h(wd(rng));
            hW_CK[c * K + k] = v;
            hW_KC[k * C + c] = v;
        }

    Tensor x({K, C}, DType::Float16); x.allocate(); x.copy_from_host(hX.data(), hX.size() * 2);
    Tensor wCK({C, K}, DType::Float16); wCK.allocate(); wCK.copy_from_host(hW_CK.data(), hW_CK.size() * 2);
    Tensor wKC({K, C}, DType::Float16); wKC.allocate(); wKC.copy_from_host(hW_KC.data(), hW_KC.size() * 2);

    Tensor ref_all({K, C}, DType::Float16); ref_all.allocate();
    linear_causal_conv(x, wCK, ref_all, ctx.stream());

    Tensor step_out({1, C}, DType::Float16); step_out.allocate();
    linear_causal_conv_step(x, wKC, step_out, ctx.stream());

    std::vector<uint16_t> hRef(K * C), hStep(C);
    ref_all.copy_to_host(hRef.data(), hRef.size() * 2);
    step_out.copy_to_host(hStep.data(), hStep.size() * 2);

    float maxAbs = 0.0f;
    int big = 0;
    for (int c = 0; c < C; ++c) {
        float r = h2f(hRef[(K - 1) * C + c]);
        float s = h2f(hStep[c]);
        float d = std::fabs(r - s);
        if (d > maxAbs) maxAbs = d;
        if (d > 5e-3f) ++big;
    }
    std::printf("C=%d  max_abs_diff=%.6f  large_diffs(>5e-3)=%d/%d\n",
                C, maxAbs, big, C);
    return 0;
}
