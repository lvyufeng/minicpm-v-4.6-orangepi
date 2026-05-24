// Sanity-test vision encoder building blocks: conv2d, gelu(tanh), batch_matmul.
// Compares against a CPU reference for small shapes.
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

static int test_gelu() {
    AclContext ctx(0);
    const int N = 64;
    std::vector<uint16_t> hx(N);
    for (int i = 0; i < N; ++i) hx[i] = f2h(-3.0f + 6.0f * i / (N - 1));
    Tensor x({N}, DType::Float16); x.allocate(); x.copy_from_host(hx.data(), N * 2);
    Tensor y({N}, DType::Float16); y.allocate();
    gelu(x, /*tanh_approx=*/true, y, ctx.stream());
    std::vector<uint16_t> hy(N); y.copy_to_host(hy.data(), N * 2);
    float maxAbs = 0.0f;
    for (int i = 0; i < N; ++i) {
        float v = h2f(hx[i]);
        float ref = 0.5f * v * (1.0f + std::tanh(0.79788456f * (v + 0.044715f * v * v * v)));
        float diff = std::fabs(h2f(hy[i]) - ref);
        if (diff > maxAbs) maxAbs = diff;
    }
    std::printf("gelu(tanh) N=%d max_abs_diff=%.4f\n", N, maxAbs);
    return maxAbs < 5e-3f ? 0 : 1;
}

static int test_conv2d() {
    AclContext ctx(0);
    const int N = 1, Cin = 3, H = 32, W = 32, Cout = 8, KH = 14, KW = 14, Sh = 14, Sw = 14;
    const int Ho = (H - KH) / Sh + 1;  // 32 doesn't fit; let's pick a fitting size
    const int H_ok = 28, W_ok = 28;
    const int Ho2 = (H_ok - KH) / Sh + 1;  // (28-14)/14+1 = 2
    const int Wo2 = (W_ok - KW) / Sw + 1;
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(-0.3f, 0.3f);

    std::vector<uint16_t> hx(N * Cin * H_ok * W_ok), hw(Cout * Cin * KH * KW), hb(Cout);
    for (auto& v : hx) v = f2h(dist(rng));
    for (auto& v : hw) v = f2h(dist(rng));
    for (auto& v : hb) v = f2h(dist(rng));

    Tensor x({N, Cin, H_ok, W_ok}, DType::Float16); x.allocate(); x.copy_from_host(hx.data(), hx.size() * 2);
    Tensor w({Cout, Cin, KH, KW}, DType::Float16); w.allocate(); w.copy_from_host(hw.data(), hw.size() * 2);
    Tensor b({Cout}, DType::Float16); b.allocate(); b.copy_from_host(hb.data(), hb.size() * 2);
    Tensor y({N, Cout, Ho2, Wo2}, DType::Float16); y.allocate();
    conv2d(x, w, &b, {Sh, Sw}, {0, 0}, y, ctx.stream());

    std::vector<uint16_t> hy(N * Cout * Ho2 * Wo2); y.copy_to_host(hy.data(), hy.size() * 2);

    // CPU reference
    float maxAbs = 0.0f;
    for (int co = 0; co < Cout; ++co) {
        for (int oh = 0; oh < Ho2; ++oh) {
            for (int ow = 0; ow < Wo2; ++ow) {
                float acc = h2f(hb[co]);
                for (int ci = 0; ci < Cin; ++ci) {
                    for (int kh = 0; kh < KH; ++kh) {
                        for (int kw = 0; kw < KW; ++kw) {
                            int ih = oh * Sh + kh;
                            int iw = ow * Sw + kw;
                            acc += h2f(hx[ci * H_ok * W_ok + ih * W_ok + iw])
                                 * h2f(hw[co * Cin * KH * KW + ci * KH * KW + kh * KW + kw]);
                        }
                    }
                }
                float got = h2f(hy[co * Ho2 * Wo2 + oh * Wo2 + ow]);
                float diff = std::fabs(got - acc);
                if (diff > maxAbs) maxAbs = diff;
            }
        }
    }
    std::printf("conv2d N=%d Cin=%d Cout=%d H=%d KH=%d max_abs_diff=%.4f\n",
                N, Cin, Cout, H_ok, KH, maxAbs);
    return maxAbs < 0.5f ? 0 : 1;  // looser for accumulating reduction
}

static int test_batch_matmul() {
    AclContext ctx(0);
    const int B = 4, M = 8, K = 16, N = 8;
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> dist(-0.3f, 0.3f);
    std::vector<uint16_t> ha(B * M * K), hb(B * K * N);
    for (auto& v : ha) v = f2h(dist(rng));
    for (auto& v : hb) v = f2h(dist(rng));
    Tensor a({B, M, K}, DType::Float16); a.allocate(); a.copy_from_host(ha.data(), ha.size() * 2);
    Tensor b({B, K, N}, DType::Float16); b.allocate(); b.copy_from_host(hb.data(), hb.size() * 2);
    Tensor o({B, M, N}, DType::Float16); o.allocate();
    batch_matmul(a, b, o, ctx.stream());
    std::vector<uint16_t> hout(B * M * N); o.copy_to_host(hout.data(), hout.size() * 2);
    float maxAbs = 0.0f;
    for (int z = 0; z < B; ++z) for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += h2f(ha[z * M * K + m * K + k]) * h2f(hb[z * K * N + k * N + n]);
        }
        float got = h2f(hout[z * M * N + m * N + n]);
        float diff = std::fabs(got - acc);
        if (diff > maxAbs) maxAbs = diff;
    }
    std::printf("batch_matmul B=%d M=%d K=%d N=%d max_abs_diff=%.4f\n", B, M, K, N, maxAbs);
    return maxAbs < 0.1f ? 0 : 1;
}

int main() {
    int rc = 0;
    rc |= test_gelu();
    rc |= test_conv2d();
    rc |= test_batch_matmul();
    std::printf(rc == 0 ? "OK\n" : "FAIL\n");
    return rc;
}
