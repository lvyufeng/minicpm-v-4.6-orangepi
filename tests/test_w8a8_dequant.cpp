#include "minicpmv/acl_context.h"
#include "minicpmv/ops.h"
#include "minicpmv/tensor.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
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
    if (exp == 0) out = sign;
    else if (exp == 31) out = sign | 0x7f800000u | (mant << 13);
    else out = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    float f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}

int main() {
    AclContext ctx(0);
    constexpr int N = 3584;
    std::vector<int32_t> hacc(N);
    std::vector<uint16_t> hws(N);
    for (int i = 0; i < N; ++i) {
        hacc[i] = i - 1024;
        hws[i] = f2h(0.01f + static_cast<float>(i % 7) * 0.001f);
    }
    std::vector<uint16_t> hxs(1, f2h(0.02f));

    Tensor acc({1, N}, DType::Int32); acc.copy_from_host(hacc.data(), hacc.size() * sizeof(int32_t));
    Tensor xs({1}, DType::Float16); xs.copy_from_host(hxs.data(), sizeof(uint16_t));
    Tensor ws({N}, DType::Float16); ws.copy_from_host(hws.data(), hws.size() * sizeof(uint16_t));
    Tensor out({1, N}, DType::Float16); out.allocate();
    w8a8_dequant(acc, xs, ws, out, ctx.stream());

    std::vector<uint16_t> ho(N);
    out.copy_to_host(ho.data(), ho.size() * sizeof(uint16_t));
    float max_abs = 0.0f;
    for (int i = 0; i < N; ++i) {
        const float ref = static_cast<float>(hacc[i]) * h2f(hxs[0]) * h2f(hws[i]);
        const float got = h2f(ho[i]);
        max_abs = std::max(max_abs, std::fabs(ref - got));
    }
    std::printf("W8A8 dequant max_abs_diff=%.6f\n", max_abs);
    return max_abs < 0.02f ? 0 : 1;
}
