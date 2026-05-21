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
    constexpr int64_t H = 64;
    constexpr int64_t V = 128;

    std::vector<uint16_t> hidden_host(N * H);
    std::vector<uint16_t> weight_host(H * V);
    for (int64_t i = 0; i < N * H; ++i) {
        float v = static_cast<float>((i * 13 + 7) % 31) / 31.0f - 0.5f;
        hidden_host[i] = f2h(v * 0.25f);
    }
    for (int64_t i = 0; i < H * V; ++i) {
        float v = static_cast<float>((i * 17 + 3) % 29) / 29.0f - 0.5f;
        weight_host[i] = f2h(v * 0.5f);
    }

    Tensor hidden({N, H}, DType::Float16);
    Tensor lm_weight({H, V}, DType::Float16);
    Tensor logits({N, V}, DType::Float16);
    Tensor tokens({N}, DType::Int64);

    hidden.copy_from_host(hidden_host.data(), hidden_host.size() * sizeof(uint16_t));
    lm_weight.copy_from_host(weight_host.data(), weight_host.size() * sizeof(uint16_t));
    logits.allocate();
    tokens.allocate();

    try {
        matmul(hidden, lm_weight, logits, ctx.stream());
    } catch (const std::exception& e) {
        std::cerr << "matmul failed: " << e.what()
                  << "\nrecent: " << aclGetRecentErrMsg() << std::endl;
        return 1;
    }

    try {
        argmax_last_dim(logits, tokens, ctx.stream());
    } catch (const std::exception& e) {
        std::cerr << "argmax failed: " << e.what()
                  << "\nrecent: " << aclGetRecentErrMsg() << std::endl;
        return 2;
    }

    std::vector<int64_t> got_tokens(N);
    tokens.copy_to_host(got_tokens.data(), got_tokens.size() * sizeof(int64_t));

    std::vector<int64_t> ref_tokens(N);
    for (int64_t n = 0; n < N; ++n) {
        float best_val = -INFINITY;
        int64_t best_v = -1;
        for (int64_t v = 0; v < V; ++v) {
            float acc = 0.0f;
            for (int64_t h = 0; h < H; ++h) {
                acc += h2f(hidden_host[n * H + h]) * h2f(weight_host[h * V + v]);
            }
            if (acc > best_val) { best_val = acc; best_v = v; }
        }
        ref_tokens[n] = best_v;
    }

    int errors = 0;
    for (int64_t n = 0; n < N; ++n) {
        if (got_tokens[n] != ref_tokens[n]) {
            std::cerr << "row " << n << " got=" << got_tokens[n]
                      << " ref=" << ref_tokens[n] << std::endl;
            ++errors;
        }
    }
    if (errors) {
        std::cerr << errors << " mismatches" << std::endl;
        return 3;
    }
    std::cout << "lm_head pipeline ok N=" << N << " H=" << H << " V=" << V
              << " tokens=";
    for (auto t : got_tokens) std::cout << t << ",";
    std::cout << std::endl;
    return 0;
}
