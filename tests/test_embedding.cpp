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
        if (mant == 0) {
            out = sign;
        } else {
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

    constexpr int64_t V = 1024;
    constexpr int64_t H = 128;
    constexpr int64_t N = 8;

    std::vector<uint16_t> weight_host(V * H);
    for (int64_t v = 0; v < V; ++v) {
        for (int64_t h = 0; h < H; ++h) {
            weight_host[v * H + h] = f2h(static_cast<float>(v) + static_cast<float>(h) * 0.01f);
        }
    }
    std::vector<int32_t> ids = {3, 0, 1023, 17, 256, 511, 1, 768};

    Tensor weight({V, H}, DType::Float16);
    Tensor out({N, H}, DType::Float16);
    out.allocate();
    weight.copy_from_host(weight_host.data(), weight_host.size() * sizeof(uint16_t));

    try {
        embedding_lookup(weight, ids, out, ctx.stream());
    } catch (const std::exception& e) {
        std::cerr << "embedding_lookup failed: " << e.what()
                  << "\nrecent: " << aclGetRecentErrMsg() << std::endl;
        return 1;
    }

    std::vector<uint16_t> got_bits(N * H);
    out.copy_to_host(got_bits.data(), got_bits.size() * sizeof(uint16_t));

    int errors = 0;
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t h = 0; h < H; ++h) {
            float got = h2f(got_bits[n * H + h]);
            float want = static_cast<float>(ids[n]) + static_cast<float>(h) * 0.01f;
            if (std::fabs(got - want) > 0.5f) {
                if (errors < 5) {
                    std::cerr << "mismatch at row " << n << " col " << h
                              << " id=" << ids[n] << " got=" << got
                              << " want=" << want << std::endl;
                }
                ++errors;
            }
        }
    }
    if (errors) {
        std::cerr << errors << " elements mismatched" << std::endl;
        return 2;
    }
    std::cout << "embedding ok: V=" << V << " H=" << H << " N=" << N << std::endl;
    return 0;
}
