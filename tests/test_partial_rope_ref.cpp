#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

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
    constexpr int64_t B = 1;
    constexpr int64_t Hh = 8;
    constexpr int64_t T = 5;
    constexpr int64_t D = 256;
    constexpr int64_t Drot = 64;
    constexpr int64_t HalfRot = Drot / 2;

    std::vector<uint16_t> q(B * Hh * T * D);
    std::vector<uint16_t> k(B * Hh * T * D);
    std::vector<uint16_t> q_ref(q.size());
    std::vector<uint16_t> k_ref(k.size());
    std::vector<float> cos(T * HalfRot), sin(T * HalfRot);

    for (int64_t i = 0; i < static_cast<int64_t>(q.size()); ++i) {
        float v = (static_cast<float>((i * 7 + 3) % 29) / 29.0f - 0.5f) * 0.5f;
        q[i] = f2h(v);
        k[i] = f2h(v * 0.8f);
    }
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t i = 0; i < HalfRot; ++i) {
            float theta = static_cast<float>(t + 1) * 0.01f * static_cast<float>(i + 1);
            cos[t * HalfRot + i] = std::cos(theta);
            sin[t * HalfRot + i] = std::sin(theta);
        }
    }

    auto apply = [&](const std::vector<uint16_t>& in, std::vector<uint16_t>& out) {
        for (int64_t b = 0; b < B; ++b) {
            for (int64_t h = 0; h < Hh; ++h) {
                for (int64_t t = 0; t < T; ++t) {
                    int64_t base = ((b * Hh + h) * T + t) * D;
                    // first Drot dimensions only
                    for (int64_t i = 0; i < HalfRot; ++i) {
                        float x1 = h2f(in[base + i]);
                        float x2 = h2f(in[base + HalfRot + i]);
                        float c = cos[t * HalfRot + i];
                        float s = sin[t * HalfRot + i];
                        out[base + i] = f2h(x1 * c - x2 * s);
                        out[base + HalfRot + i] = f2h(x2 * c + x1 * s);
                    }
                    // untouched tail
                    for (int64_t i = Drot; i < D; ++i) {
                        out[base + i] = in[base + i];
                    }
                }
            }
        }
    };

    apply(q, q_ref);
    apply(k, k_ref);

    // Sanity checks: untouched tail and changed rotated region.
    int errors = 0;
    for (int64_t t = 0; t < T; ++t) {
        int64_t base = t * D;
        for (int64_t i = Drot; i < D; ++i) {
            if (q_ref[base + i] != q[base + i] || k_ref[base + i] != k[base + i]) {
                ++errors;
            }
        }
    }
    std::cout << "partial_rope_ref tail_errors=" << errors << " first_q="
              << h2f(q_ref[0]) << " first_k=" << h2f(k_ref[0]) << std::endl;
    return errors ? 1 : 0;
}
