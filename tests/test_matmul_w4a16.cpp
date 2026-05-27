// Compare matmul_w4a16 (NPU custom kernel) against a CPU fp16 reference for
// the gate_proj shape K=1024 N=3584. Uses random weights by default, and can
// optionally unpack real GPTQ weights from MiniCPM-V-4.6-GPTQ/model.safetensors.
//
// Reference: out[n] = sum_k x[k] * (w_int8[k,n] as fp16) * scales[k/G, n]
#include "minicpmv/acl_context.h"
#include "minicpmv/ops.h"
#include "minicpmv/tensor.h"
#include "minicpmv/weights.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
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
            while ((mant & 0x400u) == 0) { mant <<= 1; --e; }
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

struct W4a16HostWeight {
    int64_t K{0};
    int64_t N{0};
    int64_t groups{0};
    std::vector<int8_t> w;
    std::vector<uint16_t> scales;
};

struct PackedW4a16Weight {
    int64_t tile_len{0};
    std::vector<int8_t> data;
};

static int64_t choose_w4a16_tile_len(int64_t N) {
    constexpr int64_t BlockNum = 8;
    constexpr int64_t MaxTileLen = 448;
    const int64_t blockLen = (N + BlockNum - 1) / BlockNum;
    const int64_t tiles = (blockLen + MaxTileLen - 1) / MaxTileLen;
    const int64_t tileLen = ((blockLen + tiles - 1) / tiles + 15) / 16 * 16;
    return std::max<int64_t>(16, tileLen);
}

static PackedW4a16Weight pack_w4a16_weight(const std::vector<int8_t>& w, int64_t K, int64_t N) {
    constexpr int64_t G = 128;
    constexpr int64_t BlockNum = 8;
    const int64_t TileLen = choose_w4a16_tile_len(N);
    const int64_t groups = K / G;
    const int64_t blockLen = (N + BlockNum - 1) / BlockNum;
    const int64_t tilesPerBlock = (blockLen + TileLen - 1) / TileLen;
    std::vector<int8_t> packed(static_cast<size_t>(groups * BlockNum * tilesPerBlock * G * TileLen), 0);
    for (int64_t g = 0; g < groups; ++g) {
        for (int64_t b = 0; b < BlockNum; ++b) {
            const int64_t blockStart = b * blockLen;
            const int64_t blockValid = std::max<int64_t>(0, std::min<int64_t>(blockLen, N - blockStart));
            for (int64_t t = 0; t < tilesPerBlock; ++t) {
                const int64_t tileStart = t * TileLen;
                const int64_t tileValid = std::max<int64_t>(0, std::min<int64_t>(TileLen, blockValid - tileStart));
                const size_t tileBase = static_cast<size_t>((((g * BlockNum + b) * tilesPerBlock + t) * G) * TileLen);
                for (int64_t kk = 0; kk < G; ++kk) {
                    const int64_t k = g * G + kk;
                    for (int64_t i = 0; i < tileValid; ++i) {
                        packed[tileBase + static_cast<size_t>(kk * TileLen + i)] = w[static_cast<size_t>(k) * N + blockStart + tileStart + i];
                    }
                }
            }
        }
    }
    return PackedW4a16Weight{TileLen, std::move(packed)};
}

static std::vector<int32_t> load_i32_tensor(const WeightsIndex& index, const std::string& name) {
    Tensor t = index.load_to_device(name);
    std::vector<int32_t> h(t.numel());
    t.copy_to_host(h.data(), h.size() * sizeof(int32_t));
    return h;
}

static std::vector<uint16_t> load_f16_tensor(const WeightsIndex& index, const std::string& name) {
    Tensor t = index.load_to_device(name);
    std::vector<uint16_t> h(t.numel());
    t.copy_to_host(h.data(), h.size() * sizeof(uint16_t));
    return h;
}

static W4a16HostWeight unpack_gptq_qweight(const WeightsIndex& index, const std::string& base) {
    const auto& qinfo = index.at(base + ".qweight");
    const auto& zinfo = index.at(base + ".qzeros");
    const auto& ginfo = index.at(base + ".g_idx");
    const auto& sinfo = index.at(base + ".scales");

    if (qinfo.dtype != DType::Int32 || zinfo.dtype != DType::Int32 || ginfo.dtype != DType::Int32 || sinfo.dtype != DType::Float16) {
        throw std::runtime_error("unexpected GPTQ tensor dtypes");
    }
    if (qinfo.shape.size() != 2 || zinfo.shape.size() != 2 || ginfo.shape.size() != 1 || sinfo.shape.size() != 2) {
        throw std::runtime_error("unexpected GPTQ tensor ranks");
    }

    const int64_t kPack = qinfo.shape[0];
    const int64_t K = kPack * 8;
    const int64_t N = qinfo.shape[1];
    const int64_t groups = sinfo.shape[0];
    if (ginfo.shape[0] != K || groups * 128 != K || sinfo.shape[1] != N || zinfo.shape[0] != groups || zinfo.shape[1] * 8 != N) {
        throw std::runtime_error("unexpected GPTQ tensor shapes");
    }

    std::vector<int32_t> qweight = load_i32_tensor(index, base + ".qweight");
    std::vector<int32_t> qzeros = load_i32_tensor(index, base + ".qzeros");
    std::vector<int32_t> g_idx = load_i32_tensor(index, base + ".g_idx");
    std::vector<uint16_t> scales = load_f16_tensor(index, base + ".scales");
    std::vector<int8_t> unpacked(static_cast<size_t>(K) * N);

    for (int64_t k0 = 0; k0 < kPack; ++k0) {
        for (int64_t n = 0; n < N; ++n) {
            const uint32_t qword = static_cast<uint32_t>(qweight[static_cast<size_t>(k0) * N + n]);
            for (int64_t nib = 0; nib < 8; ++nib) {
                const int64_t k = k0 * 8 + nib;
                const int64_t g = g_idx[static_cast<size_t>(k)];
                if (g < 0 || g >= groups) {
                    throw std::runtime_error("GPTQ g_idx out of range");
                }
                const uint32_t zword = static_cast<uint32_t>(qzeros[static_cast<size_t>(g) * (N / 8) + n / 8]);
                const int32_t zero = static_cast<int32_t>(((zword >> (4 * (n % 8))) & 0x0fu) + 1);
                const int32_t q = static_cast<int32_t>((qword >> (4 * nib)) & 0x0fu);
                unpacked[static_cast<size_t>(k) * N + n] = static_cast<int8_t>(q - zero);
            }
        }
    }

    W4a16HostWeight out;
    out.K = K;
    out.N = N;
    out.groups = groups;
    out.w = std::move(unpacked);
    out.scales = std::move(scales);
    return out;
}

int main(int argc, char** argv) {
    AclContext ctx(0);
    constexpr int K = 1024;
    constexpr int N = 3584;
    constexpr int G = 128;
    constexpr int NG = K / G;  // 8
    const bool use_gptq = argc > 1 && std::string(argv[1]) == "--gptq";

    std::mt19937 rng(0);
    std::uniform_real_distribution<float> xd(-0.3f, 0.3f);
    std::uniform_int_distribution<int> wd(-8, 7);
    std::uniform_real_distribution<float> sd(0.001f, 0.02f);

    std::vector<uint16_t> hx(K);
    for (auto& v : hx) v = f2h(xd(rng));

    std::vector<int8_t> hw(K * N);
    for (auto& v : hw) v = static_cast<int8_t>(wd(rng));

    std::vector<uint16_t> hs(NG * N);
    for (auto& v : hs) v = f2h(sd(rng));

    if (use_gptq) {
        WeightsIndex index("MiniCPM-V-4.6-GPTQ/model.safetensors");
        const std::string base = "model.language_model.layers.0.mlp.gate_proj";
        W4a16HostWeight gptq = unpack_gptq_qweight(index, base);
        if (gptq.K != K || gptq.N != N || gptq.groups != NG) {
            throw std::runtime_error("unexpected gate_proj GPTQ shape");
        }
        hw = std::move(gptq.w);
        hs = std::move(gptq.scales);
    }

    PackedW4a16Weight packed_w = pack_w4a16_weight(hw, K, N);
    Tensor x({1, K}, DType::Float16); x.allocate(); x.copy_from_host(hx.data(), hx.size() * 2);
    Tensor w({static_cast<int64_t>(packed_w.data.size() / packed_w.tile_len), packed_w.tile_len}, DType::Int8); w.allocate(); w.copy_from_host(packed_w.data.data(), packed_w.data.size());
    Tensor s({NG, N}, DType::Float16); s.allocate(); s.copy_from_host(hs.data(), hs.size() * 2);
    Tensor out({1, N}, DType::Float16); out.allocate();

    matmul_w4a16(x, w, s, out, ctx.stream());

    std::vector<uint16_t> ho(N);
    out.copy_to_host(ho.data(), ho.size() * 2);

    std::vector<float> ref(N, 0.0f);
    for (int g = 0; g < NG; ++g) {
        for (int n = 0; n < N; ++n) {
            float partial = 0.0f;
            for (int kk = 0; kk < G; ++kk) {
                int k = g * G + kk;
                partial += h2f(hx[k]) * static_cast<float>(hw[k * N + n]);
            }
            ref[n] += partial * h2f(hs[g * N + n]);
        }
    }

    float maxAbs = 0.0f, sumAbs = 0.0f;
    int big = 0;
    for (int n = 0; n < N; ++n) {
        float r = ref[n];
        float v = h2f(ho[n]);
        float d = std::fabs(r - v);
        if (d > maxAbs) maxAbs = d;
        sumAbs += d;
        if (d > 0.1f) ++big;
    }
    std::printf("K=%d N=%d  max_abs_diff=%.4f  mean_abs_diff=%.4f  large(>0.1)=%d/%d\n",
                K, N, maxAbs, sumAbs / N, big, N);
    return big == 0 ? 0 : 1;
}
