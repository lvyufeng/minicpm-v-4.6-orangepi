// Validate that AutoAWQ qweight/qzeros/scales unpack correctly through the
// existing tile-packed W4A16 custom matmul. Uses gate_proj of layer 0
// (K=1024, N=3584) from MiniCPM-V-4.6-AWQ/model.safetensors.
//
// AWQ layout (`version=gemm`, group_size=128, zero_point=true):
//   qweight  int32 [K, N/8]      -- packs 8 N-direction nibbles per int32
//   qzeros   int32 [K/128, N/8]  -- same packing for per-group zero points
//   scales   fp16  [K/128, N]
//   N column order is interleaved: stored col j contains AWQ N indices
//   (j*8 + AWQ_PACK_ORDER[i]) for i = 0..7.
//
// Reference: out[n] = sum_k x[k] * (q[k,n] - z[k/128,n]) * s[k/128,n]
#include "minicpmv/acl_context.h"
#include "minicpmv/ops.h"
#include "minicpmv/tensor.h"
#include "minicpmv/weights.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using namespace minicpmv;

// AWQ unpacks nibble slot i of each int32 word into N-column (j*8 + order[i]).
static constexpr int kAwqPackOrder[8] = {0, 2, 4, 6, 1, 3, 5, 7};

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

struct AwqHostWeight {
    int64_t K{0};
    int64_t N{0};
    int64_t groups{0};
    std::vector<int8_t> w;       // [K, N], values in [-15, 15] (q - zero)
    std::vector<uint16_t> scales; // [K/128, N]
};

static AwqHostWeight unpack_awq_weight(const WeightsIndex& index, const std::string& base) {
    const auto& qinfo = index.at(base + ".qweight");
    const auto& zinfo = index.at(base + ".qzeros");
    const auto& sinfo = index.at(base + ".scales");
    if (qinfo.dtype != DType::Int32 || zinfo.dtype != DType::Int32 || sinfo.dtype != DType::Float16) {
        throw std::runtime_error("unexpected AWQ tensor dtypes");
    }
    if (qinfo.shape.size() != 2 || zinfo.shape.size() != 2 || sinfo.shape.size() != 2) {
        throw std::runtime_error("unexpected AWQ tensor ranks");
    }
    const int64_t K = qinfo.shape[0];
    const int64_t N_packed = qinfo.shape[1];
    const int64_t N = N_packed * 8;
    const int64_t groups = sinfo.shape[0];
    if (zinfo.shape != std::vector<int64_t>{groups, N_packed} ||
        sinfo.shape != std::vector<int64_t>{groups, N} ||
        groups * 128 != K) {
        throw std::runtime_error("unexpected AWQ tensor shapes");
    }

    auto qweight = load_i32_tensor(index, base + ".qweight");
    auto qzeros = load_i32_tensor(index, base + ".qzeros");
    auto scales = load_f16_tensor(index, base + ".scales");

    std::vector<int8_t> unpacked(static_cast<size_t>(K) * N);
    for (int64_t k = 0; k < K; ++k) {
        const int64_t g = k / 128;
        for (int64_t j = 0; j < N_packed; ++j) {
            const uint32_t qword = static_cast<uint32_t>(qweight[static_cast<size_t>(k) * N_packed + j]);
            const uint32_t zword = static_cast<uint32_t>(qzeros[static_cast<size_t>(g) * N_packed + j]);
            for (int slot = 0; slot < 8; ++slot) {
                const int order = kAwqPackOrder[slot];
                const int64_t n = j * 8 + order;
                const int32_t q = static_cast<int32_t>((qword >> (4 * slot)) & 0x0fu);
                const int32_t z = static_cast<int32_t>((zword >> (4 * slot)) & 0x0fu);
                unpacked[static_cast<size_t>(k) * N + n] = static_cast<int8_t>(q - z);
            }
        }
    }
    AwqHostWeight out;
    out.K = K;
    out.N = N;
    out.groups = groups;
    out.w = std::move(unpacked);
    out.scales = std::move(scales);
    return out;
}

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

int main() {
    AclContext ctx(0);
    constexpr int K = 1024;
    constexpr int N = 3584;
    constexpr int G = 128;
    constexpr int NG = K / G;

    WeightsIndex index("MiniCPM-V-4.6-AWQ/model.safetensors");
    const std::string base = "model.language_model.layers.0.mlp.gate_proj";
    AwqHostWeight awq = unpack_awq_weight(index, base);
    if (awq.K != K || awq.N != N || awq.groups != NG) {
        throw std::runtime_error("unexpected gate_proj AWQ shape");
    }

    std::vector<uint16_t> hx(K, f2h(0.05f));
    PackedW4a16Weight packed = pack_w4a16_weight(awq.w, K, N);

    Tensor x({1, K}, DType::Float16); x.allocate(); x.copy_from_host(hx.data(), hx.size() * 2);
    Tensor w({static_cast<int64_t>(packed.data.size() / packed.tile_len), packed.tile_len}, DType::Int8);
    w.allocate(); w.copy_from_host(packed.data.data(), packed.data.size());
    Tensor s({NG, N}, DType::Float16); s.allocate(); s.copy_from_host(awq.scales.data(), awq.scales.size() * 2);
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
                partial += h2f(hx[k]) * static_cast<float>(awq.w[k * N + n]);
            }
            ref[n] += partial * h2f(awq.scales[g * N + n]);
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
    std::printf("AWQ gate_proj K=%d N=%d  max_abs_diff=%.4f  mean_abs_diff=%.4f  large(>0.1)=%d/%d\n",
                K, N, maxAbs, sumAbs / N, big, N);
    return big == 0 ? 0 : 1;
}
