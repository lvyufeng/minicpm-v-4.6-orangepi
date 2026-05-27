#include "minicpmv/quantized_weight.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace minicpmv {
namespace {

constexpr int64_t kGroupSize = 128;
constexpr int64_t kBlockNum = 8;
constexpr int64_t kMaxTileLen = 448;
constexpr int kAwqPackOrder[8] = {0, 2, 4, 6, 1, 3, 5, 7};

struct HostW4A16Weight {
    int64_t K{0};
    int64_t N{0};
    int64_t groups{0};
    std::vector<int8_t> w;
    std::vector<uint16_t> scales;
};

std::vector<int32_t> load_i32_tensor(const WeightsIndex& index, const std::string& name) {
    Tensor t = index.load_to_device(name);
    std::vector<int32_t> h(t.numel());
    t.copy_to_host(h.data(), h.size() * sizeof(int32_t));
    return h;
}

std::vector<uint16_t> load_f16_tensor(const WeightsIndex& index, const std::string& name) {
    Tensor t = index.load_to_device(name);
    std::vector<uint16_t> h(t.numel());
    t.copy_to_host(h.data(), h.size() * sizeof(uint16_t));
    return h;
}

int64_t choose_w4a16_tile_len(int64_t N) {
    const int64_t blockLen = (N + kBlockNum - 1) / kBlockNum;
    const int64_t tiles = (blockLen + kMaxTileLen - 1) / kMaxTileLen;
    const int64_t tileLen = ((blockLen + tiles - 1) / tiles + 15) / 16 * 16;
    return std::max<int64_t>(16, tileLen);
}

std::pair<int64_t, std::vector<int8_t>> pack_w4a16_weight(const std::vector<int8_t>& w, int64_t K, int64_t N) {
    const int64_t tileLen = choose_w4a16_tile_len(N);
    const int64_t groups = K / kGroupSize;
    const int64_t blockLen = (N + kBlockNum - 1) / kBlockNum;
    const int64_t tilesPerBlock = (blockLen + tileLen - 1) / tileLen;
    std::vector<int8_t> packed(static_cast<size_t>(groups * kBlockNum * tilesPerBlock * kGroupSize * tileLen), 0);
    for (int64_t g = 0; g < groups; ++g) {
        for (int64_t b = 0; b < kBlockNum; ++b) {
            const int64_t blockStart = b * blockLen;
            const int64_t blockValid = std::max<int64_t>(0, std::min<int64_t>(blockLen, N - blockStart));
            for (int64_t t = 0; t < tilesPerBlock; ++t) {
                const int64_t tileStart = t * tileLen;
                const int64_t tileValid = std::max<int64_t>(0, std::min<int64_t>(tileLen, blockValid - tileStart));
                const size_t tileBase = static_cast<size_t>((((g * kBlockNum + b) * tilesPerBlock + t) * kGroupSize) * tileLen);
                for (int64_t kk = 0; kk < kGroupSize; ++kk) {
                    const int64_t k = g * kGroupSize + kk;
                    for (int64_t i = 0; i < tileValid; ++i) {
                        packed[tileBase + static_cast<size_t>(kk * tileLen + i)] =
                            w[static_cast<size_t>(k) * N + blockStart + tileStart + i];
                    }
                }
            }
        }
    }
    return {tileLen, std::move(packed)};
}

HostW4A16Weight unpack_gptq_weight(const WeightsIndex& index, const std::string& base) {
    const auto& qinfo = index.at(base + ".qweight");
    const auto& zinfo = index.at(base + ".qzeros");
    const auto& ginfo = index.at(base + ".g_idx");
    const auto& sinfo = index.at(base + ".scales");
    if (qinfo.dtype != DType::Int32 || zinfo.dtype != DType::Int32 || ginfo.dtype != DType::Int32 || sinfo.dtype != DType::Float16) {
        throw std::runtime_error("unexpected GPTQ tensor dtypes for " + base);
    }
    if (qinfo.shape.size() != 2 || zinfo.shape.size() != 2 || ginfo.shape.size() != 1 || sinfo.shape.size() != 2) {
        throw std::runtime_error("unexpected GPTQ tensor ranks for " + base);
    }

    const int64_t kPack = qinfo.shape[0];
    const int64_t K = kPack * 8;
    const int64_t N = qinfo.shape[1];
    const int64_t groups = sinfo.shape[0];
    if (ginfo.shape[0] != K || groups * kGroupSize != K || sinfo.shape[1] != N ||
        zinfo.shape[0] != groups || zinfo.shape[1] * 8 != N) {
        throw std::runtime_error("unexpected GPTQ tensor shapes for " + base);
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
                    throw std::runtime_error("GPTQ g_idx out of range for " + base);
                }
                const uint32_t zword = static_cast<uint32_t>(qzeros[static_cast<size_t>(g) * (N / 8) + n / 8]);
                const int32_t zero = static_cast<int32_t>(((zword >> (4 * (n % 8))) & 0x0fu) + 1);
                const int32_t q = static_cast<int32_t>((qword >> (4 * nib)) & 0x0fu);
                unpacked[static_cast<size_t>(k) * N + n] = static_cast<int8_t>(q - zero);
            }
        }
    }

    return HostW4A16Weight{K, N, groups, std::move(unpacked), std::move(scales)};
}

HostW4A16Weight unpack_awq_weight(const WeightsIndex& index, const std::string& base) {
    const auto& qinfo = index.at(base + ".qweight");
    const auto& zinfo = index.at(base + ".qzeros");
    const auto& sinfo = index.at(base + ".scales");
    if (qinfo.dtype != DType::Int32 || zinfo.dtype != DType::Int32 || sinfo.dtype != DType::Float16) {
        throw std::runtime_error("unexpected AWQ tensor dtypes for " + base);
    }
    if (qinfo.shape.size() != 2 || zinfo.shape.size() != 2 || sinfo.shape.size() != 2) {
        throw std::runtime_error("unexpected AWQ tensor ranks for " + base);
    }

    const int64_t K = qinfo.shape[0];
    const int64_t N_packed = qinfo.shape[1];
    const int64_t N = N_packed * 8;
    const int64_t groups = sinfo.shape[0];
    if (zinfo.shape != std::vector<int64_t>{groups, N_packed} ||
        sinfo.shape != std::vector<int64_t>{groups, N} || groups * kGroupSize != K) {
        throw std::runtime_error("unexpected AWQ tensor shapes for " + base);
    }

    auto qweight = load_i32_tensor(index, base + ".qweight");
    auto qzeros = load_i32_tensor(index, base + ".qzeros");
    auto scales = load_f16_tensor(index, base + ".scales");
    std::vector<int8_t> unpacked(static_cast<size_t>(K) * N);
    for (int64_t k = 0; k < K; ++k) {
        const int64_t g = k / kGroupSize;
        for (int64_t j = 0; j < N_packed; ++j) {
            const uint32_t qword = static_cast<uint32_t>(qweight[static_cast<size_t>(k) * N_packed + j]);
            const uint32_t zword = static_cast<uint32_t>(qzeros[static_cast<size_t>(g) * N_packed + j]);
            for (int slot = 0; slot < 8; ++slot) {
                const int64_t n = j * 8 + kAwqPackOrder[slot];
                const int32_t q = static_cast<int32_t>((qword >> (4 * slot)) & 0x0fu);
                const int32_t z = static_cast<int32_t>((zword >> (4 * slot)) & 0x0fu);
                unpacked[static_cast<size_t>(k) * N + n] = static_cast<int8_t>(q - z);
            }
        }
    }
    return HostW4A16Weight{K, N, groups, std::move(unpacked), std::move(scales)};
}

}  // namespace

bool has_w4a16_quantized_weight(const WeightsIndex& index, const std::string& base) {
    return index.contains(base + ".qweight") &&
           index.contains(base + ".qzeros") &&
           index.contains(base + ".scales");
}

W4A16QuantizedWeight load_w4a16_quantized_weight(const WeightsIndex& index, const std::string& base) {
    HostW4A16Weight host = index.contains(base + ".g_idx") ? unpack_gptq_weight(index, base)
                                                            : unpack_awq_weight(index, base);
    if (host.K % kGroupSize != 0 || host.N % 128 != 0) {
        throw std::runtime_error("W4A16 quantized weight shape must have K multiple of 128 and N multiple of 128 for " + base);
    }

    auto packed = pack_w4a16_weight(host.w, host.K, host.N);
    W4A16QuantizedWeight out;
    out.K = host.K;
    out.N = host.N;
    out.group_size = kGroupSize;
    out.w_int8 = Tensor({static_cast<int64_t>(packed.second.size() / packed.first), packed.first}, DType::Int8);
    out.w_int8.copy_from_host(packed.second.data(), packed.second.size());
    out.scales = Tensor({host.groups, host.N}, DType::Float16);
    out.scales.copy_from_host(host.scales.data(), host.scales.size() * sizeof(uint16_t));
    return out;
}

}  // namespace minicpmv
