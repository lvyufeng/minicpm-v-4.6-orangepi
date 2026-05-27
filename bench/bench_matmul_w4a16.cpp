// Bench W4A16 kernel vs fp16 cube matmul at gate_proj shape (1024 x 3584).
#include "minicpmv/acl_context.h"
#include "minicpmv/ops.h"
#include "minicpmv/tensor.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace minicpmv;
using clk = std::chrono::steady_clock;

struct PackedW4a16Weight {
    int tile_len{0};
    std::vector<int8_t> data;
};

static int choose_w4a16_tile_len(int N) {
    constexpr int BlockNum = 8;
    constexpr int MaxTileLen = 448;
    const int blockLen = (N + BlockNum - 1) / BlockNum;
    const int tiles = (blockLen + MaxTileLen - 1) / MaxTileLen;
    const int tileLen = ((blockLen + tiles - 1) / tiles + 15) / 16 * 16;
    return std::max(16, tileLen);
}

static PackedW4a16Weight pack_w4a16_weight(const std::vector<int8_t>& w, int K, int N, int forced_tile_len = 0) {
    constexpr int G = 128;
    constexpr int BlockNum = 8;
    const int TileLen = forced_tile_len > 0 ? forced_tile_len : choose_w4a16_tile_len(N);
    const int groups = K / G;
    const int blockLen = (N + BlockNum - 1) / BlockNum;
    const int tilesPerBlock = (blockLen + TileLen - 1) / TileLen;
    std::vector<int8_t> packed(static_cast<size_t>(groups) * BlockNum * tilesPerBlock * G * TileLen, 0);
    for (int g = 0; g < groups; ++g) {
        for (int b = 0; b < BlockNum; ++b) {
            const int blockStart = b * blockLen;
            const int blockValid = std::max(0, std::min(blockLen, N - blockStart));
            for (int t = 0; t < tilesPerBlock; ++t) {
                const int tileStart = t * TileLen;
                const int tileValid = std::max(0, std::min(TileLen, blockValid - tileStart));
                const size_t tileBase = static_cast<size_t>((((g * BlockNum + b) * tilesPerBlock + t) * G) * TileLen);
                for (int kk = 0; kk < G; ++kk) {
                    const int k = g * G + kk;
                    for (int i = 0; i < tileValid; ++i) {
                        packed[tileBase + static_cast<size_t>(kk * TileLen + i)] = w[static_cast<size_t>(k) * N + blockStart + tileStart + i];
                    }
                }
            }
        }
    }
    return PackedW4a16Weight{TileLen, std::move(packed)};
}

static double ms(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

int main(int argc, char** argv) {
    AclContext ctx(0);
    const int K = argc > 1 ? std::atoi(argv[1]) : 1024;
    const int N = argc > 2 ? std::atoi(argv[2]) : 3584;
    const int iters = argc > 3 ? std::atoi(argv[3]) : 100;
    constexpr int G = 128;
    const int NG = K / G;

    // --- W4A16 path ---
    Tensor x({1, K}, DType::Float16); x.allocate();
    Tensor w({K, N}, DType::Int8);    w.allocate();
    Tensor s({NG, N}, DType::Float16); s.allocate();
    Tensor out({1, N}, DType::Float16); out.allocate();

    std::vector<uint16_t> hx(K, 0x3c00);
    std::vector<int8_t> hw(K * N, 1);
    std::vector<uint16_t> hs(NG * N, 0x3c00);
    x.copy_from_host(hx.data(), hx.size() * 2);
    s.copy_from_host(hs.data(), hs.size() * 2);

    double best_w4a16_ms = 1e30;
    int best_tile = 0;
    const int default_tile = choose_w4a16_tile_len(N);
    const int tile_candidates[] = {64, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, default_tile};
    for (int tile : tile_candidates) {
        if (tile <= 0 || tile > 448 || tile % 16 != 0) continue;
        PackedW4a16Weight hpw = pack_w4a16_weight(hw, K, N, tile);
        w = Tensor({static_cast<int64_t>(hpw.data.size() / hpw.tile_len), hpw.tile_len}, DType::Int8);
        w.copy_from_host(hpw.data.data(), hpw.data.size());

        for (int i = 0; i < 5; ++i) matmul_w4a16(x, w, s, out, ctx.stream());
        aclrtSynchronizeStream(ctx.stream());

        auto t0 = clk::now();
        for (int i = 0; i < iters; ++i) matmul_w4a16(x, w, s, out, ctx.stream());
        aclrtSynchronizeStream(ctx.stream());
        auto t1 = clk::now();
        double w4a16_ms = ms(t0, t1) / iters;
        if (w4a16_ms < best_w4a16_ms) {
            best_w4a16_ms = w4a16_ms;
            best_tile = tile;
        }
        std::printf("W4A16  K=%d N=%d tile=%d  per_call_ms=%.3f  (%d iters)\n", K, N, tile, w4a16_ms, iters);
    }

    // --- fp16 cube reference ---
    Tensor xf({1, K}, DType::Float16); xf.allocate(); xf.copy_from_host(hx.data(), hx.size() * 2);
    Tensor wf({K, N}, DType::Float16); wf.allocate();
    // Set fp16 weight to all 1.0
    std::vector<uint16_t> hwf(K * N, 0x3c00);
    wf.copy_from_host(hwf.data(), hwf.size() * 2);
    Tensor outf({1, N}, DType::Float16); outf.allocate();

    for (int i = 0; i < 5; ++i) matmul_b_transposed(xf, wf, outf, ctx.stream());
    aclrtSynchronizeStream(ctx.stream());

    auto t0 = clk::now();
    for (int i = 0; i < iters; ++i) matmul_b_transposed(xf, wf, outf, ctx.stream());
    aclrtSynchronizeStream(ctx.stream());
    auto t1 = clk::now();
    double fp16_ms = ms(t0, t1) / iters;
    std::printf("fp16   K=%d N=%d  per_call_ms=%.3f  (%d iters)\n", K, N, fp16_ms, iters);
    std::printf("best W4A16 tile=%d per_call_ms=%.3f speedup vs fp16: %.2fx\n", best_tile, best_w4a16_ms, fp16_ms / best_w4a16_ms);
    return 0;
}
