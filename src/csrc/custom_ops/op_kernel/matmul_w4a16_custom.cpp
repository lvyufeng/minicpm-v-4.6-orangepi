#include "kernel_operator.h"

using namespace AscendC;

// GPTQ W4A16 matmul, M=1 fast path, with tile-packed int8 weight.
//
// The host unpacks GPTQ qweight once and repacks it by (group, core, tile, k).
// Each packed tile is a contiguous [128, tileLen] int8 block, so the kernel can
// fetch a whole group/tile with one DMA instead of 128 strided row copies.
//
// Math: out[n] = sum_g scales[g, n] * sum_{k in group g} x[k] * w[k, n]
//   with w in int8 [-8, 7]. K must be multiple of group_size (128).
//
// Inputs (M=1, MiniCPM-V GPTQ shapes; group_size G = 128):
//   x       fp16  [1, K]
//   w       int8  [K/128 * 8 * tilesPerBlock * 128, tileLen]
//   scales  fp16  [K/G, N]
// Output:
//   out     fp16  [1, N]
//
// Block partition: 8 AICore blocks split N. Each block walks its packed tiles.

class KernelMatmulW4a16Custom {
public:
    __aicore__ inline KernelMatmulW4a16Custom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR scales, GM_ADDR out,
                                uint32_t K, uint32_t N, uint32_t blockLen,
                                uint32_t tileLen, uint32_t tilesPerBlock, uint32_t groupSize) {
        const uint32_t bIdx = GetBlockIdx();
        this->K = K;
        this->N = N;
        this->blockLen = blockLen;
        this->tileLen = tileLen;
        this->tilesPerBlock = tilesPerBlock;
        this->groupSize = groupSize;
        this->numGroups = K / groupSize;
        this->nStart = bIdx * blockLen;
        this->nBlockLen = N > nStart ? N - nStart : 0;
        if (nBlockLen > blockLen) nBlockLen = blockLen;
        this->blockIdx = bIdx;

        xGm.SetGlobalBuffer((__gm__ half*)x, K);
        wGm.SetGlobalBuffer((__gm__ int8_t*)w, static_cast<uint64_t>(numGroups) * 8 * tilesPerBlock * groupSize * tileLen);
        scGm.SetGlobalBuffer((__gm__ half*)scales, numGroups * N);
        outGm.SetGlobalBuffer((__gm__ half*)out, N);

        // UB layout (worst-case N_TILE = 448):
        //   x         K     fp16 = 2048 B at K=1024
        //   wInt8     G*N_T int8 = 128*448 = 56 KB
        //   wFp16     G*N_T fp16 = 112 KB
        //   scaleRow  N_T   fp16 = 896 B
        //   partial   N_T   fp16 = 896 B
        //   acc       N_T   fp16 = 896 B
        // Fits in the 192 KB UB and cuts the number of small strided weight DMAs.
        pipe.InitBuffer(xBuf, K * sizeof(half));
        pipe.InitBuffer(wInt8Buf, groupSize * N_TILE * sizeof(int8_t));
        pipe.InitBuffer(wFp16Buf, groupSize * N_TILE * sizeof(half));
        pipe.InitBuffer(scaleBuf, N_TILE * sizeof(half));
        pipe.InitBuffer(partialBuf, N_TILE * sizeof(half));
        pipe.InitBuffer(accBuf, N_TILE * sizeof(half));
    }

    __aicore__ inline void Process() {
        if (nBlockLen == 0) return;
        LocalTensor<half> x = xBuf.Get<half>();
        DataCopy(x, xGm, K);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

        for (uint32_t nOff = 0; nOff < nBlockLen; nOff += tileLen) {
            uint32_t nTile = nBlockLen - nOff;
            if (nTile > tileLen) nTile = tileLen;
            uint32_t nTileAligned = ((nTile + ALIGN_FP16 - 1) / ALIGN_FP16) * ALIGN_FP16;
            const uint32_t nAbs = nStart + nOff;
            ComputeNTile(x, nAbs, nTileAligned, nOff / tileLen);
        }
    }

private:
    __aicore__ inline void ComputeNTile(LocalTensor<half>& x, uint32_t nAbs, uint32_t nTile, uint32_t tileIdx) {
        LocalTensor<half> acc = accBuf.Get<half>();
        LocalTensor<half> partial = partialBuf.Get<half>();
        LocalTensor<half> scale = scaleBuf.Get<half>();
        LocalTensor<half> wFp16 = wFp16Buf.Get<half>();
        LocalTensor<int8_t> wInt8 = wInt8Buf.Get<int8_t>();

        Duplicate(acc, (half)0, nTile);
        PipeBarrier<PIPE_V>();

        for (uint32_t g = 0; g < numGroups; ++g) {
            // 1) DMA: packed [128, tileLen] block for this group/core/tile.
            const uint64_t tileBase = ((((static_cast<uint64_t>(g) * 8 + blockIdx) * tilesPerBlock) + tileIdx)
                                      * groupSize) * tileLen;
            DataCopy(wInt8, wGm[tileBase], groupSize * tileLen);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

            // 2) Cast int8 -> fp16 over the whole tile.
            Cast(wFp16, wInt8, RoundMode::CAST_NONE, groupSize * tileLen);
            PipeBarrier<PIPE_V>();

            // 3) partial = sum_{k=0..127} x[g*G + k] * wFp16[k * tileLen + :nTile]
            Duplicate(partial, (half)0, nTile);
            PipeBarrier<PIPE_V>();
            for (uint32_t k = 0; k < groupSize; ++k) {
                half xk = x.GetValue(g * groupSize + k);
                Axpy(partial, wFp16[k * tileLen], xk, nTile);
                PipeBarrier<PIPE_V>();
            }

            // 4) Multiply by group scale, accumulate.
            DataCopy(scale, scGm[g * N + nAbs], nTile);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Mul(partial, partial, scale, nTile);
            PipeBarrier<PIPE_V>();
            Add(acc, acc, partial, nTile);
            PipeBarrier<PIPE_V>();
        }

        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(outGm[nAbs], acc, nTile);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }

    static constexpr uint32_t N_TILE = 448;
    static constexpr uint32_t ALIGN_FP16 = 16;

    uint32_t K{0};
    uint32_t N{0};
    uint32_t blockLen{0};
    uint32_t tileLen{0};
    uint32_t tilesPerBlock{0};
    uint32_t blockIdx{0};
    uint32_t groupSize{128};
    uint32_t numGroups{0};
    uint32_t nStart{0};
    uint32_t nBlockLen{0};

    GlobalTensor<half> xGm;
    GlobalTensor<int8_t> wGm;
    GlobalTensor<half> scGm;
    GlobalTensor<half> outGm;

    TPipe pipe;
    TBuf<TPosition::VECCALC> xBuf;
    TBuf<TPosition::VECCALC> wInt8Buf;
    TBuf<TPosition::VECCALC> wFp16Buf;
    TBuf<TPosition::VECCALC> scaleBuf;
    TBuf<TPosition::VECCALC> partialBuf;
    TBuf<TPosition::VECCALC> accBuf;
};

extern "C" __global__ __aicore__ void matmul_w4a16_custom(GM_ADDR x,
                                                            GM_ADDR w,
                                                            GM_ADDR scales,
                                                            GM_ADDR out,
                                                            GM_ADDR workspace,
                                                            GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelMatmulW4a16Custom op;
    op.Init(x, w, scales, out, tiling_data.K, tiling_data.N, tiling_data.blockLen,
            tiling_data.tileLen, tiling_data.tilesPerBlock, tiling_data.groupSize);
    op.Process();
}
