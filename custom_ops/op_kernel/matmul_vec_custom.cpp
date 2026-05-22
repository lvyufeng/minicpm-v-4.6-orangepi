#include "kernel_operator.h"

using namespace AscendC;

// out[n] = sum_k x[k] * W[n, k] for x fp16 [K], W fp16 [N, K], out fp16 [N].
// Equivalent to matmul_b_transposed at M=1. Each block handles N/BlockDim rows.
// Within a block, rows are processed in M_TILE-sized tiles loaded to UB.

class KernelMatmulVecCustom {
public:
    __aicore__ inline KernelMatmulVecCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR out,
                                uint32_t N, uint32_t K) {
        this->K = K;
        const uint32_t blocks = GetBlockNum();
        const uint32_t bIdx   = GetBlockIdx();
        const uint32_t base   = (N + blocks - 1) / blocks;
        uint32_t start = bIdx * base;
        uint32_t end   = start + base;
        if (end   > N) end   = N;
        if (start > N) start = N;
        this->nStart  = start;
        this->nLen    = end - start;

        xGm.SetGlobalBuffer((__gm__ half*)x, K);
        wGm.SetGlobalBuffer((__gm__ half*)w + static_cast<uint64_t>(start) * K, static_cast<uint64_t>(nLen) * K);
        outGm.SetGlobalBuffer((__gm__ half*)out + start, nLen);

        pipe.InitBuffer(xFp16Buf,    K * sizeof(half));
        pipe.InitBuffer(xFp32Buf,    K * sizeof(float));
        pipe.InitBuffer(wTileFp16Buf, M_TILE * K * sizeof(half));
        pipe.InitBuffer(wRowFp32Buf, K * sizeof(float));
        pipe.InitBuffer(prodFp32Buf, K * sizeof(float));
        pipe.InitBuffer(outFp32Buf,  M_TILE * sizeof(float));
        pipe.InitBuffer(outFp16Buf,  M_TILE * sizeof(half));
        pipe.InitBuffer(reduceTmpBuf, K * sizeof(float));
        pipe.InitBuffer(reduceDstBuf, 32);
    }

    __aicore__ inline void Process() {
        if (nLen == 0) return;

        LocalTensor<half>  xFp16  = xFp16Buf.Get<half>();
        LocalTensor<float> xFp32  = xFp32Buf.Get<float>();
        LocalTensor<half>  wTile  = wTileFp16Buf.Get<half>();
        LocalTensor<float> wRow   = wRowFp32Buf.Get<float>();
        LocalTensor<float> prod   = prodFp32Buf.Get<float>();
        LocalTensor<float> outFp  = outFp32Buf.Get<float>();
        LocalTensor<half>  outFp16 = outFp16Buf.Get<half>();
        LocalTensor<float> redTmp = reduceTmpBuf.Get<float>();
        LocalTensor<float> redDst = reduceDstBuf.Get<float>();

        // x is shared across all rows; load + cast once.
        DataCopy(xFp16, xGm[0], K);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Cast(xFp32, xFp16, RoundMode::CAST_NONE, K);
        PipeBarrier<PIPE_V>();

        for (uint32_t tStart = 0; tStart < nLen; tStart += M_TILE) {
            uint32_t tLen = nLen - tStart;
            if (tLen > M_TILE) tLen = M_TILE;
            // DataCopy length must be multiple of 32 bytes (= 16 fp16);
            // K is always 1024 here, so M_TILE rows pack cleanly.
            // For tLen < M_TILE we still copy tLen*K which is fine since tLen
            // is also block-aligned (K=1024 ⇒ each row is 2 KB).
            DataCopy(wTile, wGm[static_cast<uint64_t>(tStart) * K], tLen * K);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

            for (uint32_t m = 0; m < tLen; ++m) {
                LocalTensor<half> rowFp16 = wTile[m * K];
                Cast(wRow, rowFp16, RoundMode::CAST_NONE, K);
                PipeBarrier<PIPE_V>();
                Mul(prod, xFp32, wRow, K);
                PipeBarrier<PIPE_V>();
                ReduceSum<float>(redDst, prod, redTmp, K);
                SetFlag<HardEvent::V_S>(EVENT_ID0);
                WaitFlag<HardEvent::V_S>(EVENT_ID0);
                outFp.SetValue(m, redDst.GetValue(0));
                SetFlag<HardEvent::S_V>(EVENT_ID0);
                WaitFlag<HardEvent::S_V>(EVENT_ID0);
            }

            Cast(outFp16, outFp, RoundMode::CAST_RINT, tLen);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopy(outGm[tStart], outFp16, tLen);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
    }

private:
    // M_TILE rows fp16 × K elements + M_TILE fp32 acc + scratch ≈
    // M_TILE*K*2 + ... For K=1024 and M_TILE=16: ~50 KiB. Comfortably under 192 KiB UB.
    static constexpr uint32_t M_TILE = 16;

    uint32_t K;
    uint32_t nStart;
    uint32_t nLen;
    GlobalTensor<half> xGm;
    GlobalTensor<half> wGm;
    GlobalTensor<half> outGm;
    TPipe pipe;
    TBuf<TPosition::VECCALC> xFp16Buf;
    TBuf<TPosition::VECCALC> xFp32Buf;
    TBuf<TPosition::VECCALC> wTileFp16Buf;
    TBuf<TPosition::VECCALC> wRowFp32Buf;
    TBuf<TPosition::VECCALC> prodFp32Buf;
    TBuf<TPosition::VECCALC> outFp32Buf;
    TBuf<TPosition::VECCALC> outFp16Buf;
    TBuf<TPosition::VECCALC> reduceTmpBuf;
    TBuf<TPosition::VECCALC> reduceDstBuf;
};

extern "C" __global__ __aicore__ void matmul_vec_custom(GM_ADDR x, GM_ADDR w, GM_ADDR out,
                                                          GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelMatmulVecCustom op;
    op.Init(x, w, out, tiling_data.N, tiling_data.K);
    op.Process();
}
