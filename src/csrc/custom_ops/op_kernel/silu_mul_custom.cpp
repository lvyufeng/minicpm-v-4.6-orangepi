#include "kernel_operator.h"

using namespace AscendC;

// out = silu(gate) * up, where silu(x) = x * sigmoid(x).
// Inputs and output are fp16, shape [T, intermediate]. Splits the flat element
// range across blocks; each tile is processed in UB with vector intrinsics.

class KernelSiluMulCustom {
public:
    __aicore__ inline KernelSiluMulCustom() {}

    __aicore__ inline void Init(GM_ADDR gate, GM_ADDR up, GM_ADDR out, uint32_t totalLen) {
        const uint32_t blocks  = GetBlockNum();
        const uint32_t bIdx    = GetBlockIdx();
        const uint32_t chunk   = (totalLen + blocks - 1) / blocks;
        const uint32_t aligned = ((chunk + ALIGN_FP16 - 1) / ALIGN_FP16) * ALIGN_FP16;
        uint32_t start = bIdx * aligned;
        uint32_t end   = start + aligned;
        if (end   > totalLen) end   = totalLen;
        if (start > totalLen) start = totalLen;
        this->start = start;
        this->blockLen = end - start;
        this->totalLen = totalLen;

        gateGm.SetGlobalBuffer((__gm__ half*)gate + start, blockLen);
        upGm.SetGlobalBuffer((__gm__ half*)up   + start, blockLen);
        outGm.SetGlobalBuffer((__gm__ half*)out + start, blockLen);

        pipe.InitBuffer(gFp16Buf,  TILE_ELEMS * sizeof(half));
        pipe.InitBuffer(uFp16Buf,  TILE_ELEMS * sizeof(half));
        pipe.InitBuffer(gFp32Buf,  TILE_ELEMS * sizeof(float));
        pipe.InitBuffer(uFp32Buf,  TILE_ELEMS * sizeof(float));
        pipe.InitBuffer(sigBuf,    TILE_ELEMS * sizeof(float));
        pipe.InitBuffer(outFp16Buf, TILE_ELEMS * sizeof(half));
    }

    __aicore__ inline void Process() {
        if (blockLen == 0) return;

        LocalTensor<half>  gFp16   = gFp16Buf.Get<half>();
        LocalTensor<half>  uFp16   = uFp16Buf.Get<half>();
        LocalTensor<float> gFp32   = gFp32Buf.Get<float>();
        LocalTensor<float> uFp32   = uFp32Buf.Get<float>();
        LocalTensor<float> sig     = sigBuf.Get<float>();
        LocalTensor<half>  outFp16 = outFp16Buf.Get<half>();

        for (uint32_t offset = 0; offset < blockLen; offset += TILE_ELEMS) {
            uint32_t n = blockLen - offset;
            if (n > TILE_ELEMS) n = TILE_ELEMS;
            // DataCopy needs 32B-aligned counts; pad up but never beyond UB tile.
            uint32_t nAligned = ((n + ALIGN_FP16 - 1) / ALIGN_FP16) * ALIGN_FP16;

            DataCopy(gFp16, gateGm[offset], nAligned);
            DataCopy(uFp16, upGm[offset],   nAligned);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

            Cast(gFp32, gFp16, RoundMode::CAST_NONE, nAligned);
            Cast(uFp32, uFp16, RoundMode::CAST_NONE, nAligned);
            PipeBarrier<PIPE_V>();

            // silu(g) = g / (1 + exp(-g)). Compute via Muls/Exp/Adds/Div.
            Muls(sig, gFp32, -1.0f, nAligned);
            PipeBarrier<PIPE_V>();
            Exp(sig, sig, nAligned);
            PipeBarrier<PIPE_V>();
            Adds(sig, sig, 1.0f, nAligned);
            PipeBarrier<PIPE_V>();
            Div(sig, gFp32, sig, nAligned);   // sig = silu(g)
            PipeBarrier<PIPE_V>();
            Mul(sig, sig, uFp32, nAligned);   // sig = silu(g) * u
            PipeBarrier<PIPE_V>();

            Cast(outFp16, sig, RoundMode::CAST_RINT, nAligned);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopy(outGm[offset], outFp16, nAligned);
            // V->MTE2 sync: the next iteration's DataCopy(gFp16/uFp16) must not
            // race with this iter's Cast/Muls/Exp/Div/Mul that read those UB
            // buffers in the V pipe. Without this the kernel is
            // non-deterministic when blockLen > TILE_ELEMS (multi-tile loops).
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
    }

private:
    static constexpr uint32_t TILE_ELEMS = 4096;   // 4096 fp16 = 8 KiB per tile buf
    static constexpr uint32_t ALIGN_FP16 = 16;     // 32-byte alignment for fp16

    uint32_t totalLen;
    uint32_t start;
    uint32_t blockLen;
    GlobalTensor<half> gateGm;
    GlobalTensor<half> upGm;
    GlobalTensor<half> outGm;
    TPipe pipe;
    TBuf<TPosition::VECCALC> gFp16Buf;
    TBuf<TPosition::VECCALC> uFp16Buf;
    TBuf<TPosition::VECCALC> gFp32Buf;
    TBuf<TPosition::VECCALC> uFp32Buf;
    TBuf<TPosition::VECCALC> sigBuf;
    TBuf<TPosition::VECCALC> outFp16Buf;
};

extern "C" __global__ __aicore__ void silu_mul_custom(GM_ADDR gate, GM_ADDR up, GM_ADDR out,
                                                       GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelSiluMulCustom op;
    op.Init(gate, up, out, tiling_data.totalLen);
    op.Process();
}
