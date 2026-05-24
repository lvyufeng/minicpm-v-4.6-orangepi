#include "kernel_operator.h"

using namespace AscendC;

// Fused last-step causal conv1d for linear-attention decode.
//
// At T=1 decode, the engine builds a [4, C] window (3 cached history rows + 1
// new row) and runs a generic causal conv1d that produces [4, C] but only
// reads the last row. This kernel computes only that last row directly.
//
// Math: out[c] = sum_{k=0..3} x[k, c] * w_t[k, c]
//
// Inputs:
//   x        fp16 [4, C], row-major (contiguous)
//   weight_t fp16 [4, C], pre-transposed from canonical [C, 4] layout. Each row
//            k is the weight for tap k across all channels.
// Output:
//   out      fp16 [1, C]
//
// Vectorized: each of 8 blocks handles a contiguous channel slice. 4 input
// rows + 4 weight rows DMA into UB as contiguous tiles, accumulated in fp16
// (4 mul + 3 add stays well inside fp16 range for our weights), cast back via
// V->MTE3 DMA.

class KernelLinearCausalConvStepCustom {
public:
    __aicore__ inline KernelLinearCausalConvStepCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight_t, GM_ADDR out, uint32_t channels) {
        const uint32_t blocks = GetBlockNum();
        const uint32_t bIdx = GetBlockIdx();
        const uint32_t chunk = (channels + blocks - 1) / blocks;
        const uint32_t aligned = ((chunk + ALIGN_FP16 - 1) / ALIGN_FP16) * ALIGN_FP16;
        uint32_t start = bIdx * aligned;
        uint32_t end = start + aligned;
        if (end > channels) end = channels;
        if (start > channels) start = channels;
        this->channels = channels;
        this->start = start;
        this->blockLen = end - start;

        xGm.SetGlobalBuffer((__gm__ half*)x, 4 * channels);
        wGm.SetGlobalBuffer((__gm__ half*)weight_t, 4 * channels);
        outGm.SetGlobalBuffer((__gm__ half*)out, channels);

        pipe.InitBuffer(x0Buf,  TILE_ELEMS * sizeof(half));
        pipe.InitBuffer(x1Buf,  TILE_ELEMS * sizeof(half));
        pipe.InitBuffer(x2Buf,  TILE_ELEMS * sizeof(half));
        pipe.InitBuffer(x3Buf,  TILE_ELEMS * sizeof(half));
        pipe.InitBuffer(w0Buf,  TILE_ELEMS * sizeof(half));
        pipe.InitBuffer(w1Buf,  TILE_ELEMS * sizeof(half));
        pipe.InitBuffer(w2Buf,  TILE_ELEMS * sizeof(half));
        pipe.InitBuffer(w3Buf,  TILE_ELEMS * sizeof(half));
        pipe.InitBuffer(accBuf, TILE_ELEMS * sizeof(half));
        pipe.InitBuffer(tmpBuf, TILE_ELEMS * sizeof(half));
    }

    __aicore__ inline void Process() {
        if (blockLen == 0) return;

        LocalTensor<half> x0 = x0Buf.Get<half>();
        LocalTensor<half> x1 = x1Buf.Get<half>();
        LocalTensor<half> x2 = x2Buf.Get<half>();
        LocalTensor<half> x3 = x3Buf.Get<half>();
        LocalTensor<half> w0 = w0Buf.Get<half>();
        LocalTensor<half> w1 = w1Buf.Get<half>();
        LocalTensor<half> w2 = w2Buf.Get<half>();
        LocalTensor<half> w3 = w3Buf.Get<half>();
        LocalTensor<half> acc = accBuf.Get<half>();
        LocalTensor<half> tmp = tmpBuf.Get<half>();

        for (uint32_t offset = 0; offset < blockLen; offset += TILE_ELEMS) {
            uint32_t n = blockLen - offset;
            if (n > TILE_ELEMS) n = TILE_ELEMS;
            uint32_t nAligned = ((n + ALIGN_FP16 - 1) / ALIGN_FP16) * ALIGN_FP16;

            DataCopy(x0, xGm[0 * channels + start + offset], nAligned);
            DataCopy(x1, xGm[1 * channels + start + offset], nAligned);
            DataCopy(x2, xGm[2 * channels + start + offset], nAligned);
            DataCopy(x3, xGm[3 * channels + start + offset], nAligned);
            DataCopy(w0, wGm[0 * channels + start + offset], nAligned);
            DataCopy(w1, wGm[1 * channels + start + offset], nAligned);
            DataCopy(w2, wGm[2 * channels + start + offset], nAligned);
            DataCopy(w3, wGm[3 * channels + start + offset], nAligned);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

            Mul(acc, x0, w0, nAligned);
            PipeBarrier<PIPE_V>();
            Mul(tmp, x1, w1, nAligned);
            PipeBarrier<PIPE_V>();
            Add(acc, acc, tmp, nAligned);
            PipeBarrier<PIPE_V>();
            Mul(tmp, x2, w2, nAligned);
            PipeBarrier<PIPE_V>();
            Add(acc, acc, tmp, nAligned);
            PipeBarrier<PIPE_V>();
            Mul(tmp, x3, w3, nAligned);
            PipeBarrier<PIPE_V>();
            Add(acc, acc, tmp, nAligned);
            PipeBarrier<PIPE_V>();

            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopy(outGm[start + offset], acc, nAligned);
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
    }

private:
    static constexpr uint32_t TILE_ELEMS = 4096;
    static constexpr uint32_t ALIGN_FP16 = 16;

    uint32_t channels;
    uint32_t start;
    uint32_t blockLen;

    GlobalTensor<half> xGm;
    GlobalTensor<half> wGm;
    GlobalTensor<half> outGm;

    TPipe pipe;
    TBuf<TPosition::VECCALC> x0Buf;
    TBuf<TPosition::VECCALC> x1Buf;
    TBuf<TPosition::VECCALC> x2Buf;
    TBuf<TPosition::VECCALC> x3Buf;
    TBuf<TPosition::VECCALC> w0Buf;
    TBuf<TPosition::VECCALC> w1Buf;
    TBuf<TPosition::VECCALC> w2Buf;
    TBuf<TPosition::VECCALC> w3Buf;
    TBuf<TPosition::VECCALC> accBuf;
    TBuf<TPosition::VECCALC> tmpBuf;
};

extern "C" __global__ __aicore__ void linear_causal_conv_step_custom(GM_ADDR x,
                                                                       GM_ADDR weight_t,
                                                                       GM_ADDR out,
                                                                       GM_ADDR workspace,
                                                                       GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelLinearCausalConvStepCustom op;
    op.Init(x, weight_t, out, tiling_data.channels);
    op.Process();
}
