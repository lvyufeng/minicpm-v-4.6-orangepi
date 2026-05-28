#include "kernel_operator.h"

using namespace AscendC;

class KernelW8a8DequantCustom {
public:
    __aicore__ inline KernelW8a8DequantCustom() {}

    __aicore__ inline void Init(GM_ADDR acc, GM_ADDR xScale, GM_ADDR wScale, GM_ADDR out, uint32_t totalLen) {
        const uint32_t blocks = GetBlockNum();
        const uint32_t bIdx = GetBlockIdx();
        const uint32_t chunk = (totalLen + blocks - 1) / blocks;
        const uint32_t aligned = ((chunk + ALIGN_FP16 - 1) / ALIGN_FP16) * ALIGN_FP16;
        uint32_t start = bIdx * aligned;
        uint32_t end = start + aligned;
        if (end > totalLen) end = totalLen;
        if (start > totalLen) start = totalLen;
        this->start = start;
        this->blockLen = end - start;
        this->totalLen = totalLen;
        accGm.SetGlobalBuffer((__gm__ int32_t*)acc + start, blockLen);
        xScaleGm.SetGlobalBuffer((__gm__ half*)xScale, 1);
        wScaleGm.SetGlobalBuffer((__gm__ half*)wScale + start, blockLen);
        outGm.SetGlobalBuffer((__gm__ half*)out + start, blockLen);
        pipe.InitBuffer(accBuf, TILE_ELEMS * sizeof(int32_t));
        pipe.InitBuffer(accFp32Buf, TILE_ELEMS * sizeof(float));
        pipe.InitBuffer(wScaleBuf, TILE_ELEMS * sizeof(half));
        pipe.InitBuffer(wScaleFp32Buf, TILE_ELEMS * sizeof(float));
        pipe.InitBuffer(outFp32Buf, TILE_ELEMS * sizeof(float));
        pipe.InitBuffer(outFp16Buf, TILE_ELEMS * sizeof(half));
    }

    __aicore__ inline void Process() {
        if (blockLen == 0) return;
        LocalTensor<int32_t> acc = accBuf.Get<int32_t>();
        LocalTensor<float> accFp32 = accFp32Buf.Get<float>();
        LocalTensor<half> wScale = wScaleBuf.Get<half>();
        LocalTensor<float> wScaleFp32 = wScaleFp32Buf.Get<float>();
        LocalTensor<float> outFp32 = outFp32Buf.Get<float>();
        LocalTensor<half> outFp16 = outFp16Buf.Get<half>();
        half xScaleH = xScaleGm.GetValue(0);
        float xScale = static_cast<float>(xScaleH);

        for (uint32_t offset = 0; offset < blockLen; offset += TILE_ELEMS) {
            uint32_t n = blockLen - offset;
            if (n > TILE_ELEMS) n = TILE_ELEMS;
            uint32_t nAligned = ((n + ALIGN_FP16 - 1) / ALIGN_FP16) * ALIGN_FP16;
            DataCopy(acc, accGm[offset], nAligned);
            DataCopy(wScale, wScaleGm[offset], nAligned);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Cast(accFp32, acc, RoundMode::CAST_NONE, nAligned);
            Cast(wScaleFp32, wScale, RoundMode::CAST_NONE, nAligned);
            PipeBarrier<PIPE_V>();
            Mul(outFp32, accFp32, wScaleFp32, nAligned);
            PipeBarrier<PIPE_V>();
            Muls(outFp32, outFp32, xScale, nAligned);
            PipeBarrier<PIPE_V>();
            Cast(outFp16, outFp32, RoundMode::CAST_RINT, nAligned);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopy(outGm[offset], outFp16, nAligned);
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
    }

private:
    static constexpr uint32_t TILE_ELEMS = 2048;
    static constexpr uint32_t ALIGN_FP16 = 16;
    uint32_t totalLen{0};
    uint32_t start{0};
    uint32_t blockLen{0};
    GlobalTensor<int32_t> accGm;
    GlobalTensor<half> xScaleGm;
    GlobalTensor<half> wScaleGm;
    GlobalTensor<half> outGm;
    TPipe pipe;
    TBuf<TPosition::VECCALC> accBuf;
    TBuf<TPosition::VECCALC> accFp32Buf;
    TBuf<TPosition::VECCALC> wScaleBuf;
    TBuf<TPosition::VECCALC> wScaleFp32Buf;
    TBuf<TPosition::VECCALC> outFp32Buf;
    TBuf<TPosition::VECCALC> outFp16Buf;
};

extern "C" __global__ __aicore__ void w8a8_dequant_custom(GM_ADDR acc,
                                                           GM_ADDR xScale,
                                                           GM_ADDR wScale,
                                                           GM_ADDR out,
                                                           GM_ADDR workspace,
                                                           GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelW8a8DequantCustom op;
    op.Init(acc, xScale, wScale, out, tiling_data.totalLen);
    op.Process();
}
