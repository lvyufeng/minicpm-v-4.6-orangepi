#include "kernel_operator.h"

using namespace AscendC;

class KernelRmsNorm1024Custom {
public:
    __aicore__ inline KernelRmsNorm1024Custom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR out, uint32_t totalLength) {
        totalRows = totalLength / HIDDEN_SIZE;
        rowsPerBlock = (totalRows + GetBlockNum() - 1) / GetBlockNum();
        startRow = GetBlockIdx() * rowsPerBlock;
        uint32_t remainRows = totalRows > startRow ? totalRows - startRow : 0;
        rowsPerBlock = remainRows < rowsPerBlock ? remainRows : rowsPerBlock;
        xGm.SetGlobalBuffer((__gm__ half*)x, totalLength);
        gammaGm.SetGlobalBuffer((__gm__ half*)gamma, HIDDEN_SIZE);
        outGm.SetGlobalBuffer((__gm__ half*)out, totalLength);

        pipe.InitBuffer(xFp16Buf,   HIDDEN_SIZE * sizeof(half));
        pipe.InitBuffer(xFp32Buf,   HIDDEN_SIZE * sizeof(float));
        pipe.InitBuffer(gammaFp16Buf, HIDDEN_SIZE * sizeof(half));
        pipe.InitBuffer(gammaFp32Buf, HIDDEN_SIZE * sizeof(float));
        pipe.InitBuffer(sqBuf,      HIDDEN_SIZE * sizeof(float));
        pipe.InitBuffer(outFp16Buf, HIDDEN_SIZE * sizeof(half));
        pipe.InitBuffer(reduceTmpBuf, HIDDEN_SIZE * sizeof(float));
        pipe.InitBuffer(reduceDstBuf, 32);
    }

    __aicore__ inline void Process() {
        if (rowsPerBlock == 0) return;

        LocalTensor<half>  xFp16    = xFp16Buf.Get<half>();
        LocalTensor<float> xFp32    = xFp32Buf.Get<float>();
        LocalTensor<half>  gammaFp16 = gammaFp16Buf.Get<half>();
        LocalTensor<float> gammaFp32 = gammaFp32Buf.Get<float>();
        LocalTensor<float> sq       = sqBuf.Get<float>();
        LocalTensor<half>  outFp16  = outFp16Buf.Get<half>();
        LocalTensor<float> redTmp   = reduceTmpBuf.Get<float>();
        LocalTensor<float> redDst   = reduceDstBuf.Get<float>();

        // gamma is row-invariant; load + cast once.
        DataCopy(gammaFp16, gammaGm[0], HIDDEN_SIZE);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Cast(gammaFp32, gammaFp16, RoundMode::CAST_NONE, HIDDEN_SIZE);
        PipeBarrier<PIPE_V>();

        for (uint32_t r = 0; r < rowsPerBlock; ++r) {
            const uint32_t base = (startRow + r) * HIDDEN_SIZE;

            DataCopy(xFp16, xGm[base], HIDDEN_SIZE);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Cast(xFp32, xFp16, RoundMode::CAST_NONE, HIDDEN_SIZE);
            PipeBarrier<PIPE_V>();

            Mul(sq, xFp32, xFp32, HIDDEN_SIZE);
            PipeBarrier<PIPE_V>();
            ReduceSum<float>(redDst, sq, redTmp, HIDDEN_SIZE);
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            const float sumSq = redDst.GetValue(0);
            const float invRms = 1.0f / sqrt(sumSq * INV_HIDDEN_SIZE + EPSILON);
            SetFlag<HardEvent::S_V>(EVENT_ID0);
            WaitFlag<HardEvent::S_V>(EVENT_ID0);

            // out = x * invRms * gamma  (use sq as scratch then xFp32)
            Muls(sq, xFp32, invRms, HIDDEN_SIZE);
            PipeBarrier<PIPE_V>();
            Mul(xFp32, sq, gammaFp32, HIDDEN_SIZE);
            PipeBarrier<PIPE_V>();
            Cast(outFp16, xFp32, RoundMode::CAST_RINT, HIDDEN_SIZE);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopy(outGm[base], outFp16, HIDDEN_SIZE);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
    }

private:
    static constexpr uint32_t HIDDEN_SIZE = 1024;
    static constexpr float INV_HIDDEN_SIZE = 1.0f / 1024.0f;
    static constexpr float EPSILON = 0.000001f;

    GlobalTensor<half> xGm;
    GlobalTensor<half> gammaGm;
    GlobalTensor<half> outGm;
    uint32_t totalRows;
    uint32_t rowsPerBlock;
    uint32_t startRow;
    TPipe pipe;
    TBuf<TPosition::VECCALC> xFp16Buf;
    TBuf<TPosition::VECCALC> xFp32Buf;
    TBuf<TPosition::VECCALC> gammaFp16Buf;
    TBuf<TPosition::VECCALC> gammaFp32Buf;
    TBuf<TPosition::VECCALC> sqBuf;
    TBuf<TPosition::VECCALC> outFp16Buf;
    TBuf<TPosition::VECCALC> reduceTmpBuf;
    TBuf<TPosition::VECCALC> reduceDstBuf;
};

extern "C" __global__ __aicore__ void rms_norm1024_custom(GM_ADDR x, GM_ADDR gamma, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelRmsNorm1024Custom op;
    op.Init(x, gamma, out, tiling_data.size);
    op.Process();
}
