#include "kernel_operator.h"

using namespace AscendC;

class KernelGatedRmsNormZCustom {
public:
    __aicore__ inline KernelGatedRmsNormZCustom() {}

    __aicore__ inline void Init(GM_ADDR core, GM_ADDR z_silu, GM_ADDR gamma, GM_ADDR out, uint32_t seqLen) {
        this->seqLen = seqLen;
        coreGm.SetGlobalBuffer((__gm__ half*)core, seqLen * VALUE_DIM_I);
        zSiluGm.SetGlobalBuffer((__gm__ half*)z_silu, seqLen * VALUE_DIM_I);
        gammaGm.SetGlobalBuffer((__gm__ half*)gamma, HEAD_DIM_I);
        outGm.SetGlobalBuffer((__gm__ half*)out, seqLen * VALUE_DIM_I);

        // Split work over blocks: each block takes a contiguous chunk of (t,h) groups.
        const uint32_t totalGroups = seqLen * NUM_HEADS_I;
        const uint32_t blockNum = GetBlockNum();
        const uint32_t bIdx = GetBlockIdx();
        uint32_t groupsPerBlock = (totalGroups + blockNum - 1) / blockNum;
        uint32_t startGroup = bIdx * groupsPerBlock;
        uint32_t remain = totalGroups > startGroup ? totalGroups - startGroup : 0;
        groupsPerBlock = remain < groupsPerBlock ? remain : groupsPerBlock;
        this->startGroup = startGroup;
        this->groupsPerBlock = groupsPerBlock;

        pipe.InitBuffer(coreFp16Buf, HEAD_DIM_I * sizeof(half));
        pipe.InitBuffer(coreFp32Buf, HEAD_DIM_I * sizeof(float));
        pipe.InitBuffer(zSiluFp16Buf, HEAD_DIM_I * sizeof(half));
        pipe.InitBuffer(zSiluFp32Buf, HEAD_DIM_I * sizeof(float));
        pipe.InitBuffer(gammaFp16Buf, HEAD_DIM_I * sizeof(half));
        pipe.InitBuffer(gammaFp32Buf, HEAD_DIM_I * sizeof(float));
        pipe.InitBuffer(sqBuf,        HEAD_DIM_I * sizeof(float));
        pipe.InitBuffer(scratchBuf,   HEAD_DIM_I * sizeof(float));
        pipe.InitBuffer(outFp16Buf,   HEAD_DIM_I * sizeof(half));
        pipe.InitBuffer(reduceTmpBuf, HEAD_DIM_I * sizeof(float));
        pipe.InitBuffer(reduceDstBuf, 32);
    }

    __aicore__ inline void Process() {
        if (groupsPerBlock == 0) return;

        LocalTensor<half>  coreFp16  = coreFp16Buf.Get<half>();
        LocalTensor<float> coreFp32  = coreFp32Buf.Get<float>();
        LocalTensor<half>  zSiluFp16 = zSiluFp16Buf.Get<half>();
        LocalTensor<float> zSiluFp32 = zSiluFp32Buf.Get<float>();
        LocalTensor<half>  gammaFp16 = gammaFp16Buf.Get<half>();
        LocalTensor<float> gammaFp32 = gammaFp32Buf.Get<float>();
        LocalTensor<float> sq        = sqBuf.Get<float>();
        LocalTensor<float> scratch   = scratchBuf.Get<float>();
        LocalTensor<half>  outFp16   = outFp16Buf.Get<half>();
        LocalTensor<float> redTmp    = reduceTmpBuf.Get<float>();
        LocalTensor<float> redDst    = reduceDstBuf.Get<float>();

        // gamma shared across all groups — load once.
        DataCopy(gammaFp16, gammaGm[0], HEAD_DIM_I);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Cast(gammaFp32, gammaFp16, RoundMode::CAST_NONE, HEAD_DIM_I);
        PipeBarrier<PIPE_V>();

        for (uint32_t g = 0; g < groupsPerBlock; ++g) {
            const uint32_t group = startGroup + g;
            const uint32_t t = group / NUM_HEADS_I;
            const uint32_t h = group - t * NUM_HEADS_I;
            const uint32_t base = t * VALUE_DIM_I + h * HEAD_DIM_I;

            DataCopy(coreFp16,  coreGm[base],   HEAD_DIM_I);
            DataCopy(zSiluFp16, zSiluGm[base],  HEAD_DIM_I);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Cast(coreFp32,  coreFp16,  RoundMode::CAST_NONE, HEAD_DIM_I);
            Cast(zSiluFp32, zSiluFp16, RoundMode::CAST_NONE, HEAD_DIM_I);
            PipeBarrier<PIPE_V>();

            Mul(sq, coreFp32, coreFp32, HEAD_DIM_I);
            PipeBarrier<PIPE_V>();
            ReduceSum<float>(redDst, sq, redTmp, HEAD_DIM_I);
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            const float sumSq = redDst.GetValue(0);
            const float meanSq = sumSq * INV_HEAD_DIM;
            const float rstd = 1.0f / sqrt(meanSq + EPSILON);
            SetFlag<HardEvent::S_V>(EVENT_ID0);
            WaitFlag<HardEvent::S_V>(EVENT_ID0);

            // r = (gamma * core * rstd) * z_silu
            Muls(scratch, coreFp32, rstd, HEAD_DIM_I);
            PipeBarrier<PIPE_V>();
            Mul(scratch, scratch, gammaFp32, HEAD_DIM_I);
            PipeBarrier<PIPE_V>();
            Mul(scratch, scratch, zSiluFp32, HEAD_DIM_I);
            PipeBarrier<PIPE_V>();
            Cast(outFp16, scratch, RoundMode::CAST_RINT, HEAD_DIM_I);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopy(outGm[base], outFp16, HEAD_DIM_I);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
    }

private:
    static constexpr uint32_t NUM_HEADS_I = 16;
    static constexpr uint32_t HEAD_DIM_I  = 128;
    static constexpr uint32_t VALUE_DIM_I = NUM_HEADS_I * HEAD_DIM_I;
    static constexpr float    EPSILON     = 0.000001f;
    static constexpr float    INV_HEAD_DIM = 1.0f / 128.0f;

    GlobalTensor<half> coreGm;
    GlobalTensor<half> zSiluGm;
    GlobalTensor<half> gammaGm;
    GlobalTensor<half> outGm;
    uint32_t seqLen;
    uint32_t startGroup;
    uint32_t groupsPerBlock;
    TPipe pipe;
    TBuf<TPosition::VECCALC> coreFp16Buf;
    TBuf<TPosition::VECCALC> coreFp32Buf;
    TBuf<TPosition::VECCALC> zSiluFp16Buf;
    TBuf<TPosition::VECCALC> zSiluFp32Buf;
    TBuf<TPosition::VECCALC> gammaFp16Buf;
    TBuf<TPosition::VECCALC> gammaFp32Buf;
    TBuf<TPosition::VECCALC> sqBuf;
    TBuf<TPosition::VECCALC> scratchBuf;
    TBuf<TPosition::VECCALC> outFp16Buf;
    TBuf<TPosition::VECCALC> reduceTmpBuf;
    TBuf<TPosition::VECCALC> reduceDstBuf;
};

extern "C" __global__ __aicore__ void gated_rms_norm_z_custom(GM_ADDR core,
                                                               GM_ADDR z_silu,
                                                               GM_ADDR gamma,
                                                               GM_ADDR out,
                                                               GM_ADDR workspace,
                                                               GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelGatedRmsNormZCustom op;
    op.Init(core, z_silu, gamma, out, tiling_data.seqLen);
    op.Process();
}
