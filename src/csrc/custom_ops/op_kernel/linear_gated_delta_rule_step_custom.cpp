#include "kernel_operator.h"

using namespace AscendC;

class KernelLinearGatedDeltaRuleStepCustom {
public:
    __aicore__ inline KernelLinearGatedDeltaRuleStepCustom() {}

    __aicore__ inline void Init(GM_ADDR mixed, GM_ADDR beta, GM_ADDR decay,
                                GM_ADDR state, GM_ADDR /*scratch*/, GM_ADDR out) {
        const int32_t bIdx = static_cast<int32_t>(GetBlockIdx());
        this->blockIdx = bIdx;
        mixedGm.SetGlobalBuffer((__gm__ half*)mixed, CONV_DIM);
        betaGm.SetGlobalBuffer((__gm__ half*)beta, NUM_HEADS);
        decayGm.SetGlobalBuffer((__gm__ half*)decay, NUM_HEADS);
        outGm.SetGlobalBuffer((__gm__ half*)out, VALUE_DIM);
        stateGm.SetGlobalBuffer((__gm__ float*)state, NUM_HEADS * STATE_ELEMS);

        // UB allocations. Total ~135 KiB / block; the 310B AICore UB is 256 KiB.
        pipe.InitBuffer(stateBuf, HEADS_PER_BLOCK * STATE_ELEMS * sizeof(float));
        pipe.InitBuffer(mixedFp16Buf, 3 * HEAD_DIM * sizeof(half));
        pipe.InitBuffer(mixedFpBuf,   3 * HEAD_DIM * sizeof(float));
        pipe.InitBuffer(kvBuf,        HEAD_DIM * sizeof(float));
        pipe.InitBuffer(deltaBuf,     HEAD_DIM * sizeof(float));
        pipe.InitBuffer(outFpBuf,     HEAD_DIM * sizeof(float));
        pipe.InitBuffer(outFp16Buf,   HEAD_DIM * sizeof(half));
        pipe.InitBuffer(normSrcBuf,   HEAD_DIM * sizeof(float));
        pipe.InitBuffer(normTmpBuf,   HEAD_DIM * sizeof(float));
        pipe.InitBuffer(normDstBuf,   32);  // ReduceSum dst slot
    }

    __aicore__ inline void Process() {
        const int32_t headStart = blockIdx * HEADS_PER_BLOCK;

        LocalTensor<float> stateLocal = stateBuf.Get<float>();
        LocalTensor<half>  mixedFp16  = mixedFp16Buf.Get<half>();
        LocalTensor<float> mixedFp    = mixedFpBuf.Get<float>();
        LocalTensor<float> kvLocal    = kvBuf.Get<float>();
        LocalTensor<float> deltaLocal = deltaBuf.Get<float>();
        LocalTensor<float> outFp      = outFpBuf.Get<float>();
        LocalTensor<half>  outFp16    = outFp16Buf.Get<half>();
        LocalTensor<float> normSrc    = normSrcBuf.Get<float>();
        LocalTensor<float> normTmp    = normTmpBuf.Get<float>();
        LocalTensor<float> normDst    = normDstBuf.Get<float>();

        // Load this block's slice of the state matrix into UB once.
        DataCopy(stateLocal, stateGm[headStart * STATE_ELEMS],
                 HEADS_PER_BLOCK * STATE_ELEMS);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

        for (int32_t localH = 0; localH < HEADS_PER_BLOCK; ++localH) {
            const int32_t h = headStart + localH;
            LocalTensor<float> stateH = stateLocal[localH * STATE_ELEMS];

            // ---- Load q, k, v for this head into mixedFp16 (fp16), then cast to fp32.
            const int32_t qOff = h * HEAD_DIM;
            const int32_t kOff = KEY_DIM + h * HEAD_DIM;
            const int32_t vOff = 2 * KEY_DIM + h * HEAD_DIM;
            DataCopy(mixedFp16[0 * HEAD_DIM], mixedGm[qOff], HEAD_DIM);
            DataCopy(mixedFp16[1 * HEAD_DIM], mixedGm[kOff], HEAD_DIM);
            DataCopy(mixedFp16[2 * HEAD_DIM], mixedGm[vOff], HEAD_DIM);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

            Cast(mixedFp, mixedFp16, RoundMode::CAST_NONE, 3 * HEAD_DIM);
            PipeBarrier<PIPE_V>();

            LocalTensor<float> q = mixedFp[0];
            LocalTensor<float> k = mixedFp[HEAD_DIM];
            LocalTensor<float> v = mixedFp[2 * HEAD_DIM];

            // ---- qNorm, kNorm via vectorized ReduceSum of (x * x).
            Mul(normSrc, q, q, HEAD_DIM);
            PipeBarrier<PIPE_V>();
            ReduceSum<float>(normDst, normSrc, normTmp, HEAD_DIM);
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            const float qNorm = normDst.GetValue(0);

            Mul(normSrc, k, k, HEAD_DIM);
            PipeBarrier<PIPE_V>();
            ReduceSum<float>(normDst, normSrc, normTmp, HEAD_DIM);
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            const float kNorm = normDst.GetValue(0);

            const float qInv = Q_SCALE / sqrt(qNorm + EPSILON);
            const float kInv = 1.0f / sqrt(kNorm + EPSILON);
            SetFlag<HardEvent::S_V>(EVENT_ID0);
            WaitFlag<HardEvent::S_V>(EVENT_ID0);

            Muls(q, q, qInv, HEAD_DIM);
            Muls(k, k, kInv, HEAD_DIM);
            PipeBarrier<PIPE_V>();

            // ---- Load beta, decay scalars (1 elem each; GetValue is fine here).
            const float betaH  = static_cast<float>(betaGm.GetValue(h));
            const float decayH = static_cast<float>(decayGm.GetValue(h));

            // ---- Pass A: state *= decay (rowwise), kv = state^T . k.
            Duplicate(kvLocal, 0.0f, HEAD_DIM);
            PipeBarrier<PIPE_V>();
            for (int32_t i = 0; i < HEAD_DIM; ++i) {
                LocalTensor<float> rowI = stateH[i * HEAD_DIM];
                Muls(rowI, rowI, decayH, HEAD_DIM);
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_S>(EVENT_ID0);
                WaitFlag<HardEvent::V_S>(EVENT_ID0);
                const float ki = k.GetValue(i);
                SetFlag<HardEvent::S_V>(EVENT_ID0);
                WaitFlag<HardEvent::S_V>(EVENT_ID0);
                Axpy<float, float>(kvLocal, rowI, ki, HEAD_DIM);
                PipeBarrier<PIPE_V>();
            }

            // ---- delta = (v - kv) * beta.
            Sub(deltaLocal, v, kvLocal, HEAD_DIM);
            PipeBarrier<PIPE_V>();
            Muls(deltaLocal, deltaLocal, betaH, HEAD_DIM);
            PipeBarrier<PIPE_V>();

            // ---- Pass B: state[i] += k[i] * delta, then out += state[i] * q[i].
            Duplicate(outFp, 0.0f, HEAD_DIM);
            PipeBarrier<PIPE_V>();
            for (int32_t i = 0; i < HEAD_DIM; ++i) {
                LocalTensor<float> rowI = stateH[i * HEAD_DIM];
                SetFlag<HardEvent::V_S>(EVENT_ID0);
                WaitFlag<HardEvent::V_S>(EVENT_ID0);
                const float ki = k.GetValue(i);
                const float qi = q.GetValue(i);
                SetFlag<HardEvent::S_V>(EVENT_ID0);
                WaitFlag<HardEvent::S_V>(EVENT_ID0);
                Axpy<float, float>(rowI, deltaLocal, ki, HEAD_DIM);
                PipeBarrier<PIPE_V>();
                Axpy<float, float>(outFp, rowI, qi, HEAD_DIM);
                PipeBarrier<PIPE_V>();
            }

            // ---- Cast out to fp16 and write to GM.
            Cast(outFp16, outFp, RoundMode::CAST_RINT, HEAD_DIM);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopy(outGm[h * HEAD_DIM], outFp16, HEAD_DIM);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }

        // ---- Write the updated state back to GM.
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(stateGm[headStart * STATE_ELEMS], stateLocal,
                 HEADS_PER_BLOCK * STATE_ELEMS);
    }

private:
    static constexpr int32_t NUM_HEADS       = 16;
    static constexpr int32_t HEAD_DIM        = 128;
    static constexpr int32_t KEY_DIM         = NUM_HEADS * HEAD_DIM;
    static constexpr int32_t VALUE_DIM       = NUM_HEADS * HEAD_DIM;
    static constexpr int32_t CONV_DIM        = 2 * KEY_DIM + VALUE_DIM;
    static constexpr int32_t STATE_ELEMS     = HEAD_DIM * HEAD_DIM;
    static constexpr int32_t HEADS_PER_BLOCK = 2;
    static constexpr float   EPSILON         = 0.000001f;
    static constexpr float   Q_SCALE         = 0.08838834764831845f;

    int32_t blockIdx;
    GlobalTensor<half>  mixedGm;
    GlobalTensor<half>  betaGm;
    GlobalTensor<half>  decayGm;
    GlobalTensor<half>  outGm;
    GlobalTensor<float> stateGm;
    TPipe pipe;
    TBuf<TPosition::VECCALC> stateBuf;
    TBuf<TPosition::VECCALC> mixedFp16Buf;
    TBuf<TPosition::VECCALC> mixedFpBuf;
    TBuf<TPosition::VECCALC> kvBuf;
    TBuf<TPosition::VECCALC> deltaBuf;
    TBuf<TPosition::VECCALC> outFpBuf;
    TBuf<TPosition::VECCALC> outFp16Buf;
    TBuf<TPosition::VECCALC> normSrcBuf;
    TBuf<TPosition::VECCALC> normTmpBuf;
    TBuf<TPosition::VECCALC> normDstBuf;
};

extern "C" __global__ __aicore__ void linear_gated_delta_rule_step_custom(GM_ADDR mixed,
                                                                            GM_ADDR beta,
                                                                            GM_ADDR decay,
                                                                            GM_ADDR state,
                                                                            GM_ADDR scratch,
                                                                            GM_ADDR out,
                                                                            GM_ADDR workspace,
                                                                            GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    (void)tiling_data;
    KernelLinearGatedDeltaRuleStepCustom op;
    op.Init(mixed, beta, decay, state, scratch, out);
    op.Process();
}
