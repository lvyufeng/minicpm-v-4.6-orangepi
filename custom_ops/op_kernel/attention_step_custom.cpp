#include "kernel_operator.h"

using namespace AscendC;

// One block per q-head. Each block handles one full attention head's
// q · K^T → softmax → · V for the current decode token (S_q = 1).
//
// Inputs (BSH-style memory, head_dim D fixed at compile-time via tiling):
//   query:   fp16 [num_q_heads * D]                — flat q for this token
//   k_cache: fp16 [max_seq, num_kv_heads * D]      — rows [0, context) valid
//   v_cache: fp16 [max_seq, num_kv_heads * D]      — rows [0, context) valid
// Output:
//   out:     fp16 [num_q_heads * D]                — flat attn output
//
// Tiling supplies context, num_q_heads, num_kv_heads, head_dim, max_seq,
// and the inverse-sqrt scale.

class KernelAttentionStepCustom {
public:
    __aicore__ inline KernelAttentionStepCustom() {}

    __aicore__ inline void Init(GM_ADDR query, GM_ADDR k_cache, GM_ADDR v_cache, GM_ADDR out,
                                uint32_t context, uint32_t numQHeads, uint32_t numKvHeads,
                                uint32_t headDim, uint32_t maxSeq, float scale) {
        this->context     = context;
        this->numQHeads   = numQHeads;
        this->numKvHeads  = numKvHeads;
        this->headDim     = headDim;
        this->maxSeq      = maxSeq;
        this->kvDim       = numKvHeads * headDim;
        this->qPerKv      = numQHeads / numKvHeads;
        this->scale       = scale;

        const int32_t bIdx = static_cast<int32_t>(GetBlockIdx());
        this->qHead  = bIdx;
        this->kvHead = bIdx / static_cast<int32_t>(qPerKv);

        qGm.SetGlobalBuffer((__gm__ half*)query + qHead * headDim, headDim);
        kGm.SetGlobalBuffer((__gm__ half*)k_cache, static_cast<uint64_t>(maxSeq) * kvDim);
        vGm.SetGlobalBuffer((__gm__ half*)v_cache, static_cast<uint64_t>(maxSeq) * kvDim);
        outGm.SetGlobalBuffer((__gm__ half*)out + qHead * headDim, headDim);

        pipe.InitBuffer(qFp16Buf,  headDim * sizeof(half));
        pipe.InitBuffer(qFp32Buf,  headDim * sizeof(float));
        pipe.InitBuffer(kvFp16Buf, headDim * sizeof(half));
        pipe.InitBuffer(kvFp32Buf, headDim * sizeof(float));
        pipe.InitBuffer(prodBuf,   headDim * sizeof(float));
        pipe.InitBuffer(outFp32Buf, headDim * sizeof(float));
        pipe.InitBuffer(outFp16Buf, headDim * sizeof(half));
        pipe.InitBuffer(scoresBuf,  MAX_CONTEXT * sizeof(float));
        pipe.InitBuffer(reduceTmpBuf, headDim * sizeof(float));
        pipe.InitBuffer(reduceDstBuf, 32);
    }

    __aicore__ inline void Process() {
        LocalTensor<half>  qFp16   = qFp16Buf.Get<half>();
        LocalTensor<float> qFp32   = qFp32Buf.Get<float>();
        LocalTensor<half>  kvFp16  = kvFp16Buf.Get<half>();
        LocalTensor<float> kvFp32  = kvFp32Buf.Get<float>();
        LocalTensor<float> prod    = prodBuf.Get<float>();
        LocalTensor<float> outFp32 = outFp32Buf.Get<float>();
        LocalTensor<half>  outFp16 = outFp16Buf.Get<half>();
        LocalTensor<float> scores  = scoresBuf.Get<float>();
        LocalTensor<float> redTmp  = reduceTmpBuf.Get<float>();
        LocalTensor<float> redDst  = reduceDstBuf.Get<float>();

        DataCopy(qFp16, qGm[0], headDim);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Cast(qFp32, qFp16, RoundMode::CAST_NONE, headDim);
        PipeBarrier<PIPE_V>();

        // Pass 1: scores[t] = scale * (q · K[t, kvHead, :])
        for (uint32_t t = 0; t < context; ++t) {
            const uint64_t kOff = static_cast<uint64_t>(t) * kvDim + kvHead * headDim;
            DataCopy(kvFp16, kGm[kOff], headDim);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Cast(kvFp32, kvFp16, RoundMode::CAST_NONE, headDim);
            PipeBarrier<PIPE_V>();
            Mul(prod, qFp32, kvFp32, headDim);
            PipeBarrier<PIPE_V>();
            ReduceSum<float>(redDst, prod, redTmp, headDim);
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            scores.SetValue(t, redDst.GetValue(0) * scale);
            SetFlag<HardEvent::S_V>(EVENT_ID0);
            WaitFlag<HardEvent::S_V>(EVENT_ID0);
        }

        // Softmax(scores[0..context]).
        PipeBarrier<PIPE_V>();
        ReduceMax<float>(redDst, scores, redTmp, context);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        const float maxScore = redDst.GetValue(0);
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        Adds(scores, scores, -maxScore, context);
        PipeBarrier<PIPE_V>();
        Exp(scores, scores, context);
        PipeBarrier<PIPE_V>();
        ReduceSum<float>(redDst, scores, redTmp, context);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        const float invSum = 1.0f / redDst.GetValue(0);
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        Muls(scores, scores, invSum, context);
        PipeBarrier<PIPE_V>();

        // Pass 2: out = sum_t scores[t] * V[t, kvHead, :]
        Duplicate(outFp32, 0.0f, headDim);
        PipeBarrier<PIPE_V>();
        for (uint32_t t = 0; t < context; ++t) {
            const uint64_t vOff = static_cast<uint64_t>(t) * kvDim + kvHead * headDim;
            DataCopy(kvFp16, vGm[vOff], headDim);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Cast(kvFp32, kvFp16, RoundMode::CAST_NONE, headDim);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            const float pt = scores.GetValue(t);
            SetFlag<HardEvent::S_V>(EVENT_ID0);
            WaitFlag<HardEvent::S_V>(EVENT_ID0);
            Axpy<float, float>(outFp32, kvFp32, pt, headDim);
            PipeBarrier<PIPE_V>();
        }

        Cast(outFp16, outFp32, RoundMode::CAST_RINT, headDim);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(outGm[0], outFp16, headDim);
    }

private:
    static constexpr uint32_t MAX_CONTEXT = 8192;

    uint32_t context;
    uint32_t numQHeads;
    uint32_t numKvHeads;
    uint32_t headDim;
    uint32_t kvDim;
    uint32_t qPerKv;
    uint32_t maxSeq;
    float    scale;
    int32_t  qHead;
    int32_t  kvHead;

    GlobalTensor<half> qGm;
    GlobalTensor<half> kGm;
    GlobalTensor<half> vGm;
    GlobalTensor<half> outGm;
    TPipe pipe;
    TBuf<TPosition::VECCALC> qFp16Buf;
    TBuf<TPosition::VECCALC> qFp32Buf;
    TBuf<TPosition::VECCALC> kvFp16Buf;
    TBuf<TPosition::VECCALC> kvFp32Buf;
    TBuf<TPosition::VECCALC> prodBuf;
    TBuf<TPosition::VECCALC> outFp32Buf;
    TBuf<TPosition::VECCALC> outFp16Buf;
    TBuf<TPosition::VECCALC> scoresBuf;
    TBuf<TPosition::VECCALC> reduceTmpBuf;
    TBuf<TPosition::VECCALC> reduceDstBuf;
};

extern "C" __global__ __aicore__ void attention_step_custom(GM_ADDR query,
                                                              GM_ADDR k_cache,
                                                              GM_ADDR v_cache,
                                                              GM_ADDR out,
                                                              GM_ADDR workspace,
                                                              GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelAttentionStepCustom op;
    op.Init(query, k_cache, v_cache, out,
            tiling_data.context, tiling_data.numQHeads, tiling_data.numKvHeads,
            tiling_data.headDim, tiling_data.maxSeq, tiling_data.scale);
    op.Process();
}
