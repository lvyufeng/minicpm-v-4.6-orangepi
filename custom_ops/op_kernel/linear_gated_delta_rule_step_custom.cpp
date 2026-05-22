#include "kernel_operator.h"

using namespace AscendC;

class KernelLinearGatedDeltaRuleStepCustom {
public:
    __aicore__ inline KernelLinearGatedDeltaRuleStepCustom() {}

    __aicore__ inline void Init(GM_ADDR mixed, GM_ADDR beta, GM_ADDR decay,
                                GM_ADDR state, GM_ADDR scratch, GM_ADDR out) {
        const int32_t blockIdx = static_cast<int32_t>(GetBlockIdx());
        this->blockIdx = blockIdx;
        mixedGm.SetGlobalBuffer((__gm__ half*)mixed, CONV_DIM_I);
        betaGm.SetGlobalBuffer((__gm__ half*)beta, NUM_HEADS_I);
        decayGm.SetGlobalBuffer((__gm__ half*)decay, NUM_HEADS_I);
        outGm.SetGlobalBuffer((__gm__ half*)out, VALUE_DIM_I);
        stateGm.SetGlobalBuffer((__gm__ float*)state, NUM_HEADS_I * STATE_ELEMS);
        const int32_t scratchOffset = blockIdx * SCRATCH_PER_BLOCK;
        qBuf.SetGlobalBuffer((__gm__ float*)scratch + scratchOffset, HEAD_DIM_I);
        kBuf.SetGlobalBuffer((__gm__ float*)scratch + scratchOffset + HEAD_DIM_I, HEAD_DIM_I);
        vBuf.SetGlobalBuffer((__gm__ float*)scratch + scratchOffset + 2 * HEAD_DIM_I, HEAD_DIM_I);
        kvBuf.SetGlobalBuffer((__gm__ float*)scratch + scratchOffset + 3 * HEAD_DIM_I, HEAD_DIM_I);
        dBuf.SetGlobalBuffer((__gm__ float*)scratch + scratchOffset + 4 * HEAD_DIM_I, HEAD_DIM_I);
        outFloatBuf.SetGlobalBuffer((__gm__ float*)scratch + scratchOffset + 5 * HEAD_DIM_I, HEAD_DIM_I);
    }

    __aicore__ inline void Process() {
        const int32_t headStart = blockIdx * HEADS_PER_BLOCK;
        const int32_t headEnd = headStart + HEADS_PER_BLOCK;
        for (int32_t h = headStart; h < headEnd; ++h) {
            const int32_t stateBase = h * STATE_ELEMS;
            const int32_t qBase = h * HEAD_DIM_I;
            const int32_t kBase = KEY_DIM_I + h * HEAD_DIM_I;
            const int32_t vBase = 2 * KEY_DIM_I + h * HEAD_DIM_I;

            // Load q, k, v into scratch as fp32, computing q/k l2 norms along the way.
            float qNorm = 0.0f;
            float kNorm = 0.0f;
            for (int32_t i = 0; i < HEAD_DIM_I; ++i) {
                float qv = static_cast<float>(mixedGm.GetValue(qBase + i));
                float kv = static_cast<float>(mixedGm.GetValue(kBase + i));
                float vv = static_cast<float>(mixedGm.GetValue(vBase + i));
                qBuf.SetValue(i, qv);
                kBuf.SetValue(i, kv);
                vBuf.SetValue(i, vv);
                qNorm += qv * qv;
                kNorm += kv * kv;
            }
            float qInv = 1.0f / sqrt(qNorm + EPSILON);
            float kInv = 1.0f / sqrt(kNorm + EPSILON);
            for (int32_t i = 0; i < HEAD_DIM_I; ++i) {
                qBuf.SetValue(i, qBuf.GetValue(i) * qInv * Q_SCALE);
                kBuf.SetValue(i, kBuf.GetValue(i) * kInv);
            }

            const float beta = static_cast<float>(betaGm.GetValue(h));
            const float decay = static_cast<float>(decayGm.GetValue(h));

            // Pass A: apply decay, compute kv = state' . k, then delta = (v - kv) * beta.
            // Each state element is read once and written once.
            for (int32_t j = 0; j < HEAD_DIM_I; ++j) {
                kvBuf.SetValue(j, 0.0f);
            }
            for (int32_t i = 0; i < HEAD_DIM_I; ++i) {
                const float kVal = kBuf.GetValue(i);
                const int32_t rowBase = stateBase + i * HEAD_DIM_I;
                for (int32_t j = 0; j < HEAD_DIM_I; ++j) {
                    float s = stateGm.GetValue(rowBase + j) * decay;
                    stateGm.SetValue(rowBase + j, s);
                    kvBuf.SetValue(j, kvBuf.GetValue(j) + s * kVal);
                }
            }
            for (int32_t j = 0; j < HEAD_DIM_I; ++j) {
                dBuf.SetValue(j, (vBuf.GetValue(j) - kvBuf.GetValue(j)) * beta);
                outFloatBuf.SetValue(j, 0.0f);
            }

            // Pass B: add outer(k, delta) into state and accumulate out = state_final . q.
            for (int32_t i = 0; i < HEAD_DIM_I; ++i) {
                const float kVal = kBuf.GetValue(i);
                const float qVal = qBuf.GetValue(i);
                const int32_t rowBase = stateBase + i * HEAD_DIM_I;
                for (int32_t j = 0; j < HEAD_DIM_I; ++j) {
                    float s = stateGm.GetValue(rowBase + j) + kVal * dBuf.GetValue(j);
                    stateGm.SetValue(rowBase + j, s);
                    outFloatBuf.SetValue(j, outFloatBuf.GetValue(j) + s * qVal);
                }
            }
            for (int32_t j = 0; j < HEAD_DIM_I; ++j) {
                outGm.SetValue(h * HEAD_DIM_I + j, static_cast<half>(outFloatBuf.GetValue(j)));
            }
        }
    }

private:
    static constexpr int32_t NUM_HEADS_I = 16;
    static constexpr int32_t HEAD_DIM_I = 128;
    static constexpr int32_t KEY_DIM_I = 2048;
    static constexpr int32_t VALUE_DIM_I = 2048;
    static constexpr int32_t CONV_DIM_I = 6144;
    static constexpr int32_t STATE_ELEMS = HEAD_DIM_I * HEAD_DIM_I;
    static constexpr int32_t HEADS_PER_BLOCK = 2;
    static constexpr int32_t SCRATCH_PER_BLOCK = 6 * HEAD_DIM_I;
    static constexpr float EPSILON = 0.000001f;
    static constexpr float Q_SCALE = 0.08838834764831845f;
    GlobalTensor<half> mixedGm;
    GlobalTensor<half> betaGm;
    GlobalTensor<half> decayGm;
    GlobalTensor<half> outGm;
    GlobalTensor<float> stateGm;
    GlobalTensor<float> qBuf;
    GlobalTensor<float> kBuf;
    GlobalTensor<float> vBuf;
    GlobalTensor<float> kvBuf;
    GlobalTensor<float> dBuf;
    GlobalTensor<float> outFloatBuf;
    int32_t blockIdx;
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
