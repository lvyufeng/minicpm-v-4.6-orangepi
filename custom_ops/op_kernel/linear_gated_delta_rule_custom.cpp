#include "kernel_operator.h"

using namespace AscendC;

class KernelLinearGatedDeltaRuleCustom {
public:
    __aicore__ inline KernelLinearGatedDeltaRuleCustom() {}

    __aicore__ inline void Init(GM_ADDR mixed, GM_ADDR beta, GM_ADDR decay, GM_ADDR scratch, GM_ADDR out, uint32_t seqLen) {
        this->seqLen = seqLen;
        const int32_t blockIdx = static_cast<int32_t>(GetBlockIdx());
        this->blockIdx = blockIdx;
        mixedGm.SetGlobalBuffer((__gm__ half*)mixed, seqLen * CONV_DIM_I);
        betaGm.SetGlobalBuffer((__gm__ half*)beta, seqLen * NUM_HEADS_I);
        decayGm.SetGlobalBuffer((__gm__ half*)decay, seqLen * NUM_HEADS_I);
        outGm.SetGlobalBuffer((__gm__ half*)out, seqLen * VALUE_DIM_I);
        const int32_t blockOffset = blockIdx * BLOCK_SCRATCH_ELEMS;
        stateGm.SetGlobalBuffer((__gm__ float*)scratch + blockOffset, STATE_ELEMS);
        kvBuf.SetGlobalBuffer((__gm__ float*)scratch + blockOffset + STATE_ELEMS, HEAD_DIM_I);
        dBuf.SetGlobalBuffer((__gm__ float*)scratch + blockOffset + STATE_ELEMS + HEAD_DIM_I, HEAD_DIM_I);
        qBuf.SetGlobalBuffer((__gm__ float*)scratch + blockOffset + STATE_ELEMS + 2 * HEAD_DIM_I, HEAD_DIM_I);
        kBuf.SetGlobalBuffer((__gm__ float*)scratch + blockOffset + STATE_ELEMS + 3 * HEAD_DIM_I, HEAD_DIM_I);
        vBuf.SetGlobalBuffer((__gm__ float*)scratch + blockOffset + STATE_ELEMS + 4 * HEAD_DIM_I, HEAD_DIM_I);
    }

    __aicore__ inline void Process() {
        const int32_t headStart = blockIdx * HEADS_PER_BLOCK;
        const int32_t headEnd = headStart + HEADS_PER_BLOCK;
        for (int32_t h = headStart; h < headEnd; ++h) {
            for (int32_t i = 0; i < HEAD_DIM_I; ++i) {
                for (int32_t j = 0; j < HEAD_DIM_I; ++j) {
                    stateGm.SetValue(i * HEAD_DIM_I + j, 0.0f);
                }
            }

            for (int32_t t = 0; t < static_cast<int32_t>(seqLen); ++t) {
                float qNorm = 0.0f;
                float kNorm = 0.0f;
                int32_t qBase = t * CONV_DIM_I + h * HEAD_DIM_I;
                int32_t kBase = t * CONV_DIM_I + KEY_DIM_I + h * HEAD_DIM_I;
                int32_t vBase = t * CONV_DIM_I + 2 * KEY_DIM_I + h * HEAD_DIM_I;
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

                float beta = static_cast<float>(betaGm.GetValue(t * NUM_HEADS_I + h));
                float decay = static_cast<float>(decayGm.GetValue(t * NUM_HEADS_I + h));
                for (int32_t i = 0; i < HEAD_DIM_I; ++i) {
                    for (int32_t j = 0; j < HEAD_DIM_I; ++j) {
                        stateGm.SetValue(i * HEAD_DIM_I + j,
                                         stateGm.GetValue(i * HEAD_DIM_I + j) * decay);
                    }
                }

                for (int32_t j = 0; j < HEAD_DIM_I; ++j) {
                    float acc = 0.0f;
                    for (int32_t i = 0; i < HEAD_DIM_I; ++i) {
                        acc += stateGm.GetValue(i * HEAD_DIM_I + j) * kBuf.GetValue(i);
                    }
                    kvBuf.SetValue(j, acc);
                    dBuf.SetValue(j, (vBuf.GetValue(j) - acc) * beta);
                }
                for (int32_t i = 0; i < HEAD_DIM_I; ++i) {
                    float kVal = kBuf.GetValue(i);
                    for (int32_t j = 0; j < HEAD_DIM_I; ++j) {
                        stateGm.SetValue(i * HEAD_DIM_I + j,
                                         stateGm.GetValue(i * HEAD_DIM_I + j) + kVal * dBuf.GetValue(j));
                    }
                }
                for (int32_t j = 0; j < HEAD_DIM_I; ++j) {
                    float acc = 0.0f;
                    for (int32_t i = 0; i < HEAD_DIM_I; ++i) {
                        acc += stateGm.GetValue(i * HEAD_DIM_I + j) * qBuf.GetValue(i);
                    }
                    outGm.SetValue(t * VALUE_DIM_I + h * HEAD_DIM_I + j, static_cast<half>(acc));
                }
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
    static constexpr int32_t BLOCK_SCRATCH_ELEMS = STATE_ELEMS + 5 * HEAD_DIM_I;
    static constexpr int32_t HEADS_PER_BLOCK = 2;
    static constexpr float EPSILON = 0.000001f;
    static constexpr float Q_SCALE = 0.08838834764831845f;
    GlobalTensor<half> mixedGm;
    GlobalTensor<half> betaGm;
    GlobalTensor<half> decayGm;
    GlobalTensor<half> outGm;
    GlobalTensor<float> stateGm;
    GlobalTensor<float> kvBuf;
    GlobalTensor<float> dBuf;
    GlobalTensor<float> qBuf;
    GlobalTensor<float> kBuf;
    GlobalTensor<float> vBuf;
    uint32_t seqLen;
    int32_t blockIdx;
};

extern "C" __global__ __aicore__ void linear_gated_delta_rule_custom(GM_ADDR mixed,
                                                                      GM_ADDR beta,
                                                                      GM_ADDR decay,
                                                                      GM_ADDR scratch,
                                                                      GM_ADDR out,
                                                                      GM_ADDR workspace,
                                                                      GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelLinearGatedDeltaRuleCustom op;
    op.Init(mixed, beta, decay, scratch, out, tiling_data.seqLen);
    op.Process();
}
