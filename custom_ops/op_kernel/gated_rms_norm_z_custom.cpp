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
    }

    __aicore__ inline void Process() {
        for (int32_t t = 0; t < static_cast<int32_t>(seqLen); ++t) {
            for (int32_t h = 0; h < NUM_HEADS_I; ++h) {
                int32_t base = t * VALUE_DIM_I + h * HEAD_DIM_I;
                float meanSq = 0.0f;
                for (int32_t i = 0; i < HEAD_DIM_I; ++i) {
                    float v = static_cast<float>(coreGm.GetValue(base + i));
                    meanSq += v * v;
                }
                meanSq /= static_cast<float>(HEAD_DIM_I);
                float rstd = 1.0f / sqrt(meanSq + EPSILON);
                for (int32_t i = 0; i < HEAD_DIM_I; ++i) {
                    float v = static_cast<float>(coreGm.GetValue(base + i));
                    float gamma = static_cast<float>(gammaGm.GetValue(i));
                    float z_silu = static_cast<float>(zSiluGm.GetValue(base + i));
                    float r = (gamma * v * rstd) * z_silu;
                    outGm.SetValue(base + i, static_cast<half>(r));
                }
            }
        }
    }

private:
    static constexpr int32_t NUM_HEADS_I = 16;
    static constexpr int32_t HEAD_DIM_I = 128;
    static constexpr int32_t VALUE_DIM_I = 2048;
    static constexpr float EPSILON = 0.000001f;
    GlobalTensor<half> coreGm;
    GlobalTensor<half> zSiluGm;
    GlobalTensor<half> gammaGm;
    GlobalTensor<half> outGm;
    uint32_t seqLen;
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
