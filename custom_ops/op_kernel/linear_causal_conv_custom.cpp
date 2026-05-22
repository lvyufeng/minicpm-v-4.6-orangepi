#include "kernel_operator.h"

using namespace AscendC;

class KernelLinearCausalConvCustom {
public:
    __aicore__ inline KernelLinearCausalConvCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR out,
                                uint32_t totalLength, uint32_t seqLen, uint32_t channels) {
        this->totalLength = totalLength;
        this->seqLen = seqLen;
        this->channels = channels;
        this->blockLength = (totalLength + GetBlockNum() - 1) / GetBlockNum();
        uint32_t startOffset = GetBlockIdx() * this->blockLength;
        uint32_t remain = totalLength > startOffset ? totalLength - startOffset : 0;
        this->blockLength = remain < this->blockLength ? remain : this->blockLength;
        this->start = startOffset;
        xGm.SetGlobalBuffer((__gm__ half*)x, totalLength);
        weightGm.SetGlobalBuffer((__gm__ half*)weight, channels * KERNEL_SIZE);
        outGm.SetGlobalBuffer((__gm__ half*)out, totalLength);
    }

    __aicore__ inline void Process() {
        for (uint32_t i = 0; i < blockLength; ++i) {
            uint32_t linear = start + i;
            uint32_t t = linear / channels;
            uint32_t c = linear - t * channels;
            float acc = 0.0f;
            for (uint32_t k = 0; k < KERNEL_SIZE; ++k) {
                int32_t srcT = static_cast<int32_t>(t) - static_cast<int32_t>(KERNEL_SIZE - 1) + static_cast<int32_t>(k);
                if (srcT >= 0) {
                    float x = static_cast<float>(xGm.GetValue(static_cast<uint32_t>(srcT) * channels + c));
                    float w = static_cast<float>(weightGm.GetValue(c * KERNEL_SIZE + k));
                    acc += x * w;
                }
            }
            outGm.SetValue(linear, static_cast<half>(acc));
        }
    }

private:
    static constexpr uint32_t KERNEL_SIZE = 4;
    GlobalTensor<half> xGm;
    GlobalTensor<half> weightGm;
    GlobalTensor<half> outGm;
    uint32_t totalLength;
    uint32_t seqLen;
    uint32_t channels;
    uint32_t start;
    uint32_t blockLength;
};

extern "C" __global__ __aicore__ void linear_causal_conv_custom(GM_ADDR x,
                                                                 GM_ADDR weight,
                                                                 GM_ADDR out,
                                                                 GM_ADDR workspace,
                                                                 GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelLinearCausalConvCustom op;
    op.Init(x, weight, out, tiling_data.totalLength, tiling_data.seqLen, tiling_data.channels);
    op.Process();
}
