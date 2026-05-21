#include "kernel_operator.h"

using namespace AscendC;

class KernelAddCustom {
public:
    __aicore__ inline KernelAddCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength) {
        this->totalLength = totalLength;
        this->blockLength = (totalLength + GetBlockNum() - 1) / GetBlockNum();
        uint32_t start = GetBlockIdx() * this->blockLength;
        uint32_t remain = totalLength > start ? totalLength - start : 0;
        this->blockLength = remain < this->blockLength ? remain : this->blockLength;
        xGm.SetGlobalBuffer((__gm__ half*)x + start, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ half*)y + start, this->blockLength);
        zGm.SetGlobalBuffer((__gm__ half*)z + start, this->blockLength);
    }

    __aicore__ inline void Process() {
        for (uint32_t i = 0; i < blockLength; ++i) {
            float a = static_cast<float>(xGm.GetValue(i));
            float b = static_cast<float>(yGm.GetValue(i));
            zGm.SetValue(i, static_cast<half>(a + b));
        }
    }

private:
    GlobalTensor<half> xGm;
    GlobalTensor<half> yGm;
    GlobalTensor<half> zGm;
    uint32_t totalLength;
    uint32_t blockLength;
};

extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelAddCustom op;
    op.Init(x, y, z, tiling_data.size);
    op.Process();
}