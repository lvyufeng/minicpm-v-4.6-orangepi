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
    }

    __aicore__ inline void Process() {
        for (uint32_t row = 0; row < rowsPerBlock; ++row) {
            uint32_t base = (startRow + row) * HIDDEN_SIZE;
            float sumSq = 0.0f;
            for (uint32_t i = 0; i < HIDDEN_SIZE; ++i) {
                float v = static_cast<float>(xGm.GetValue(base + i));
                sumSq += v * v;
            }
            float invRms = 1.0f / sqrt(sumSq * INV_HIDDEN_SIZE + EPSILON);
            for (uint32_t i = 0; i < HIDDEN_SIZE; ++i) {
                float v = static_cast<float>(xGm.GetValue(base + i));
                float g = static_cast<float>(gammaGm.GetValue(i));
                outGm.SetValue(base + i, static_cast<half>(v * invRms * g));
            }
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
};

extern "C" __global__ __aicore__ void rms_norm1024_custom(GM_ADDR x, GM_ADDR gamma, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelRmsNorm1024Custom op;
    op.Init(x, gamma, out, tiling_data.size);
    op.Process();
}