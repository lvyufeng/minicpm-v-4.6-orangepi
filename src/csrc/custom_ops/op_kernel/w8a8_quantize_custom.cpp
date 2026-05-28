#include "kernel_operator.h"

using namespace AscendC;

class KernelW8a8QuantizeCustom {
public:
    __aicore__ inline KernelW8a8QuantizeCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR xq, GM_ADDR scale, uint32_t K) {
        this->K = K;
        xGm.SetGlobalBuffer((__gm__ half*)x, K);
        xqGm.SetGlobalBuffer((__gm__ int8_t*)xq, K);
        scaleGm.SetGlobalBuffer((__gm__ half*)scale, 1);
        pipe.InitBuffer(xBuf, K * sizeof(half));
        pipe.InitBuffer(xqBuf, K * sizeof(int8_t));
    }

    __aicore__ inline void Process() {
        LocalTensor<half> x = xBuf.Get<half>();
        LocalTensor<int8_t> xq = xqBuf.Get<int8_t>();
        DataCopy(x, xGm, K);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

        float maxAbs = 0.0f;
        for (uint32_t i = 0; i < K; ++i) {
            float v = static_cast<float>(x.GetValue(i));
            if (v < 0.0f) v = -v;
            if (v > maxAbs) maxAbs = v;
        }
        float scale = maxAbs / 127.0f;
        if (scale < 0.00000001f) scale = 0.00000001f;
        const float inv = 1.0f / scale;
        for (uint32_t i = 0; i < K; ++i) {
            float v = static_cast<float>(x.GetValue(i)) * inv;
            if (v > 127.0f) v = 127.0f;
            if (v < -128.0f) v = -128.0f;
            int32_t qi = static_cast<int32_t>(v >= 0.0f ? v + 0.5f : v - 0.5f);
            xq.SetValue(i, static_cast<int8_t>(qi));
        }
        scaleGm.SetValue(0, static_cast<half>(scale));
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(xqGm, xq, K);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }

private:
    uint32_t K{0};
    GlobalTensor<half> xGm;
    GlobalTensor<int8_t> xqGm;
    GlobalTensor<half> scaleGm;
    TPipe pipe;
    TBuf<TPosition::VECCALC> xBuf;
    TBuf<TPosition::VECCALC> xqBuf;
};

extern "C" __global__ __aicore__ void w8a8_quantize_custom(GM_ADDR x,
                                                            GM_ADDR xq,
                                                            GM_ADDR scale,
                                                            GM_ADDR workspace,
                                                            GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelW8a8QuantizeCustom op;
    op.Init(x, xq, scale, tiling_data.K);
    op.Process();
}
