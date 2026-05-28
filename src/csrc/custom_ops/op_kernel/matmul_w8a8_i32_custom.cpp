#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;
using namespace matmul;

__aicore__ inline void CopyTiling(TCubeTiling* tiling, GM_ADDR tilingGm) {
    uint32_t* ptr = reinterpret_cast<uint32_t*>(tiling);
    auto tiling32 = reinterpret_cast<__gm__ uint32_t*>(tilingGm);
    for (uint32_t i = 0; i < sizeof(TCubeTiling) / sizeof(uint32_t); ++i, ++ptr) {
        *ptr = *(tiling32 + i);
    }
}

extern "C" __global__ __aicore__ void matmul_w8a8_i32_custom(GM_ADDR x,
                                                              GM_ADDR w,
                                                              GM_ADDR out,
                                                              GM_ADDR workspace,
                                                              GM_ADDR tilingGm) {
    using AT = int8_t;
    using BT = int8_t;
    using CT = int32_t;
    using BiasT = int32_t;

    TPipe pipe;
    TCubeTiling tiling;
    CopyTiling(&tiling, tilingGm);
    if (GetBlockIdx() >= tiling.usedCoreNum) return;

    GlobalTensor<AT> xGm;
    GlobalTensor<BT> wGm;
    GlobalTensor<CT> outGm;
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ AT*>(x), static_cast<uint64_t>(tiling.M) * tiling.Ka);
    wGm.SetGlobalBuffer(reinterpret_cast<__gm__ BT*>(w), static_cast<uint64_t>(tiling.Ka) * tiling.N);
    outGm.SetGlobalBuffer(reinterpret_cast<__gm__ CT*>(out), static_cast<uint64_t>(tiling.M) * tiling.N);

    typedef MatmulType<TPosition::GM, CubeFormat::ND, AT> aType;
    typedef MatmulType<TPosition::GM, CubeFormat::ND, BT> bType;
    typedef MatmulType<TPosition::GM, CubeFormat::ND, CT> cType;
    typedef MatmulType<TPosition::GM, CubeFormat::ND, BiasT> biasType;

    MatmulImpl<aType, bType, cType, biasType> mm;
    mm.SetSubBlockIdx(0);
    mm.Init(&tiling, &pipe);
    mm.SetTensorA(xGm);
    mm.SetTensorB(wGm);
    mm.IterateAll(outGm);
}
