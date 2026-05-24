// Minimum-viable cube-unit matmul using AscendC MatmulImpl high-level API.
// out = A @ B, with B optionally stored transposed (matching our existing
// `matmul_b_transposed` convention). Goal: see whether a custom cube kernel
// can match aclnnMm on 310B.
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

extern "C" __global__ __aicore__ void matmul_cube_custom(GM_ADDR a, GM_ADDR b, GM_ADDR c,
                                                          GM_ADDR workspace, GM_ADDR tilingGm) {
    // 310B has unified AICore (cube + vector in one), so don't skip on g_coreType.

    using AT = half;
    using BT = half;
    using CT = half;
    using BiasT = half;

    TPipe pipe;
    TCubeTiling tiling;
    CopyTiling(&tiling, tilingGm);

    if (GetBlockIdx() >= tiling.usedCoreNum) return;

    GlobalTensor<AT> aGm;
    GlobalTensor<BT> bGm;
    GlobalTensor<CT> cGm;
    aGm.SetGlobalBuffer(reinterpret_cast<__gm__ AT*>(a), static_cast<uint64_t>(tiling.M) * tiling.Ka);
    bGm.SetGlobalBuffer(reinterpret_cast<__gm__ BT*>(b), static_cast<uint64_t>(tiling.Ka) * tiling.N);
    cGm.SetGlobalBuffer(reinterpret_cast<__gm__ CT*>(c), static_cast<uint64_t>(tiling.M) * tiling.N);

    // Treat B as natural [K, N] (NOT transposed); the engine pre-transposes
    // weight tensors at load time. This is the simpler / robust cube
    // configuration, and the transpose cost is amortized across all calls.
    typedef MatmulType<TPosition::GM, CubeFormat::ND, AT> aType;
    typedef MatmulType<TPosition::GM, CubeFormat::ND, BT> bType;
    typedef MatmulType<TPosition::GM, CubeFormat::ND, CT> cType;
    typedef MatmulType<TPosition::GM, CubeFormat::ND, BiasT> biasType;

    MatmulImpl<aType, bType, cType, biasType> mm;
    mm.SetSubBlockIdx(0);
    mm.Init(&tiling, &pipe);
    mm.SetTensorA(aGm);
    mm.SetTensorB(bGm);
    mm.IterateAll(cGm);
}
