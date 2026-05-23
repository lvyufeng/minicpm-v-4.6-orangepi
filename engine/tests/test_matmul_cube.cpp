// Verify aclnnMatmulCubeCustom produces correct results vs aclnnMm for M=1.
#include "minicpmv/acl_context.h"
#include "minicpmv/tensor.h"

#include "aclnn_matmul_cube_custom.h"
#include <acl/acl.h>
#include <aclnnop/aclnn_mm.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace minicpmv;

static uint16_t f2h(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;
    if (exp <= 0) return static_cast<uint16_t>(sign);
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}

static float h2f(uint16_t h) {
    uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    uint32_t out;
    if (exp == 0) { out = sign; }
    else if (exp == 31) { out = sign | 0x7f800000u | (mant << 13); }
    else { out = sign | ((exp + 127 - 15) << 23) | (mant << 13); }
    float f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}

int main(int argc, char** argv) {
    AclContext ctx(0);
    const int N = argc > 1 ? std::atoi(argv[1]) : 3584;
    const int K = 1024;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<uint16_t> hx(K), hw((size_t)N * K);
    for (auto& v : hx) v = f2h(dist(rng));
    for (auto& v : hw) v = f2h(dist(rng));

    // Pre-transpose weight from [N, K] (matmul_b_transposed convention) to
    // [K, N] (natural for matmul_cube which doesn't do internal transpose).
    std::vector<uint16_t> hwT((size_t)K * N);
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            hwT[k * N + n] = hw[n * K + k];
        }
    }

    Tensor x({1, K}, DType::Float16); x.allocate(); x.copy_from_host(hx.data(), hx.size() * 2);
    Tensor w({N, K}, DType::Float16); w.allocate(); w.copy_from_host(hw.data(), hw.size() * 2);
    Tensor wT({K, N}, DType::Float16); wT.copy_from_host(hwT.data(), hwT.size() * 2);
    Tensor oCube({1, N}, DType::Float16); oCube.allocate();
    Tensor oRef ({1, N}, DType::Float16); oRef.allocate();

    auto stream = ctx.stream();

    // Cube (B is natural [K, N], pre-transposed wT)
    {
        const int64_t aDims[2] = {1, K};
        const int64_t bDims[2] = {K, N};
        const int64_t oDims[2] = {1, N};
        const int64_t aStride[2] = {K, 1};
        const int64_t bStride[2] = {N, 1};
        const int64_t oStride[2] = {N, 1};
        aclTensor* ta = aclCreateTensor(aDims, 2, ACL_FLOAT16, aStride, 0, ACL_FORMAT_ND, aDims, 2, x.data());
        aclTensor* tb = aclCreateTensor(bDims, 2, ACL_FLOAT16, bStride, 0, ACL_FORMAT_ND, bDims, 2, wT.data());
        aclTensor* to = aclCreateTensor(oDims, 2, ACL_FLOAT16, oStride, 0, ACL_FORMAT_ND, oDims, 2, oCube.data());
        uint64_t ws = 0;
        aclOpExecutor* exec = nullptr;
        aclnnMatmulCubeCustomGetWorkspaceSize(ta, tb, to, &ws, &exec);
        void* wsp = nullptr;
        if (ws > 0) aclrtMalloc(&wsp, ws, ACL_MEM_MALLOC_HUGE_FIRST);
        aclnnMatmulCubeCustom(wsp, ws, exec, stream);
        aclrtSynchronizeStream(stream);
        if (wsp) aclrtFree(wsp);
        aclDestroyTensor(ta); aclDestroyTensor(tb); aclDestroyTensor(to);
    }

    // aclnnMm reference (matching matmul_b_transposed semantics).
    {
        const int64_t aDims[2] = {1, K};
        const int64_t bView[2] = {K, N};
        const int64_t bStor[2] = {N, K};
        const int64_t oDims[2] = {1, N};
        const int64_t aStride[2] = {K, 1};
        const int64_t bStride[2] = {1, K};
        const int64_t oStride[2] = {N, 1};
        aclTensor* ta = aclCreateTensor(aDims, 2, ACL_FLOAT16, aStride, 0, ACL_FORMAT_ND, aDims, 2, x.data());
        aclTensor* tb = aclCreateTensor(bView, 2, ACL_FLOAT16, bStride, 0, ACL_FORMAT_ND, bStor, 2, w.data());
        aclTensor* to = aclCreateTensor(oDims, 2, ACL_FLOAT16, oStride, 0, ACL_FORMAT_ND, oDims, 2, oRef.data());
        uint64_t ws = 0;
        aclOpExecutor* exec = nullptr;
        aclnnMmGetWorkspaceSize(ta, tb, to, /*cubeMathType=*/1, &ws, &exec);
        void* wsp = nullptr;
        if (ws > 0) aclrtMalloc(&wsp, ws, ACL_MEM_MALLOC_HUGE_FIRST);
        aclnnMm(wsp, ws, exec, stream);
        aclrtSynchronizeStream(stream);
        if (wsp) aclrtFree(wsp);
        aclDestroyTensor(ta); aclDestroyTensor(tb); aclDestroyTensor(to);
    }

    std::vector<uint16_t> hc(N), hr(N);
    oCube.copy_to_host(hc.data(), N * 2);
    oRef.copy_to_host(hr.data(), N * 2);

    float maxAbs = 0.0f;
    int big = 0;
    for (int i = 0; i < N; ++i) {
        float c = h2f(hc[i]), r = h2f(hr[i]);
        float d = std::fabs(c - r);
        if (d > maxAbs) maxAbs = d;
        if (d > 0.05f) ++big;
    }
    std::printf("N=%d  max_abs_diff=%.4f  large_diffs(>0.05)=%d/%d\n", N, maxAbs, big, N);
    if (N <= 8 || big == 0) {
        std::printf("first8 cube: ");
        for (int i = 0; i < std::min(8, N); ++i) std::printf("%.4f ", h2f(hc[i]));
        std::printf("\nfirst8 ref : ");
        for (int i = 0; i < std::min(8, N); ++i) std::printf("%.4f ", h2f(hr[i]));
        std::printf("\n");
    }
    return 0;
}
