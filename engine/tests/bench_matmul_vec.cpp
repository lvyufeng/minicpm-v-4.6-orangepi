// Compare aclnnMatmulVecCustom vs aclnnMm (via matmul_b_transposed) at T=1
// over a sweep of N values. K is fixed at 1024 (the only shape used in decode).
#include "minicpmv/acl_context.h"
#include "minicpmv/ops.h"
#include "minicpmv/tensor.h"

#include "aclnn_matmul_vec_custom.h"

#include <acl/acl.h>
#include <aclnnop/aclnn_mm.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace minicpmv;
using clk = std::chrono::steady_clock;

static double ms(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

static double bench_custom(int N, int K, int iters, aclrtStream stream) {
    Tensor x({K}, DType::Float16); x.allocate();
    Tensor w({N, K}, DType::Float16); w.allocate();
    Tensor o({N}, DType::Float16); o.allocate();
    std::vector<uint16_t> hx(K, 0x3c00), hw((size_t)N * K, 0x3c00);
    x.copy_from_host(hx.data(), hx.size() * sizeof(uint16_t));
    w.copy_from_host(hw.data(), hw.size() * sizeof(uint16_t));

    // We can't directly call the engine wrapper from here because it's matmul_b_transposed
    // with 2D shapes; just call via aclnn directly.
    auto run_once = [&]() {
        const int64_t xDims[1] = {K};
        const int64_t wDims[2] = {N, K};
        const int64_t oDims[1] = {N};
        const int64_t xStride[1] = {1};
        const int64_t wStride[2] = {K, 1};
        const int64_t oStride[1] = {1};
        aclTensor* tx = aclCreateTensor(xDims, 1, ACL_FLOAT16, xStride, 0, ACL_FORMAT_ND, xDims, 1, x.data());
        aclTensor* tw = aclCreateTensor(wDims, 2, ACL_FLOAT16, wStride, 0, ACL_FORMAT_ND, wDims, 2, w.data());
        aclTensor* to = aclCreateTensor(oDims, 1, ACL_FLOAT16, oStride, 0, ACL_FORMAT_ND, oDims, 1, o.data());

        uint64_t ws = 0;
        aclOpExecutor* exec = nullptr;
        aclnnMatmulVecCustomGetWorkspaceSize(tx, tw, to, &ws, &exec);
        void* wsp = nullptr;
        if (ws > 0) aclrtMalloc(&wsp, ws, ACL_MEM_MALLOC_HUGE_FIRST);
        aclnnMatmulVecCustom(wsp, ws, exec, stream);
        aclrtSynchronizeStream(stream);
        if (wsp) aclrtFree(wsp);
        aclDestroyTensor(tx);
        aclDestroyTensor(tw);
        aclDestroyTensor(to);
    };

    for (int i = 0; i < 5; ++i) run_once();
    auto t0 = clk::now();
    for (int i = 0; i < iters; ++i) run_once();
    auto t1 = clk::now();
    return ms(t0, t1) / iters;
}

static double bench_ref(int N, int K, int iters, aclrtStream stream) {
    Tensor x({1, K}, DType::Float16); x.allocate();
    Tensor w({N, K}, DType::Float16); w.allocate();
    Tensor o({1, N}, DType::Float16); o.allocate();
    std::vector<uint16_t> hx(K, 0x3c00), hw((size_t)N * K, 0x3c00);
    x.copy_from_host(hx.data(), hx.size() * sizeof(uint16_t));
    w.copy_from_host(hw.data(), hw.size() * sizeof(uint16_t));

    // Call aclnnMm with B transposed view, similar to engine's matmul_b_transposed.
    auto run_once = [&]() {
        const int64_t aDims[2] = {1, K};
        const int64_t bView[2] = {K, N};
        const int64_t bStor[2] = {N, K};
        const int64_t oDims[2] = {1, N};
        const int64_t aStride[2] = {K, 1};
        const int64_t bStride[2] = {1, K};  // transposed view
        const int64_t oStride[2] = {N, 1};
        aclTensor* ta = aclCreateTensor(aDims, 2, ACL_FLOAT16, aStride, 0, ACL_FORMAT_ND, aDims, 2, x.data());
        aclTensor* tb = aclCreateTensor(bView, 2, ACL_FLOAT16, bStride, 0, ACL_FORMAT_ND, bStor, 2, w.data());
        aclTensor* to = aclCreateTensor(oDims, 2, ACL_FLOAT16, oStride, 0, ACL_FORMAT_ND, oDims, 2, o.data());
        uint64_t ws = 0;
        aclOpExecutor* exec = nullptr;
        aclnnMmGetWorkspaceSize(ta, tb, to, /*kCubeMathType=*/1, &ws, &exec);
        void* wsp = nullptr;
        if (ws > 0) aclrtMalloc(&wsp, ws, ACL_MEM_MALLOC_HUGE_FIRST);
        aclnnMm(wsp, ws, exec, stream);
        aclrtSynchronizeStream(stream);
        if (wsp) aclrtFree(wsp);
        aclDestroyTensor(ta);
        aclDestroyTensor(tb);
        aclDestroyTensor(to);
    };

    for (int i = 0; i < 5; ++i) run_once();
    auto t0 = clk::now();
    for (int i = 0; i < iters; ++i) run_once();
    auto t1 = clk::now();
    return ms(t0, t1) / iters;
}

int main() {
    AclContext ctx(0);
    int Ns[] = {512, 1024, 2048, 3584, 4096, 5120, 6144, 7168, 8224, 248094};
    std::printf("%-8s %-12s %-12s %-8s\n", "N", "custom_ms", "aclnnMm_ms", "speedup");
    for (int N : Ns) {
        int iters = (N >= 100000) ? 5 : 30;
        double c = bench_custom(N, 1024, iters, ctx.stream());
        double r = bench_ref(N, 1024, iters, ctx.stream());
        std::printf("%-8d %-12.3f %-12.3f %-8.2fx\n", N, c, r, r / c);
    }
    return 0;
}
