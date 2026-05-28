#include "minicpmv/acl_context.h"
#include "minicpmv/ops.h"
#include "minicpmv/tensor.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace minicpmv;
using clk = std::chrono::steady_clock;

static double ms(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

int main(int argc, char** argv) {
    AclContext ctx(0);
    const int K = argc > 1 ? std::atoi(argv[1]) : 1024;
    const int N = argc > 2 ? std::atoi(argv[2]) : 3584;
    const int iters = argc > 3 ? std::atoi(argv[3]) : 100;

    std::vector<int8_t> hx(K, 1);
    std::vector<int8_t> hw(static_cast<size_t>(K) * N, 1);
    Tensor x({1, K}, DType::Int8);
    Tensor w({K, N}, DType::Int8);
    Tensor out({1, N}, DType::Int32);
    Tensor xScale({1}, DType::Float16);
    Tensor wScale({N}, DType::Float16);
    Tensor outDeq({1, N}, DType::Float16);
    std::vector<uint16_t> hxs(1, 0x3c00);
    std::vector<uint16_t> hws(N, 0x3c00);
    x.copy_from_host(hx.data(), hx.size());
    w.copy_from_host(hw.data(), hw.size());
    xScale.copy_from_host(hxs.data(), hxs.size() * sizeof(uint16_t));
    wScale.copy_from_host(hws.data(), hws.size() * sizeof(uint16_t));
    out.allocate();
    outDeq.allocate();

    for (int i = 0; i < 3; ++i) matmul_w8a8_i32(x, w, out, ctx.stream());
    aclrtSynchronizeStream(ctx.stream());

    auto t0 = clk::now();
    for (int i = 0; i < iters; ++i) matmul_w8a8_i32(x, w, out, ctx.stream());
    aclrtSynchronizeStream(ctx.stream());
    auto t1 = clk::now();
    const double w8a8_ms = ms(t0, t1) / iters;
    std::printf("W8A8_i32 K=%d N=%d per_call_ms=%.3f (%d iters)\n", K, N, w8a8_ms, iters);

    for (int i = 0; i < 3; ++i) {
        matmul_w8a8_i32(x, w, out, ctx.stream());
        w8a8_dequant(out, xScale, wScale, outDeq, ctx.stream());
    }
    aclrtSynchronizeStream(ctx.stream());
    t0 = clk::now();
    for (int i = 0; i < iters; ++i) {
        matmul_w8a8_i32(x, w, out, ctx.stream());
        w8a8_dequant(out, xScale, wScale, outDeq, ctx.stream());
    }
    aclrtSynchronizeStream(ctx.stream());
    t1 = clk::now();
    const double w8a8_deq_ms = ms(t0, t1) / iters;
    std::printf("W8A8_deq K=%d N=%d per_call_ms=%.3f (%d iters)\n", K, N, w8a8_deq_ms, iters);

    std::vector<uint16_t> hxf(K, 0x3c00);
    std::vector<uint16_t> hwf(static_cast<size_t>(K) * N, 0x3c00);
    Tensor xf({1, K}, DType::Float16);
    Tensor wf({K, N}, DType::Float16);
    Tensor outf({1, N}, DType::Float16);
    xf.copy_from_host(hxf.data(), hxf.size() * sizeof(uint16_t));
    wf.copy_from_host(hwf.data(), hwf.size() * sizeof(uint16_t));
    outf.allocate();

    for (int i = 0; i < 5; ++i) matmul_b_transposed(xf, wf, outf, ctx.stream());
    aclrtSynchronizeStream(ctx.stream());

    t0 = clk::now();
    for (int i = 0; i < iters; ++i) matmul_b_transposed(xf, wf, outf, ctx.stream());
    aclrtSynchronizeStream(ctx.stream());
    t1 = clk::now();
    const double fp16_ms = ms(t0, t1) / iters;
    std::printf("fp16     K=%d N=%d per_call_ms=%.3f (%d iters)\n", K, N, fp16_ms, iters);
    std::printf("speedup W8A8_i32 vs fp16: %.2fx\n", fp16_ms / w8a8_ms);
    std::printf("speedup W8A8_deq vs fp16: %.2fx\n", fp16_ms / w8a8_deq_ms);
    return 0;
}
