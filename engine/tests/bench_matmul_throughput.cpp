// Quick NPU compute throughput probe: large fp16 GEMM via aclnnMm.
// Reports TFLOPS estimate based on 2*M*N*K work.
#include "minicpmv/acl_context.h"
#include "minicpmv/ops.h"
#include "minicpmv/tensor.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>

using namespace minicpmv;
using clk = std::chrono::steady_clock;

static double ms(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

int main(int argc, char** argv) {
    int M = (argc > 1) ? std::atoi(argv[1]) : 1024;
    int N = (argc > 2) ? std::atoi(argv[2]) : 1024;
    int K = (argc > 3) ? std::atoi(argv[3]) : 1024;
    int iters = (argc > 4) ? std::atoi(argv[4]) : 50;
    int warmup = 5;

    AclContext ctx(0);
    Tensor a({M, K}, DType::Float16); a.allocate();
    Tensor b({K, N}, DType::Float16); b.allocate();
    Tensor c({M, N}, DType::Float16); c.allocate();
    std::vector<uint16_t> ha(static_cast<size_t>(M) * K, 0x3c00);  // 1.0 fp16
    std::vector<uint16_t> hb(static_cast<size_t>(K) * N, 0x3c00);
    a.copy_from_host(ha.data(), ha.size() * sizeof(uint16_t));
    b.copy_from_host(hb.data(), hb.size() * sizeof(uint16_t));

    for (int i = 0; i < warmup; ++i) matmul(a, b, c, ctx.stream());
    check_acl(aclrtSynchronizeStream(ctx.stream()), "warmup sync");

    auto t0 = clk::now();
    for (int i = 0; i < iters; ++i) matmul(a, b, c, ctx.stream());
    check_acl(aclrtSynchronizeStream(ctx.stream()), "bench sync");
    auto t1 = clk::now();

    double total_ms = ms(t0, t1);
    double per_ms = total_ms / iters;
    double flops = 2.0 * static_cast<double>(M) * N * K;
    double tflops = flops / (per_ms * 1e-3) / 1e12;
    std::printf("matmul [%d,%d]x[%d,%d] iters=%d total_ms=%.2f per_iter_ms=%.3f tflops=%.3f\n",
                M, K, K, N, iters, total_ms, per_ms, tflops);
    return 0;
}
