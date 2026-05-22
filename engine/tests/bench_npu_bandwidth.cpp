// Measure NPU memory bandwidth: D2D copy and elementwise add over large fp16 tensors.
#include "minicpmv/acl_context.h"
#include "minicpmv/ops.h"
#include "minicpmv/tensor.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace minicpmv;
using clk = std::chrono::steady_clock;

static double ms(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

int main(int argc, char** argv) {
    size_t bytes = (argc > 1) ? static_cast<size_t>(std::atoi(argv[1])) << 20 : 128ULL << 20;
    int iters = (argc > 2) ? std::atoi(argv[2]) : 30;

    AclContext ctx(0);
    size_t n_fp16 = bytes / sizeof(uint16_t);
    int64_t cols = 1024;
    int64_t rows = static_cast<int64_t>(n_fp16 / cols);
    n_fp16 = static_cast<size_t>(rows) * cols;
    bytes = n_fp16 * sizeof(uint16_t);
    std::printf("buffer=%.1f MiB shape=[%ld,%ld] fp16\n", bytes / (1024.0 * 1024.0),
                static_cast<long>(rows), static_cast<long>(cols));

    Tensor a({rows, cols}, DType::Float16); a.allocate();
    Tensor b({rows, cols}, DType::Float16); b.allocate();
    Tensor c({rows, cols}, DType::Float16); c.allocate();
    std::vector<uint16_t> host(n_fp16, 0x3c00);
    a.copy_from_host(host.data(), bytes);
    b.copy_from_host(host.data(), bytes);

    // Device-to-device copy
    {
        for (int i = 0; i < 3; ++i) {
            check_acl(aclrtMemcpyAsync(c.data(), bytes, a.data(), bytes,
                                       ACL_MEMCPY_DEVICE_TO_DEVICE, ctx.stream()), "warm");
        }
        check_acl(aclrtSynchronizeStream(ctx.stream()), "warm sync");
        auto t0 = clk::now();
        for (int i = 0; i < iters; ++i) {
            check_acl(aclrtMemcpyAsync(c.data(), bytes, a.data(), bytes,
                                       ACL_MEMCPY_DEVICE_TO_DEVICE, ctx.stream()), "d2d");
        }
        check_acl(aclrtSynchronizeStream(ctx.stream()), "d2d sync");
        auto t1 = clk::now();
        double total_ms = ms(t0, t1);
        double gbs = 2.0 * bytes * iters / (total_ms * 1e-3) / 1e9;
        std::printf("d2d_copy iters=%d total_ms=%.2f gbs=%.2f (counts R+W)\n", iters, total_ms, gbs);
    }
    // aclnnAdd elementwise (2R + 1W)
    {
        for (int i = 0; i < 3; ++i) add(a, b, c, ctx.stream());
        check_acl(aclrtSynchronizeStream(ctx.stream()), "warm add");
        auto t0 = clk::now();
        for (int i = 0; i < iters; ++i) add(a, b, c, ctx.stream());
        check_acl(aclrtSynchronizeStream(ctx.stream()), "add sync");
        auto t1 = clk::now();
        double total_ms = ms(t0, t1);
        double gbs = 3.0 * bytes * iters / (total_ms * 1e-3) / 1e9;
        std::printf("add      iters=%d total_ms=%.2f gbs=%.2f (counts 2R+1W)\n", iters, total_ms, gbs);
    }
    // H2D
    {
        auto t0 = clk::now();
        for (int i = 0; i < iters; ++i) {
            check_acl(aclrtMemcpy(a.data(), bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), "h2d");
        }
        auto t1 = clk::now();
        double total_ms = ms(t0, t1);
        double gbs = bytes * iters / (total_ms * 1e-3) / 1e9;
        std::printf("h2d      iters=%d total_ms=%.2f gbs=%.2f\n", iters, total_ms, gbs);
    }
    // D2H
    {
        auto t0 = clk::now();
        for (int i = 0; i < iters; ++i) {
            check_acl(aclrtMemcpy(host.data(), bytes, a.data(), bytes, ACL_MEMCPY_DEVICE_TO_HOST), "d2h");
        }
        auto t1 = clk::now();
        double total_ms = ms(t0, t1);
        double gbs = bytes * iters / (total_ms * 1e-3) / 1e9;
        std::printf("d2h      iters=%d total_ms=%.2f gbs=%.2f\n", iters, total_ms, gbs);
    }
    return 0;
}
