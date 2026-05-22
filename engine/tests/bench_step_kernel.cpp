// Microbench: how fast is linear_gated_delta_rule_step on its own?
#include "minicpmv/acl_context.h"
#include "minicpmv/ops.h"
#include "minicpmv/tensor.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace minicpmv;
using clk = std::chrono::steady_clock;

static double ms(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

int main(int argc, char** argv) {
    int iters = (argc > 1) ? std::atoi(argv[1]) : 1152;

    AclContext ctx(0);
    Tensor mixed({1, 6144}, DType::Float16); mixed.allocate();
    Tensor beta({1, 16}, DType::Float16);    beta.allocate();
    Tensor decay({1, 16}, DType::Float16);   decay.allocate();
    Tensor state({16, 128, 128}, DType::Float32); state.allocate();
    Tensor scratch({8 * 6 * 128}, DType::Float32); scratch.allocate();
    Tensor out({1, 2048}, DType::Float16);   out.allocate();

    // Zero everything.
    std::vector<uint16_t> mz(6144, 0); mixed.copy_from_host(mz.data(), mz.size() * sizeof(uint16_t));
    std::vector<uint16_t> bz(16, 0x3c00); beta.copy_from_host(bz.data(), bz.size() * sizeof(uint16_t));
    decay.copy_from_host(bz.data(), bz.size() * sizeof(uint16_t));
    std::vector<float> sz(static_cast<size_t>(16 * 128 * 128), 0.0f);
    state.copy_from_host(sz.data(), sz.size() * sizeof(float));
    std::vector<float> sc(static_cast<size_t>(8 * 6 * 128), 0.0f);
    scratch.copy_from_host(sc.data(), sc.size() * sizeof(float));

    // Warm-up.
    for (int i = 0; i < 5; ++i) {
        linear_gated_delta_rule_step(mixed, beta, decay, state, scratch, out, ctx.stream());
    }
    check_acl(aclrtSynchronizeStream(ctx.stream()), "warmup sync");

    auto t0 = clk::now();
    for (int i = 0; i < iters; ++i) {
        linear_gated_delta_rule_step(mixed, beta, decay, state, scratch, out, ctx.stream());
    }
    check_acl(aclrtSynchronizeStream(ctx.stream()), "bench sync");
    auto t1 = clk::now();

    double total = ms(t0, t1);
    std::printf("linear_gated_delta_rule_step iters=%d total_ms=%.2f per_call_ms=%.3f\n",
                iters, total, total / iters);

    // Also try: launch many calls WITHOUT host-side syncing between (the run_op
    // inside has its own sync though).
    auto t2 = clk::now();
    for (int i = 0; i < iters; ++i) {
        linear_gated_delta_rule_step(mixed, beta, decay, state, scratch, out, ctx.stream());
    }
    auto t3 = clk::now();
    check_acl(aclrtSynchronizeStream(ctx.stream()), "final sync");
    auto t4 = clk::now();
    std::printf("loop_only_ms=%.2f tail_sync_ms=%.2f total_ms=%.2f\n",
                ms(t2, t3), ms(t3, t4), ms(t2, t4));

    return 0;
}
