// Quick per-op probe of linear-attention decode step subcomponents at T=1.
// Times: linear_causal_conv on [4, 6144], the dt_bias/a_log host round-trip,
// and the step kernel on its own.
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

int main() {
    AclContext ctx(0);

    // linear_causal_conv on the step-shaped input [4, 6144].
    {
        Tensor x({4, 6144}, DType::Float16); x.allocate();
        Tensor w({6144, 1, 4}, DType::Float16); w.allocate();
        Tensor out({4, 6144}, DType::Float16); out.allocate();
        std::vector<uint16_t> hx(4 * 6144, 0x3c00), hw(6144 * 4, 0x3c00);
        x.copy_from_host(hx.data(), hx.size() * sizeof(uint16_t));
        w.copy_from_host(hw.data(), hw.size() * sizeof(uint16_t));

        for (int i = 0; i < 5; ++i) linear_causal_conv(x, w, out, ctx.stream());
        check_acl(aclrtSynchronizeStream(ctx.stream()), "warm");
        auto t0 = clk::now();
        const int iters = 200;
        for (int i = 0; i < iters; ++i) linear_causal_conv(x, w, out, ctx.stream());
        check_acl(aclrtSynchronizeStream(ctx.stream()), "bench");
        auto t1 = clk::now();
        std::printf("linear_causal_conv [4,6144]    per_call_ms=%.3f  (%d iters)\n",
                    ms(t0, t1) / iters, iters);
    }

    // Host round-trip: 4 copy_to_host of 16 fp16 each + host math + 2 copy_from_host of 16 fp16.
    {
        Tensor a({1, 16}, DType::Float16); a.allocate();
        Tensor b({1, 16}, DType::Float16); b.allocate();
        Tensor dt({16}, DType::Float16); dt.allocate();
        Tensor al({16}, DType::Float16); al.allocate();
        Tensor beta({1, 16}, DType::Float16); beta.allocate();
        Tensor decay({1, 16}, DType::Float16); decay.allocate();
        std::vector<uint16_t> h16(16, 0);
        a.copy_from_host(h16.data(), 32);
        b.copy_from_host(h16.data(), 32);
        dt.copy_from_host(h16.data(), 32);
        al.copy_from_host(h16.data(), 32);

        auto round_trip = [&]() {
            check_acl(aclrtSynchronizeStream(ctx.stream()), "sync");
            std::vector<uint16_t> ah(16), bh(16), dh(16), lh(16);
            a.copy_to_host(ah.data(), 32);
            b.copy_to_host(bh.data(), 32);
            dt.copy_to_host(dh.data(), 32);
            al.copy_to_host(lh.data(), 32);
            // host arith (negligible)
            std::vector<uint16_t> betah(16, 0), decayh(16, 0);
            beta.copy_from_host(betah.data(), 32);
            decay.copy_from_host(decayh.data(), 32);
        };
        for (int i = 0; i < 5; ++i) round_trip();
        auto t0 = clk::now();
        const int iters = 500;
        for (int i = 0; i < iters; ++i) round_trip();
        auto t1 = clk::now();
        std::printf("beta_decay host round-trip       per_call_ms=%.3f  (%d iters)\n",
                    ms(t0, t1) / iters, iters);
    }

    // rms_norm on [1, 1024] (decode hidden).
    {
        Tensor x({1, 1024}, DType::Float16); x.allocate();
        Tensor g({1024}, DType::Float16); g.allocate();
        Tensor o({1, 1024}, DType::Float16); o.allocate();
        std::vector<uint16_t> h(1024, 0x3c00);
        x.copy_from_host(h.data(), 2048);
        g.copy_from_host(h.data(), 2048);
        for (int i = 0; i < 5; ++i) rms_norm(x, g, o, 1e-6, ctx.stream());
        check_acl(aclrtSynchronizeStream(ctx.stream()), "warm");
        auto t0 = clk::now();
        const int iters = 500;
        for (int i = 0; i < iters; ++i) rms_norm(x, g, o, 1e-6, ctx.stream());
        check_acl(aclrtSynchronizeStream(ctx.stream()), "bench");
        auto t1 = clk::now();
        std::printf("rms_norm [1, 1024]               per_call_ms=%.3f  (%d iters)\n",
                    ms(t0, t1) / iters, iters);
    }

    // attention_step (custom) at full-attention decode shape:
    //   q [8, 256], k_cache [maxSeq, 512], v_cache [maxSeq, 512], context varies.
    {
        const int64_t MaxSeq = 256;
        Tensor q({8, 256}, DType::Float16); q.allocate();
        Tensor kc({MaxSeq, 512}, DType::Float16); kc.allocate();
        Tensor vc({MaxSeq, 512}, DType::Float16); vc.allocate();
        Tensor o({1, 2048}, DType::Float16); o.allocate();
        std::vector<uint16_t> hq(8 * 256, 0x3c00), hkv(MaxSeq * 512, 0x3c00);
        q.copy_from_host(hq.data(), hq.size() * 2);
        kc.copy_from_host(hkv.data(), hkv.size() * 2);
        vc.copy_from_host(hkv.data(), hkv.size() * 2);

        for (int ctxLen : {16, 64, 256}) {
            for (int i = 0; i < 5; ++i) {
                incre_flash_attention(q, kc, vc, ctxLen, 8, 2, 256, 0.0625f, o, ctx.stream());
            }
            check_acl(aclrtSynchronizeStream(ctx.stream()), "warm");
            auto t0 = clk::now();
            const int iters = 200;
            for (int i = 0; i < iters; ++i) {
                incre_flash_attention(q, kc, vc, ctxLen, 8, 2, 256, 0.0625f, o, ctx.stream());
            }
            check_acl(aclrtSynchronizeStream(ctx.stream()), "bench");
            auto t1 = clk::now();
            std::printf("attention_step ctx=%-4d         per_call_ms=%.3f  (%d iters)\n",
                        ctxLen, ms(t0, t1) / iters, iters);
        }
    }
    return 0;
}
