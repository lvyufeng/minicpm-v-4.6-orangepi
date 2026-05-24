// Simple host DDR bandwidth benchmark, NEON-vectorized read/copy/write/triad.
// Compile: aarch64-linux-gnu-g++ -O3 -ftree-vectorize -march=armv8-a+simd -fopenmp
//          bench_ddr_bandwidth.cpp -o bench_ddr_bandwidth
#include <arm_neon.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using clk = std::chrono::steady_clock;

static double sec(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

static void touch_pages(void* p, size_t bytes) {
    auto* q = static_cast<volatile uint8_t*>(p);
    for (size_t i = 0; i < bytes; i += 4096) q[i] = 0;
}

static double bench_read(const float* a, size_t n, int iters) {
    auto t0 = clk::now();
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    for (int it = 0; it < iters; ++it) {
        for (size_t i = 0; i + 8 <= n; i += 8) {
            acc0 = vaddq_f32(acc0, vld1q_f32(a + i));
            acc1 = vaddq_f32(acc1, vld1q_f32(a + i + 4));
        }
    }
    auto t1 = clk::now();
    volatile float sink = vaddvq_f32(acc0) + vaddvq_f32(acc1);
    (void)sink;
    double total_bytes = static_cast<double>(n) * sizeof(float) * iters;
    return total_bytes / sec(t0, t1);
}

static double bench_write(float* a, size_t n, int iters) {
    float32x4_t v = vdupq_n_f32(1.5f);
    auto t0 = clk::now();
    for (int it = 0; it < iters; ++it) {
        float32x4_t vv = vaddq_f32(v, vdupq_n_f32(static_cast<float>(it)));
        for (size_t i = 0; i + 8 <= n; i += 8) {
            vst1q_f32(a + i, vv);
            vst1q_f32(a + i + 4, vv);
        }
    }
    auto t1 = clk::now();
    double total_bytes = static_cast<double>(n) * sizeof(float) * iters;
    return total_bytes / sec(t0, t1);
}

static double bench_copy(const float* a, float* b, size_t n, int iters) {
    auto t0 = clk::now();
    for (int it = 0; it < iters; ++it) {
        for (size_t i = 0; i + 8 <= n; i += 8) {
            float32x4_t v0 = vld1q_f32(a + i);
            float32x4_t v1 = vld1q_f32(a + i + 4);
            vst1q_f32(b + i, v0);
            vst1q_f32(b + i + 4, v1);
        }
    }
    auto t1 = clk::now();
    double total_bytes = static_cast<double>(n) * sizeof(float) * iters * 2.0;
    return total_bytes / sec(t0, t1);
}

static double bench_triad(const float* a, const float* b, float* c, size_t n, int iters) {
    float32x4_t scale = vdupq_n_f32(1.0001f);
    auto t0 = clk::now();
    for (int it = 0; it < iters; ++it) {
        for (size_t i = 0; i + 8 <= n; i += 8) {
            float32x4_t va0 = vld1q_f32(a + i);
            float32x4_t va1 = vld1q_f32(a + i + 4);
            float32x4_t vb0 = vld1q_f32(b + i);
            float32x4_t vb1 = vld1q_f32(b + i + 4);
            vst1q_f32(c + i, vmlaq_f32(va0, vb0, scale));
            vst1q_f32(c + i + 4, vmlaq_f32(va1, vb1, scale));
        }
    }
    auto t1 = clk::now();
    double total_bytes = static_cast<double>(n) * sizeof(float) * iters * 3.0;
    return total_bytes / sec(t0, t1);
}

int main(int argc, char** argv) {
    size_t bytes = (argc > 1) ? std::stoull(argv[1]) * (1ULL << 20) : 256ULL << 20;
    int threads = (argc > 2) ? std::atoi(argv[2]) : static_cast<int>(std::thread::hardware_concurrency());
    int iters = (argc > 3) ? std::atoi(argv[3]) : 5;
    size_t n = bytes / sizeof(float);
    std::printf("buffer=%.1f MiB threads=%d iters=%d\n", bytes / (1024.0 * 1024.0), threads, iters);

    auto alloc = [](size_t n) {
        void* p = nullptr;
        if (posix_memalign(&p, 64, n * sizeof(float)) != 0) std::abort();
        return static_cast<float*>(p);
    };

    auto run = [&](const char* name, auto fn) {
        std::vector<float*> bufs;
        std::vector<double> bws(threads, 0.0);
        std::vector<std::thread> ts;
        ts.reserve(threads);
        for (int t = 0; t < threads; ++t) {
            ts.emplace_back([&, t]() {
                fn(t, bws[t]);
            });
        }
        for (auto& th : ts) th.join();
        double total = 0.0;
        for (auto bw : bws) total += bw;
        std::printf("%-8s aggregate=%.2f GB/s (per-thread avg %.2f GB/s)\n",
                    name, total / 1e9, total / 1e9 / threads);
    };

    run("read", [&](int /*tid*/, double& out) {
        float* a = alloc(n);
        touch_pages(a, n * sizeof(float));
        std::memset(a, 1, n * sizeof(float));
        out = bench_read(a, n, iters);
        std::free(a);
    });
    run("write", [&](int /*tid*/, double& out) {
        float* a = alloc(n);
        touch_pages(a, n * sizeof(float));
        out = bench_write(a, n, iters);
        std::free(a);
    });
    run("copy", [&](int /*tid*/, double& out) {
        float* a = alloc(n);
        float* b = alloc(n);
        touch_pages(a, n * sizeof(float));
        touch_pages(b, n * sizeof(float));
        std::memset(a, 1, n * sizeof(float));
        out = bench_copy(a, b, n, iters);
        std::free(a);
        std::free(b);
    });
    run("triad", [&](int /*tid*/, double& out) {
        float* a = alloc(n);
        float* b = alloc(n);
        float* c = alloc(n);
        touch_pages(a, n * sizeof(float));
        touch_pages(b, n * sizeof(float));
        touch_pages(c, n * sizeof(float));
        std::memset(a, 1, n * sizeof(float));
        std::memset(b, 1, n * sizeof(float));
        out = bench_triad(a, b, c, n, iters);
        std::free(a);
        std::free(b);
        std::free(c);
    });
    return 0;
}
