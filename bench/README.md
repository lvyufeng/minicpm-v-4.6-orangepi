# Host-side benchmarks

Standalone CPU/RAM-side diagnostics — not linked against the engine. Used to
establish hardware ceilings against which engine performance is measured.

## bench_ddr_bandwidth

NEON-vectorized read/copy/write/triad benchmark over an aligned buffer. Lets
you see the realistic host DRAM bandwidth your toolchain can extract from a
single thread or multiple threads. Used to sanity-check that the NPU side
(see `engine/build/bench_npu_bandwidth`) isn't bottlenecked by host transfers.

Build:
```bash
aarch64-linux-gnu-g++ -O3 -ftree-vectorize -march=armv8-a+simd -fopenmp \
    bench/bench_ddr_bandwidth.cpp -o bench/bench_ddr_bandwidth
```

Run (256 MiB buffer, 4 threads, 5 iters):
```bash
./bench/bench_ddr_bandwidth 256 4 5
```

Output is aggregate GB/s for sequential read, write, copy (`a → b`), and
triad (`c = a + scale * b`).
