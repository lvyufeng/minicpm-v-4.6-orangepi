# MiniCPM-V 4.6 NPU Inference Engine (Ascend 310B)

A from-scratch C++/AscendC inference engine for the MiniCPM-V 4.6 language
model, targeting the Ascend 310B NPU on an Orange Pi AIPro 20T board.

Single-batch decode on the 24-layer hybrid linear/full-attention LM (hidden
1024, vocab 248094) reaches **~5.9 tokens/s** in fp16. Starting from a stock
`aclnnMm` baseline at 2.88 tps, three rounds of cube/custom-kernel work
roughly doubled throughput:

| Stage | Tokens/s | Per-step (ms) | Saved |
|---|---:|---:|---:|
| Stock `aclnnMm` | 2.88 | 350 | — |
| + custom cube matmul (M=1) | 4.37 | 229 | 121 |
| + lm_head 16-chunk cube path | 4.99 | 200 | 29 |
| + vectorized conv1d step | **5.90** | **170** | 30 |

Measured on Orange Pi AIPro 20T (Ascend 310B), prompt_T=8, decode 30 tokens.

The remaining ~170 ms / step is dominated by matmul weight-bandwidth cost
(lm_head ~40 ms, MLP ~36 ms, linear-attn projections ~36 ms). Theoretical
floor at the measured 44 GB/s NPU bandwidth is ~32 ms, so further fp16 gains
are limited — the next lever is weight quantization.

## Layout

```
.
├── engine/                     C++ host-side inference engine
│   ├── include/minicpmv/       Public headers (tensor, ops, decoder, language_model)
│   ├── src/                    Implementation
│   ├── tests/                  Unit tests + benchmarks
│   ├── scripts/                set_env.sh, build.sh
│   └── CMakeLists.txt
├── custom_ops/                 AscendC custom NPU kernels
│   ├── op_host/                Host-side tiling + op registration
│   ├── op_kernel/              On-core kernels (AICore)
│   ├── *.json                  Op input/output descriptors
│   └── build.sh                Builds the custom_opp package
├── bench/                      Host-side diagnostics (DDR bandwidth)
└── compare_language_logits.py  PyTorch reference comparison harness
```

## Hardware

- Orange Pi AIPro 20T (Ascend 310B SoC)
- 8 AICore blocks, 192 KB UB per core
- Measured NPU bandwidth: ~44 GB/s d2d, ~27 GB/s vector R+W+R
- Stock `aclnnMm` underutilizes the cube unit at M=1; custom cube path
  recovers 2-7× per shape (see `engine/build/bench_matmul_vec`).

## Build

### 1. Build and install custom NPU operators

```bash
cd custom_ops
./build.sh
./build_out/custom_opp_ubuntu_aarch64.run \
    --install-path=$(pwd)/../custom_opp_install --quiet
```

This compiles the custom AscendC kernels (cube matmul, fused norms, vectorized
SiLU·mul, conv1d step, gated-delta-rule step, etc.) into a self-extracting
`.run` package and installs it under `custom_opp_install/vendors/customize/`.

### 2. Build the engine

```bash
cmake -S engine -B engine/build
cmake --build engine/build -j$(nproc)
```

## Run

Source the env script before running any engine binary (sets
`ASCEND_CUSTOM_OPP_PATH` and `LD_LIBRARY_PATH`):

```bash
source engine/scripts/set_env.sh

# Decode benchmark (prompt of 8 tokens, generate 30 more)
engine/build/bench_decode 8 30
# → prompt_T=8  prefill_ms=...  per_step_ms=...  tokens/s=...

# End-to-end logits regression vs PyTorch reference
python3 compare_language_logits.py

# Full forward smoke (24 layers, 4 tokens)
engine/build/test_full_language_model

# Single-step incremental decode regression
engine/build/test_incremental_decode

# Autoregressive multi-step decode
engine/build/test_autoregressive_decode
```

The model weights are expected at `MiniCPM-V-4.6/` next to this README.

## Key optimizations

### 1. Custom cube matmul kernel

`aclnnMatmulCubeCustom` (AscendC `MatmulImpl`, `custom_ops/op_kernel/matmul_cube_custom.cpp`)
beats stock `aclnnMm` 2-7× at M=1 for the decode shapes we hit. The cube unit
is genuinely well-tuned even for vector-matrix multiply — a pure vector-unit
implementation can't match it (see negative-result kernel
`custom_ops/op_kernel/matmul_vec_custom.cpp` and `bench_matmul_vec`).

`engine/src/ops.cpp` wraps both via `matmul_b_transposed`. The cube fast path
requires `N <= 16384 && N % 128 == 0` and `M == 1`; it accepts both legacy
`[N, K]` and natural `[K, N]` weight layouts. Wide MLP/attn weights are
pre-transposed at load time to `[K, N]` via `load_matmul_weight_transposed`.

### 2. lm_head chunking

The vocab-projection matmul is `[1, 1024] × [1024, 248094]` — too large for
the cube tiling (which breaks above ~32k). `load_language_model_weights`
pre-builds 16 cube-friendly `[K=1024, N=16384]` chunks of the lm_head weight
(last chunk zero-padded out to 16384); padded columns produce a 0 logit that
`lm_head_greedy` ignores by clipping the host argmax to `start_vocab + valid`.

Decode iterates chunks, runs the cube matmul on each, D2H-copies the 32 KB
chunk logits, and reduces argmax on host. lm_head went from ~75 ms (aclnnMm)
to ~40 ms (16 × cube).

### 3. Vectorized conv1d step kernel

The generic `linear_causal_conv` kernel was scalar (`GetValue`/`SetValue`
per element) AND computed all 4 output rows even though decode only uses the
last row — costing ~32 ms / step.

`custom_ops/op_kernel/linear_causal_conv_step_custom.cpp` takes a
pre-transposed `[K=4, C]` weight (built once at load by
`build_conv_step_weight`), DMA's 4 input rows + 4 weight rows into UB as
contiguous tiles, does fp16 vector mul+add (4 muls + 3 adds), and writes a
single `[1, C]` row directly. Wired in via the optional
`LinearAttentionDecoderLayerWeights::conv1d_step_weight` pointer so prefill
paths and manual-weight tests keep using the generic kernel.

### Other custom kernels

`custom_ops/` also contains:

- `rms_norm1024_custom` — fused RMSNorm for the hidden-1024 shape
- `gated_rms_norm_z_custom` — per-head gated RMSNorm with z gate for linear-attn
- `silu_mul_custom` — fused SwiGLU activation
- `attention_step_custom` — single-token full-attention (softmax(q·K)·V)
- `linear_gated_delta_rule_custom` / `_step_custom` — gated-delta-rule recurrence

## Diagnostics

End-to-end:

- `engine/build/bench_decode prompt_T decode_N` — tokens/s + per-step latency
- `engine/build/bench_prefill` — multi-token prefill latency
- `engine/build/bench_decode_subops` — per-op micro-benches at decode shapes

Per-op:

- `engine/build/bench_matmul_throughput M N K iters` — aclnnMm latency by shape
- `engine/build/bench_matmul_vec` — cube vs vector matmul comparison
- `engine/build/bench_step_kernel` — gated-delta-rule step kernel
- `engine/build/bench_npu_bandwidth` — NPU memcpy / add / H2D / D2H GB/s
- `bench/bench_ddr_bandwidth` — host DRAM bandwidth (see `bench/README.md`)

Correctness:

- `engine/build/test_natural_b_matmul` — `[K,N]` natural vs `[N,K]` transposed B
- `engine/build/test_linear_causal_conv_step` — new conv kernel vs generic
- `engine/build/test_matmul_cube` — cube matmul vs reference
- `engine/build/test_lm_head_pipeline` — lm_head end-to-end
- `python3 compare_language_logits.py` — engine logits vs PyTorch reference

## What's next

Bandwidth-bound. At ~44 GB/s NPU memory bandwidth, lm_head's 508 MB weight
read alone has a 11.5 ms floor. The cube path is at ~3× that floor on most
ops — fixed overhead, not something user-mode kernel work can close further.

The single remaining lever is **weight quantization**: int8 halves the bytes
read for every matmul → roughly halves lm_head + MLP + attn-proj cost →
estimated ~10 tps. int4 → ~14 tps. This would need new dequant-fused matmul
kernels on the cube path.
