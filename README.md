# MiniCPM-V 4.6 on Ascend 310B (Orange Pi AIPro 20T)

A from-scratch C++/AscendC inference engine for the [MiniCPM-V 4.6][minicpmv]
language model, targeting the Ascend 310B NPU on Orange Pi AIPro 20T boards
where stock toolchains underutilize the hardware.

Three rounds of cube-unit / custom-kernel work took single-batch decode from
**2.88 → 5.90 tokens/s** (~2×) on the full 24-layer hybrid linear/full
attention LM (hidden 1024, vocab 248094, fp16):

| Stage | Tokens/s | Per-step (ms) | Saved |
|---|---:|---:|---:|
| Stock `aclnnMm` baseline | 2.88 | 350 | — |
| + custom cube matmul (M=1) | 4.37 | 229 | 121 |
| + lm_head 16-chunk cube path | 4.99 | 200 | 29 |
| + vectorized causal-conv1d step kernel | **5.90** | **170** | 30 |

Measured at prompt_T=8, decode 30 tokens. The remaining ~170 ms / step is
dominated by matmul weight-bandwidth cost; the next lever is weight
quantization (see [Roadmap](#roadmap)).

> ⚠ Single-batch decode only. The engine currently does greedy sampling and
> targets a specific layer schedule (24 layers, 3:1 linear:full attention).
> Different MiniCPM-V variants need config edits in `language_model.cpp`.

## Hardware & software prerequisites

- **Board**: Orange Pi AIPro 20T (or any Ascend 310B-based device)
- **OS**: Ubuntu 22.04 aarch64
- **CANN toolkit**: ≥ 7.0 at `/usr/local/Ascend/ascend-toolkit/latest/` (set
  `MINICPMV_ASCEND_TOOLKIT_ROOT` if elsewhere)
- **CMake**: ≥ 3.20
- **Python**: 3.8+ with `torch`, `torch_npu`, `transformers` (only needed for
  the Python helpers in `src/python/`, not the engine itself)

## Layout

```
.
├── README.md                    
├── LICENSE                       Apache-2.0
├── CMakeLists.txt                Engine library + per-test/per-bench targets
├── scripts/
│   ├── build.sh                  Configure + build the engine into build/
│   ├── install_custom_ops.sh     Build + install the AscendC custom ops
│   └── set_env.sh                Sources Ascend toolkit + custom_opp paths
├── src/
│   ├── csrc/                     C++ source tree
│   │   ├── include/minicpmv/     Public headers (tensor, ops, layers, lm)
│   │   ├── lib/                  Engine library implementation (.cpp)
│   │   ├── tools/                User-runnable binaries (hybrid runner)
│   │   └── custom_ops/           AscendC custom NPU kernels (own build system)
│   └── python/                   Python helpers
│       ├── run_hybrid.py         Python prompt prep + engine decode loop
│       └── compare_logits.py     PyTorch reference vs engine logits diff
├── tests/                        C++ unit & regression tests (28 files)
└── bench/                        C++ benchmarks (8 NPU + 1 host DDR)
```

`MiniCPM-V-4.6/` (model weights, ~5 GB) is gitignored — see
[Download the model](#download-the-model).

## Build

```bash
# 1. Build and install the AscendC custom operators (cube matmul, fused
#    norms, vectorized SiLU·mul, conv1d step, gated-delta-rule step, etc.).
#    Output: ./custom_opp_install/vendors/customize/
./scripts/install_custom_ops.sh

# 2. Build the engine (library + ~40 binaries) into ./build/
./scripts/build.sh
```

If your Ascend toolkit lives elsewhere, set
`MINICPMV_ASCEND_TOOLKIT_ROOT=/path/to/aarch64-linux` before running
`set_env.sh`, and pass `-DASCEND_TOOLKIT_ROOT=...` to cmake.

## Download the model

```bash
# Either git-lfs clone:
git clone https://huggingface.co/openbmb/MiniCPM-V-4_6 MiniCPM-V-4.6

# Or pull a specific safetensors revision via huggingface-cli:
huggingface-cli download openbmb/MiniCPM-V-4_6 \
    --local-dir MiniCPM-V-4.6 --local-dir-use-symlinks False
```

The binaries default to `./MiniCPM-V-4.6/model.safetensors` (relative to the
process CWD). Override with `MINICPMV_MODEL_PATH=/elsewhere/MiniCPM-V-4.6`.

## Run

```bash
# Source the env script once per shell. Exports ASCEND_CUSTOM_OPP_PATH so the
# custom kernels are discovered by ACL.
source scripts/set_env.sh

# Decode benchmark — primary tps number.
./build/bench_decode 8 30
# → prompt_T=8  prefill_ms=...  per_step_ms=...  tokens/s=...

# End-to-end greedy generation, text-only:
python3 src/python/run_hybrid.py --prompt "你好,介绍一下你自己"

# End-to-end with an image (multimodal):
python3 src/python/run_hybrid.py \
    --prompt "What's in this image?" --image path/to/img.png

# Compare engine logits against a PyTorch reference (regression sanity):
python3 src/python/compare_logits.py
```

For unit tests and the full bench suite, see [Diagnostics](#diagnostics).

## Key optimizations

### 1. Custom cube matmul at M=1

Stock `aclnnMm` underutilizes the cube unit at M=1 (vector-matrix multiply),
which is the dominant decode shape. The custom `aclnnMatmulCubeCustom` kernel
[(`src/csrc/custom_ops/op_kernel/matmul_cube_custom.cpp`)][cube-kernel] uses
AscendC `MatmulImpl` directly and beats `aclnnMm` 2-7× across the shapes we
hit:

| N (output) | Cube vs `aclnnMm` |
|---|---|
| 512  | 3.4× |
| 2048 | 7.3× |
| 3584 | 2.9× |
| 6144 | 2.4× |

`matmul_b_transposed` dispatches to the cube path when `N ≤ 16384 && N % 128
== 0 && M == 1`, with both legacy `[N, K]` and pre-transposed `[K, N]` weight
layouts supported. Wide weights are pre-transposed once at load time via
`load_matmul_weight_transposed` to keep the hot path on the cube.

A purely vector-unit alternative was tried (`MatmulVecCustom`) and is
**30-50% slower than `aclnnMm` at every shape** — kept in-tree as a negative
result. The cube unit really is the right answer here.

### 2. lm_head chunking

The lm_head matmul is `[1, 1024] × [1024, 248094]` — too wide for the cube
tiling (which breaks above ~32k). At load time the engine pre-builds 16
cube-friendly `[K=1024, N=16384]` chunks of the lm_head weight (last chunk
zero-padded); decode runs 16 sequential cube matmuls + a host-side argmax
reduce.

Took the lm_head matmul from ~75 ms (`aclnnMm` slow path) to ~40 ms.

### 3. Vectorized causal-conv1d step kernel

The generic `linear_causal_conv` kernel was completely scalar
(`GetValue`/`SetValue` per element) **and** computed all 4 output rows even
though the decode step only consumes the last row. ~32 ms / step wasted.

`linear_causal_conv_step` takes a pre-transposed `[K=4, C]` weight, DMA's 4
input rows + 4 weight rows into UB as contiguous tiles, does fp16 vector
mul+add (4 mul + 3 add), and writes a single `[1, C]` output row. ~30 ms
saved.

Wired through an optional `conv1d_step_weight` field so prefill paths and
manual-weight tests keep using the generic kernel for correctness.

### Other custom kernels

- `rms_norm1024` / `gated_rms_norm_z` — fused RMSNorm variants
- `silu_mul` — fused SwiGLU activation
- `attention_step` — single-token full-attention (softmax(q·K)·V) per head
- `linear_gated_delta_rule` (+ `_step`) — gated-delta-rule recurrence

## Diagnostics

End-to-end:

| Binary | Purpose |
|---|---|
| `./build/bench_decode prompt_T decode_N` | tokens/s + per-step latency |
| `./build/bench_prefill` | Multi-token prefill latency |
| `./build/bench_decode_subops` | Per-op micro-benches at decode shapes |
| `python3 src/python/run_hybrid.py` | End-to-end greedy generation |
| `python3 src/python/compare_logits.py` | Logits diff vs PyTorch reference |

Per-op:

| Binary | Purpose |
|---|---|
| `./build/bench_matmul_throughput M N K iters` | aclnnMm latency by shape |
| `./build/bench_matmul_vec` | cube vs vector matmul comparison |
| `./build/bench_step_kernel` | gated-delta-rule step kernel |
| `./build/bench_npu_bandwidth` | NPU memcpy / add / H2D / D2H GB/s |
| `./bench/bench_ddr_bandwidth` | host DRAM bandwidth (see `bench/README.md`) |

Correctness:

| Binary | Purpose |
|---|---|
| `./build/test_natural_b_matmul` | `[K,N]` natural vs `[N,K]` view |
| `./build/test_linear_causal_conv_step` | new conv kernel vs generic |
| `./build/test_matmul_cube` | cube matmul vs reference |
| `./build/test_prefill_from_embeddings` | prefill regression |
| `./build/test_autoregressive_decode` | multi-step decode regression |
| `./build/test_incremental_decode` | single-step decode regression |
| `./build/test_full_language_model` | 24-layer forward smoke |

## Performance breakdown

Per-step time (~170 ms) at 5.90 tps, decode T=1:

| Component | Cost | Share |
|---|---:|---:|
| lm_head (16 cube chunks) | ~40 ms | 24% |
| 24 × MLP cube matmuls (gate+up+down) | ~36 ms | 21% |
| 18 × linear-attn matmuls (qkv+z+ab+out) | ~36 ms | 21% |
| 18 × linear-attn gated-delta-rule step | ~20 ms | 12% |
| 48 × rms_norm calls | ~17 ms | 10% |
| 6 × full-attn matmuls + attention_step | ~15 ms | 9% |
| conv1d_step + silu/mul/sigmoid/copies | ~10 ms | 6% |

Theoretical floor at the measured ~44 GB/s NPU d2d bandwidth is **~32 ms**
(1408 MB weight read per step). The cube path sits at ~5.3× that floor —
mostly fixed kernel overhead at M=1, not something further user-mode work
can close.

## Roadmap

- **Weight quantization** is the only remaining lever for serious tps gains.
  int8 halves the bytes read for every matmul → estimated ~10 tps. int4 →
  estimated ~14 tps. Will need dequant-fused matmul kernels on the cube path.
- KV-cache offload to host for longer contexts (currently bounded by NPU memory).
- Beam search / top-p sampling. Currently greedy-only.

## License

Apache License 2.0. See [`LICENSE`](LICENSE).

The MiniCPM-V model weights are released by [OpenBMB][openbmb] under their
own license; please respect the upstream terms when redistributing.

## Acknowledgements

- [OpenBMB / MiniCPM-V team][minicpmv] for the model
- Huawei Ascend AscendC sample code as the baseline for the custom op build
  system (`src/csrc/custom_ops/cmake/`, `framework/`, `scripts/`)

[minicpmv]: https://huggingface.co/openbmb/MiniCPM-V-4_6
[openbmb]: https://github.com/OpenBMB
[cube-kernel]: src/csrc/custom_ops/op_kernel/matmul_cube_custom.cpp
