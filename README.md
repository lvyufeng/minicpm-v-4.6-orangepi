# MiniCPM-V 4.6 on Ascend 310B (Orange Pi AIPro 20T)

[English](README.md) · [中文](README_ZH.md)

A from-scratch C++/AscendC inference engine for [MiniCPM-V 4.6][minicpmv]
on the Ascend 310B NPU in the Orange Pi AIPro 20T board. **Text and image
chat both run entirely on the NPU through one engine subprocess — Python
only does CPU-side tokenization and image-preprocessing, with no
`torch_npu` dependency on the hot path.**

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

The vision tower (SigLIP-so400m → vit_merger → projection to LM hidden,
27 transformer layers total) is also ported to C++ / aclnn and validated
end-to-end against the HF CPU reference: final LM-facing `image_features`
match to within `max_abs_diff = 0.0098` for a 448×448 input.

> ⚠ Single-batch greedy decode only. Vision currently runs **one** 448×448
> slice per image (HF MiniCPM-V's high-res multi-slice pipeline is on the
> [Roadmap](#roadmap)). Different MiniCPM-V variants need config edits in
> `language_model.cpp` and `vision.cpp`.

## Quickstart

End-to-end from a fresh clone, assuming an Orange Pi AIPro 20T with CANN
already installed under `/usr/local/Ascend/ascend-toolkit/latest`:

```bash
# 0) install Python deps for the processor + Gradio (one-time).
pip install transformers pillow gradio

# 1) build & install the AscendC custom operators (cube matmul, fused
#    SiLU·mul, etc.). Output: ./custom_opp_install/
./scripts/install_custom_ops.sh

# 2) build the engine (library + ~40 tests/benches/tools). Output: ./build/
./scripts/build.sh

# 3) download model weights (~5 GB) next to the repo.
git lfs install
git clone https://huggingface.co/openbmb/MiniCPM-V-4_6 MiniCPM-V-4.6

# 4) source env and launch the multimodal web UI.
source scripts/set_env.sh
python3 src/python/gradio_app.py
# → open http://localhost:7860, drop an image + ask a question.
```

Cold-start expectations: step 1 is ~3 min, step 2 ~1 min, step 4 takes
~75 s to load model weights to NPU before serving the first request. Every
request after that is GPU-like — no per-shape JIT cliff.

If your CANN install lives elsewhere, set
`MINICPMV_ASCEND_TOOLKIT_ROOT=/path/to/aarch64-linux` before step 1 and
pass `-DASCEND_TOOLKIT_ROOT=...` to cmake in step 2.

## Hardware & software prerequisites

- **Board**: Orange Pi AIPro 20T (or any Ascend 310B-based device)
- **OS**: Ubuntu 22.04 aarch64
- **CANN toolkit**: 8.3.RC2 at `/usr/local/Ascend/ascend-toolkit/latest/`
  (override with `MINICPMV_ASCEND_TOOLKIT_ROOT`)
- **CMake**: ≥ 3.20
- **Python**: 3.8+ with `transformers` ≥ 4.46 (for `MiniCPMV4_6Processor`)
  and `pillow`. `gradio` for the web UI. `torch` / `torch_npu` are not
  required on the inference path — they may be transitively installed by
  `transformers` but no NPU op runs in Python.

## Build

If you skipped the Quickstart and want to do the steps manually:

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

The primary surface is the [Gradio Web UI](#web-ui-gradio) below. CLI helpers
and benchmarks are documented under [Diagnostics](#diagnostics).

### Web UI (Gradio)

Streaming multimodal chat over the C++ engine — text + image (single-slice
448×448), no torch_npu required:

```bash
# One-time: install Gradio in your Python env (alongside the existing
# transformers / pillow you already have for the processor).
pip install gradio pillow

source scripts/set_env.sh
python3 src/python/gradio_app.py            # text + vision (default)
# or:
python3 src/python/gradio_app.py --text-only  # skip the ~1.5 GB vision-tower
                                              # load if you only need text chat
```

Open <http://localhost:7860>. Drop an image into the message box alongside
your prompt and the C++ engine runs the entire SigLIP vision tower on the
NPU before prefill. Each response streams token-by-token; the footer reports
time-to-first-token and decode tokens/s.

Startup time:

| Mode | First-ready cost (one-shot) |
|---|---|
| `--text-only` | ~50 s (LM weights → NPU) + ~2 s warmup |
| default (with vision) | ~75 s (LM + SigLIP → NPU) + ~2 s warmup |

After that, **any new prompt shape is GPU-like** — no per-shape JIT cliff
(see [SigLIP vision tower in C++](#4-siglip-vision-tower-in-c-no-torch_npu)
for how Python avoids the per-shape `torch_npu` JIT compile that previously
dominated first-token latency).

> ⚠ **Vision is single-slice for now.** The HF MiniCPM-V processor can split
> high-res images into up to 9 sub-images for finer detail. The engine
> currently runs the SigLIP tower over **one** 448×448 slice (32×32 patch
> grid → 64 image tokens), and the Python side passes `max_slice_nums=1` to
> the processor so token counts line up. High-res detail recognition is
> reduced; multi-slice scatter is on the roadmap.

### CLI

```bash
source scripts/set_env.sh

# Decode benchmark — primary tps number.
./build/bench_decode 8 30
# → prompt_T=8  prefill_ms=...  per_step_ms=...  tokens/s=...

# Greedy generation from a text prompt.
python3 src/python/run_hybrid.py --prompt "你好,介绍一下你自己"

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

### 4. SigLIP vision tower in C++ (no torch_npu)

The whole vision path runs on the NPU through the same engine subprocess as
the LM. Python only does CPU-side tokenization and image-preprocessing via
HF's `AutoProcessor` — no `torch_npu` import, no HF model load.

Pipeline implemented in `src/csrc/lib/vision.cpp`:

```
pixel_values [1, 3, 14, P·14]  (HF naflex patch-strip layout)
   ↓  vision_patch_embed       (aclnnConvolution)
   ↓  vision_position_embed    (4900-entry table indexed via host bucketize)
   ↓  encoder_layer × 7        (SigLIP: pre-LN attn + GELU(tanh) MLP)
   ↓  vit_merger               (2×2 window attn + 4-patch concat MLP)
   ↓  encoder_layer × 20
   ↓  layer_norm (post)
   ↓  merger_mlp               (2×2 spatial concat + pre-LN + linear + GELU + linear)
   ↓  image_features [1, P/16, 1024]  ← LM-facing tokens
```

For a 448×448 image (32×32 patch grid) this is 1024 → 256 → 64 image
tokens. Validated against a CPU torch reference end-to-end:

| Stage | max abs diff vs HF |
|---|---:|
| patch_embed | 0.0001 |
| 7-layer encoder stack | 0.0273 |
| vit_merger | 0.0498 |
| post_layernorm | 0.0469 |
| **image_features** | **0.0098** |

Two ACL gotchas surfaced and fixed during the port:

- CANN's precompiled `MatMulV2_FP16` binary returns `kernel pointer null`
  (errno 361001) for shapes with `M >= 64 && K > 4096`. `matmul_b_transposed`
  now K-tiles to 4096 chunks and accumulates partials for those shapes.
- Plain `aclnnAdd` with input and output aliased to the same buffer leaves
  the MatMulV2 kernel cache in a bad state, breaking subsequent `aclnnMm`
  calls. `linear_bias` uses `aclnnInplaceAdd` (the dedicated in-place API)
  for the bias add.

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
| `./build/test_vision_ops` | conv2d / gelu / batch_matmul standalone |
| `./build/test_vision_patch_embed` | full vision tower vs HF (needs `/tmp/dump_vision_ref.py` ref dump) |

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

- **Multi-slice vision** so high-res images keep their detail. The HF
  MiniCPM-V processor splits images into up to 9 sub-images at full
  resolution; the engine currently runs the SigLIP tower on a single 448×448
  slice (~64 image tokens). Need to loop the tower N times and stitch
  features into the prompt per slice.
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
