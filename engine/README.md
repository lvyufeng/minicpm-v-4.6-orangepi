# MiniCPM-V Engine

C++ inference engine for the MiniCPM-V 4.6 language model on Ascend 310B.
See the [project README](../README.md) for an overview, performance numbers,
and build instructions.

## Layout

- `include/minicpmv/` — public headers
  - `tensor.h` — `Tensor` (device-resident buffer + shape/dtype) and `DType`
  - `ops.h` — all device ops (matmul, rms_norm, silu_mul, rope, conv1d,
    linear/gated-delta-rule recurrence, attention step, etc.)
  - `decoder_layer.h` — full-attn + linear-attn decoder layers, prefill and
    incremental-decode entry points, `DecodeState` for KV/recurrent caches
  - `language_model.h` — `LanguageModelConfig`, weight container, prefill
    and decode entry points, lm_head
  - `weights.h` — safetensors index + loader
  - `acl_context.h` — RAII wrapper around ACL device/context/stream
- `src/` — implementations of the above.
- `tests/` — unit tests, regression tests, and benchmarks. See "Diagnostics"
  in the [project README](../README.md).
- `scripts/`
  - `set_env.sh` — exports `ASCEND_CUSTOM_OPP_PATH` and `LD_LIBRARY_PATH`
    for the toolkit and installed custom ops; source before running any
    binary.
  - `build.sh` — wrapper around the cmake build.
- `CMakeLists.txt` — engine library + per-test/per-bench targets.

## Build and run

```bash
cmake -S . -B build
cmake --build build -j$(nproc)

source scripts/set_env.sh
./build/bench_decode 8 30        # decode tps
./build/test_incremental_decode  # regression
```

The engine depends on the installed custom ops package; see
[`../custom_ops/`](../custom_ops/) for how to build that.
