# MiniCPM-V 4.6 on Ascend 310B(香橙派 Orange Pi AIPro 20T)

[English](README.md) · [中文](README_ZH.md)

一个完全从零写的 C++/AscendC 推理引擎,把 [MiniCPM-V 4.6][minicpmv] 跑在 Orange Pi AIPro 20T 板载的 Ascend 310B NPU 上。
**文本和图像对话都完全跑在 NPU 上,Python 端只在 CPU 上做 tokenize 和图像预处理,推理热路径完全不依赖 `torch_npu`。**

通过三轮 cube unit / 自定义 kernel 工作,单 batch 解码从 **2.88 → 5.90 tokens/s**(~2×),
跑的是完整 24 层 hybrid 线性 + full attention 模型(hidden 1024,vocab 248094,fp16):

| 阶段 | Tokens/s | 单步耗时 (ms) | 节省 |
|---|---:|---:|---:|
| 原生 `aclnnMm` baseline | 2.88 | 350 | — |
| + 自定义 cube matmul(M=1) | 4.37 | 229 | 121 |
| + lm_head 切 16 块走 cube | 4.99 | 200 | 29 |
| + 向量化 causal-conv1d step kernel | **5.90** | **170** | 30 |

测试条件:prompt_T=8,decode 30 个 token。剩下的 ~170 ms / step 主要被
matmul 权重带宽吃掉;下一步只能上权重量化(见 [Roadmap](#roadmap))。

视觉塔(SigLIP-so400m → vit_merger → 投影到 LM hidden,总共 27 层 transformer)
也已经移植到 C++/aclnn,端到端对照 HF CPU 参考实现验证过:最终给 LM 用的
`image_features` 跟 HF 输出的 `max_abs_diff = 0.0098`(448×448 输入)。

> ⚠ 当前只支持单 batch greedy decode。图像目前每张只跑**一个** 448×448 切片
> (HF MiniCPM-V 的高分辨率多切片管线在 [Roadmap](#roadmap) 里)。不同的
> MiniCPM-V 变体需要改 `language_model.cpp` 和 `vision.cpp` 里的 config。

## 快速开始

从干净 clone 开始跑通整条链路,前提是 Orange Pi AIPro 20T 已经装好 CANN
(在 `/usr/local/Ascend/ascend-toolkit/latest` 下面):

```bash
# 0)一次性安装 processor + Gradio 所需的 Python 依赖
pip install transformers pillow gradio

# 1)编译并安装 AscendC 自定义算子(cube matmul、fused SiLU·mul 等)
#    输出:./custom_opp_install/
./scripts/install_custom_ops.sh

# 2)编译引擎(库 + 约 40 个 tests/benches/tools)。输出:./build/
./scripts/build.sh

# 3)下载模型权重(~5 GB)放到 repo 同级目录
git lfs install
git clone https://huggingface.co/openbmb/MiniCPM-V-4_6 MiniCPM-V-4.6

# 4)source env 后启动多模态 Web UI
source scripts/set_env.sh
python3 src/python/gradio_app.py
# → 浏览器打开 http://localhost:7860,拖一张图片加文字提问就行
```

冷启动耗时:步骤 1 ≈ 3 min,步骤 2 ≈ 1 min,步骤 4 加载权重到 NPU 大约 75 s
再开始服务。**之后任意 prompt 长度都是 GPU 级体验**,没有按 shape 重编译的卡顿。

如果你的 CANN 安装在别处,在执行步骤 1 之前 `export
MINICPMV_ASCEND_TOOLKIT_ROOT=/路径/aarch64-linux`,执行步骤 2 时给 cmake
传 `-DASCEND_TOOLKIT_ROOT=...`。

## 硬件 & 软件要求

- **板子**:Orange Pi AIPro 20T(或任何 Ascend 310B 的设备)
- **系统**:Ubuntu 22.04 aarch64
- **CANN toolkit**:8.3.RC2,安装在 `/usr/local/Ascend/ascend-toolkit/latest/`
  (可以用 `MINICPMV_ASCEND_TOOLKIT_ROOT` 覆盖)
- **CMake**:≥ 3.20
- **Python**:3.8+,需要 `transformers ≥ 4.46`(用 `MiniCPMV4_6Processor`)
  和 `pillow`。Web UI 需要 `gradio`。`torch` / `torch_npu` **不在推理路径上需要**——
  它们可能被 `transformers` 间接装上,但没有任何 NPU op 在 Python 里跑。

## 编译

如果跳过了快速开始想手动执行:

```bash
# 1. 编译并安装 AscendC 自定义算子(cube matmul、fused norms、向量化
#    SiLU·mul、conv1d step、gated-delta-rule step 等)
#    输出:./custom_opp_install/vendors/customize/
./scripts/install_custom_ops.sh

# 2. 编译引擎(库 + 约 40 个 binary),输出 ./build/
./scripts/build.sh
```

CANN 装在别的位置时:执行 `set_env.sh` 之前 `export
MINICPMV_ASCEND_TOOLKIT_ROOT=/路径/aarch64-linux`,cmake 时加上
`-DASCEND_TOOLKIT_ROOT=...`。

## 下载模型

```bash
# 用 git-lfs:
git clone https://huggingface.co/openbmb/MiniCPM-V-4_6 MiniCPM-V-4.6

# 或者用 huggingface-cli:
huggingface-cli download openbmb/MiniCPM-V-4_6 \
    --local-dir MiniCPM-V-4.6 --local-dir-use-symlinks False
```

各 binary 默认读 `./MiniCPM-V-4.6/model.safetensors`(相对 CWD)。
用 `MINICPMV_MODEL_PATH=/其它路径/MiniCPM-V-4.6` 可以覆盖。

## 运行

主入口是下面的 [Gradio Web UI](#web-ui-gradio)。命令行工具和 benchmark
在 [Diagnostics](#diagnostics)。

### Web UI(Gradio)

跑在 C++ 引擎上的流式多模态对话——文本 + 图片(单切片 448×448),
不需要 torch_npu:

```bash
# 一次性安装 Gradio(已经装过 transformers / pillow 的话只补这个)
pip install gradio pillow

source scripts/set_env.sh
python3 src/python/gradio_app.py            # 文本 + 视觉(默认)
# 或者:
python3 src/python/gradio_app.py --text-only  # 跳过 ~1.5 GB 的 vision tower
                                              # 加载,只跑文本会更省显存 + 更快启动
```

浏览器打开 <http://localhost:7860>。把图片拖进消息框 + 提问,C++ 引擎会
在 prefill 之前把整条 SigLIP 视觉塔在 NPU 上跑完。每条回复 token-by-token
流式输出,底部脚注显示 ttft 和 decode tokens/s。

启动时间:

| 模式 | 首次就绪耗时 |
|---|---|
| `--text-only` | ~50 s(LM 权重 → NPU)+ ~2 s warmup |
| 默认(带视觉) | ~75 s(LM + SigLIP → NPU)+ ~2 s warmup |

启动完之后,**任意新 prompt 长度都是 GPU 级体验**——没有按 shape 编译的 JIT
卡顿(具体怎么做到的看
[SigLIP vision tower in C++](#4-siglip-vision-tower-in-c-no-torch_npu)
的说明,核心是 Python 端绕开了之前主导 ttft 的 `torch_npu` per-shape 编译)。

> ⚠ **当前只支持单切片视觉。** HF MiniCPM-V 的处理器会把高分辨率图片切成
> 最多 9 个 sub-image 来保留细节。我们引擎现在只跑**一个** 448×448 切片
> (32×32 patch 网格 → 64 个图像 token),Python 端给 processor 传
> `max_slice_nums=1` 让 token 数对得上。高分辨率细节识别因此打了折扣;
> 多切片支持在 roadmap 上。

### CLI

```bash
source scripts/set_env.sh

# Decode 基准测试 — 最主要的 tps 数字
./build/bench_decode 8 30
# → prompt_T=8  prefill_ms=...  per_step_ms=...  tokens/s=...

# 从一段文字 prompt 跑 greedy 生成
python3 src/python/run_hybrid.py --prompt "你好,介绍一下你自己"

# 把引擎 logits 跟 PyTorch 参考对一下(回归 sanity check)
python3 src/python/compare_logits.py
```

完整的单元测试和 benchmark 列表在 [Diagnostics](#diagnostics)。

## 关键优化

### 1. M=1 时的自定义 cube matmul

原生 `aclnnMm` 在 M=1(vector-matrix 矩阵乘)的形状下没把 cube unit 跑满,
而 M=1 恰好是 decode 的主导 shape。自定义的 `aclnnMatmulCubeCustom` kernel
([`src/csrc/custom_ops/op_kernel/matmul_cube_custom.cpp`][cube-kernel])
直接用 AscendC `MatmulImpl`,在我们关心的所有 shape 上比 `aclnnMm` 快 2-7×:

| N(输出维) | Cube 相对 `aclnnMm` 提速 |
|---|---|
| 512  | 3.4× |
| 2048 | 7.3× |
| 3584 | 2.9× |
| 6144 | 2.4× |

`matmul_b_transposed` 在 `N ≤ 16384 && N % 128 == 0 && M == 1` 时走 cube 路径,
支持原有的 `[N, K]` 和预转置后的 `[K, N]` 两种权重布局。宽矩阵权重在 load 阶段
通过 `load_matmul_weight_transposed` 预转置一次,保证热路径都在 cube 上。

我们也试过纯 vector unit 的方案(`MatmulVecCustom`),**所有 shape 上都比
`aclnnMm` 慢 30-50%** —— 作为反面教材留在树里。cube unit 才是正解。

### 2. lm_head 分块

lm_head 是 `[1, 1024] × [1024, 248094]` 的矩阵乘 —— 太宽,cube 切片(超过 ~32k
就会错)装不下。引擎在加载阶段把 lm_head 权重预切成 16 块 cube 友好的
`[K=1024, N=16384]`(最后一块用零 padding);decode 时跑 16 个 cube matmul,
host 端做 argmax reduce。

lm_head 这步从 ~75 ms(`aclnnMm` 慢路径)降到 ~40 ms。

### 3. 向量化 causal-conv1d step kernel

通用的 `linear_causal_conv` kernel 完全是标量实现(每个元素都用 `GetValue`/
`SetValue`),**而且** 4 行输出全算了,但 decode 实际上只用最后一行。这一步
平均每个 step 浪费 ~32 ms。

`linear_causal_conv_step` 收一个预转置过的 `[K=4, C]` 权重,DMA 4 行输入 +
4 行权重进 UB 当连续 tile,做 fp16 向量 mul+add(4 个 mul + 3 个 add)直接
写一行 `[1, C]` 输出。省了 ~30 ms。

通过可选的 `conv1d_step_weight` 字段接进来,prefill 路径和手工权重测试还是用
原 kernel,保证正确性回归。

### 其它自定义 kernel

- `rms_norm1024` / `gated_rms_norm_z` — 融合的 RMSNorm 变体
- `silu_mul` — 融合 SwiGLU 激活
- `attention_step` — 单 token full-attention(`softmax(q·K)·V`,逐 head)
- `linear_gated_delta_rule`(+ `_step`)— gated-delta-rule 递推

### 4. C++ 版 SigLIP 视觉塔(不依赖 torch_npu)

整条视觉路径都跑在 NPU 上,跟 LM 共用同一个引擎 subprocess。Python 端只通过
HF 的 `AutoProcessor` 做 CPU 端的 tokenize + 图像预处理,**不 import `torch_npu`,
不 load HF 模型。**

管线实现在 `src/csrc/lib/vision.cpp`:

```
pixel_values [1, 3, 14, P·14]  (HF naflex patch-strip 布局)
   ↓  vision_patch_embed       (aclnnConvolution)
   ↓  vision_position_embed    (4900 项位置表 + host 端 bucketize)
   ↓  encoder_layer × 7        (SigLIP:pre-LN attn + GELU(tanh) MLP)
   ↓  vit_merger               (2×2 窗口注意力 + 4-patch 拼接 MLP)
   ↓  encoder_layer × 20
   ↓  layer_norm(post)
   ↓  merger_mlp               (2×2 空间拼接 + pre-LN + linear + GELU + linear)
   ↓  image_features [1, P/16, 1024]  ← 喂给 LM 的图像 token
```

448×448 图片(32×32 patch 网格)走完是 1024 → 256 → 64 个图像 token。
端到端对照 CPU torch 实现做了数值验证:

| 阶段 | 跟 HF 的 max abs diff |
|---|---:|
| patch_embed | 0.0001 |
| 7 层 encoder 堆叠 | 0.0273 |
| vit_merger | 0.0498 |
| post_layernorm | 0.0469 |
| **image_features** | **0.0098** |

移植过程中踩到两个 ACL 的坑:

- CANN 预编译的 `MatMulV2_FP16` 二进制在 `M >= 64 && K > 4096` 时会返回
  `kernel pointer null`(errno 361001)。`matmul_b_transposed` 现在对这种 shape
  会按 K 切成 4096 的小块累加,绕开这个限制。
- 在 input 和 output 是同一个 buffer 的情况下,直接用 `aclnnAdd` 会把
  MatMulV2 的 kernel cache 留在一个坏状态,后续 `aclnnMm` 调用就挂了。
  `linear_bias` 现在用专门的 in-place API `aclnnInplaceAdd` 做 bias 加法。

## Diagnostics

端到端:

| Binary | 用途 |
|---|---|
| `./build/bench_decode prompt_T decode_N` | tokens/s + 单步延迟 |
| `./build/bench_prefill` | 多 token prefill 延迟 |
| `./build/bench_decode_subops` | decode shape 下的 per-op micro-bench |
| `python3 src/python/run_hybrid.py` | 端到端 greedy 生成 |
| `python3 src/python/compare_logits.py` | 跟 PyTorch 参考对 logits 差异 |

逐 op:

| Binary | 用途 |
|---|---|
| `./build/bench_matmul_throughput M N K iters` | 按 shape 测 aclnnMm 延迟 |
| `./build/bench_matmul_vec` | cube vs vector matmul 对比 |
| `./build/bench_step_kernel` | gated-delta-rule step kernel |
| `./build/bench_npu_bandwidth` | NPU memcpy / add / H2D / D2H 带宽 |
| `./bench/bench_ddr_bandwidth` | host DRAM 带宽(见 `bench/README.md`) |

正确性:

| Binary | 用途 |
|---|---|
| `./build/test_natural_b_matmul` | `[K,N]` 自然 vs `[N,K]` view |
| `./build/test_linear_causal_conv_step` | 新 conv kernel vs 通用版 |
| `./build/test_matmul_cube` | cube matmul vs 参考 |
| `./build/test_prefill_from_embeddings` | prefill 回归 |
| `./build/test_autoregressive_decode` | 多步 decode 回归 |
| `./build/test_incremental_decode` | 单步 decode 回归 |
| `./build/test_full_language_model` | 24 层 forward 烟雾测试 |
| `./build/test_vision_ops` | conv2d / gelu / batch_matmul 单测 |
| `./build/test_vision_patch_embed` | 完整 vision tower vs HF(需要 `/tmp/dump_vision_ref.py` 先跑出参考) |

## 性能拆解

单步耗时(~170 ms),5.90 tps,decode T=1:

| 组件 | 成本 | 占比 |
|---|---:|---:|
| lm_head(16 个 cube 块) | ~40 ms | 24% |
| 24 × MLP cube matmul(gate+up+down) | ~36 ms | 21% |
| 18 × linear-attn matmul(qkv+z+ab+out) | ~36 ms | 21% |
| 18 × linear-attn gated-delta-rule step | ~20 ms | 12% |
| 48 × rms_norm 调用 | ~17 ms | 10% |
| 6 × full-attn matmul + attention_step | ~15 ms | 9% |
| conv1d_step + silu/mul/sigmoid/拷贝 | ~10 ms | 6% |

按实测 ~44 GB/s 的 NPU d2d 带宽算,理论下界是 **~32 ms**(每步 1408 MB 权重读)。
cube 路径离这个下界 ~5.3× —— 主要是 M=1 时 kernel 启动相关的固定开销,
user-mode 这层已经压不下去了。

## Roadmap

- **多切片视觉**,让高分辨率图片保留细节。HF MiniCPM-V 处理器会把图片切成
  最多 9 个全分辨率 sub-image;引擎现在只跑一个 448×448 切片(~64 个图像
  token)。需要把视觉塔循环跑 N 次,把每个切片的特征按位置拼回 prompt。
- **权重量化** 是 tps 提升上唯一剩下的真正杠杆。int8 把每个 matmul 的字节读减半
  → 估计能到 ~10 tps;int4 → 估计 ~14 tps。需要在 cube 路径上写 dequant-fused 的 matmul kernel。
- KV-cache 卸载到 host 内存,支持更长上下文(目前受 NPU 显存限制)。
- Beam search / top-p 采样。目前只支持 greedy。

## License

Apache License 2.0,见 [`LICENSE`](LICENSE)。

MiniCPM-V 模型权重由 [OpenBMB][openbmb] 按它们自己的许可发布;再分发时
请尊重上游条款。

## 致谢

- [OpenBMB / MiniCPM-V 团队][minicpmv] 的模型
- 华为 Ascend AscendC 示例代码,作为自定义算子构建系统的基础
  (`src/csrc/custom_ops/cmake/`、`framework/`、`scripts/`)

[minicpmv]: https://huggingface.co/openbmb/MiniCPM-V-4_6
[openbmb]: https://github.com/OpenBMB
[cube-kernel]: src/csrc/custom_ops/op_kernel/matmul_cube_custom.cpp
