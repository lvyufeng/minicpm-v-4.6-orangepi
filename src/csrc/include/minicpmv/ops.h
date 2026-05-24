#pragma once

#include "minicpmv/tensor.h"

#include <acl/acl.h>

#include <cstdint>
#include <vector>

namespace minicpmv {

void embedding_lookup(const Tensor& weight,
                      const std::vector<int32_t>& host_ids,
                      Tensor& out,
                      aclrtStream stream);

void matmul(const Tensor& a, const Tensor& b, Tensor& out, aclrtStream stream);

void matmul_b_transposed(const Tensor& a, const Tensor& b, Tensor& out, aclrtStream stream);

void argmax_last_dim(const Tensor& self, Tensor& out, aclrtStream stream);

// Incremental flash attention for single-token decode.
// query: fp16 [num_q_heads, head_dim]  (the new token's post-RoPE Q, packed)
// k_cache, v_cache: fp16 [max_seq, num_kv_heads * head_dim]  (rows [0, context) valid)
// out: fp16 [1, num_q_heads * head_dim]
// Calls aclnnIncreFlashAttention with BSND layout views.
void incre_flash_attention(const Tensor& query,
                           const Tensor& k_cache,
                           const Tensor& v_cache,
                           int64_t context,
                           int64_t num_q_heads,
                           int64_t num_kv_heads,
                           int64_t head_dim,
                           float scale,
                           Tensor& out,
                           aclrtStream stream);

// Fused SwiGLU activation: out = silu(gate) * up.
// gate, up, out: fp16 same shape.
void silu_mul(const Tensor& gate, const Tensor& up, Tensor& out, aclrtStream stream);

void add(const Tensor& a, const Tensor& b, Tensor& out, aclrtStream stream);

void mul(const Tensor& a, const Tensor& b, Tensor& out, aclrtStream stream);

void silu(const Tensor& self, Tensor& out, aclrtStream stream);

void sigmoid(const Tensor& self, Tensor& out, aclrtStream stream);

void softmax_last_dim(const Tensor& self, Tensor& out, aclrtStream stream);

void rms_norm1024(const Tensor& x, const Tensor& gamma, Tensor& out,
                  double epsilon, aclrtStream stream);

void rms_norm(const Tensor& x, const Tensor& gamma, Tensor& out,
              double epsilon, aclrtStream stream);

void layer_norm(const Tensor& x, const Tensor& weight, const Tensor& bias,
                Tensor& out, double epsilon, aclrtStream stream);

void cast(const Tensor& self, Tensor& out, aclrtStream stream);

// Partial RoPE: apply rotary embedding to the first `rot` dims of head_dim,
// pass through the remaining dims unchanged.
// x: [N, D], cos/sin: [T, rot/2] expanded by row-index `row_to_t`. Both fp16.
// `row_to_t[n]` selects which token row in (cos, sin) row n belongs to.
void apply_rope_partial(const Tensor& x,
                        const Tensor& cos_table,
                        const Tensor& sin_table,
                        const std::vector<int32_t>& row_to_t,
                        int64_t rot,
                        Tensor& out,
                        aclrtStream stream);

// Causal depthwise conv1d for linear-attention. Input x: [T, C] fp16, weight: [C, K] fp16
// (or [C, 1, K] view collapsed to [C, K]), output out: [T, C] fp16.
// Computes y[t, c] = sum_{k=0..K-1} x[t - (K-1) + k, c] * weight[c, k]
// with left-padding (entries before t=0 treated as zero). Currently K must be 4.
void linear_causal_conv(const Tensor& x,
                        const Tensor& weight,
                        Tensor& out,
                        aclrtStream stream);

// Single-step variant of linear_causal_conv for K=4 decode. Input x: [4, C] fp16
// (3 cached history rows + 1 new row). weight_t: [4, C] fp16, transposed from
// the canonical [C, 4] layout so each tap's weights are contiguous. Output:
// [1, C] fp16 corresponding to the last row of the full conv. Vectorized with
// UB DMA + fp16 vector mul/add — ~5-10x faster than the generic kernel at T=1.
void linear_causal_conv_step(const Tensor& x,
                             const Tensor& weight_t,
                             Tensor& out,
                             aclrtStream stream);

// Linear-attention recurrent gated delta rule on NPU.
// mixed:  [T, 6144] fp16 (post conv1d+SiLU), layout q|k|v contiguous over heads.
// beta:   [T, 16]   fp16 (precomputed sigmoid(b))
// decay:  [T, 16]   fp16 (precomputed exp(-exp(A_log) * softplus(a + dt_bias)))
// out:    [T, 2048] fp16 (core_attn_out)
// Currently fixed at NumHeads=16, HeadDim=128.
void linear_gated_delta_rule(const Tensor& mixed,
                             const Tensor& beta,
                             const Tensor& decay,
                             Tensor& scratch,
                             Tensor& out,
                             aclrtStream stream);

// Single-step gated delta rule with persistent recurrent state on device.
// mixed: [1, 6144] fp16, beta/decay: [1, 16] fp16
// state: [16, 128, 128] fp32 (read+written in place)
// scratch: [6144] fp32 (8 blocks * 6 * head_dim temp buffers)
// out: [1, 2048] fp16
void linear_gated_delta_rule_step(const Tensor& mixed,
                                  const Tensor& beta,
                                  const Tensor& decay,
                                  Tensor& state,
                                  Tensor& scratch,
                                  Tensor& out,
                                  aclrtStream stream);

// Per-head gated RMSNorm with z gate for linear-attention.
// core:   [T, 2048] fp16  (recurrence output)
// z_silu: [T, 2048] fp16  (precomputed z * sigmoid(z))
// gamma:  [128]     fp16  (per-head RMSNorm gain, broadcast across heads)
// out:    [T, 2048] fp16  (= gamma * core * rstd_per_head * z_silu)
// Currently fixed at NumHeads=16, HeadDim=128.
void gated_rms_norm_z(const Tensor& core,
                      const Tensor& z_silu,
                      const Tensor& gamma,
                      Tensor& out,
                      aclrtStream stream);

// 2D convolution. input: [N, Cin, H, W] fp16, weight: [Cout, Cin/groups, kH, kW] fp16,
// optional bias: [Cout] fp16. stride/padding are size-2 vectors (H, W).
// dilation defaults to {1,1} and groups defaults to 1 — set explicitly for
// other patterns. out: [N, Cout, H', W'] where H', W' are derived from input.
void conv2d(const Tensor& input,
            const Tensor& weight,
            const Tensor* bias,
            const std::vector<int64_t>& stride,
            const std::vector<int64_t>& padding,
            Tensor& out,
            aclrtStream stream);

// GELU activation. `tanh` selects between the exact (false) and
// tanh-approximation (true) variants used by `gelu_pytorch_tanh`.
// Same shape in / out. fp16.
void gelu(const Tensor& self, bool tanh_approx, Tensor& out, aclrtStream stream);

// Batched matrix multiply: out[b, m, n] = sum_k a[b, m, k] * b[b, k, n].
// Supports any batch rank >= 1; the last two dims are the matmul.
void batch_matmul(const Tensor& a, const Tensor& b, Tensor& out, aclrtStream stream);

// Linear with bias: out = x @ W^T + bias.
//   x:    [M, K] fp16
//   w:    [N, K] fp16 (or [K, N] natural — matches matmul_b_transposed semantics)
//   bias: [N]   fp16 (broadcast across rows). Pass nullptr to skip the add.
//   out:  [M, N] fp16
// Convenience wrapper around matmul_b_transposed + optional broadcast add.
void linear_bias(const Tensor& x,
                 const Tensor& w,
                 const Tensor* bias,
                 Tensor& out,
                 aclrtStream stream);

// Permute (transpose) self's axes by `dims`. out's shape must be
// `[self.shape()[dims[0]], self.shape()[dims[1]], ...]`.
void permute(const Tensor& self,
             const std::vector<int64_t>& dims,
             Tensor& out,
             aclrtStream stream);

// Element-wise multiply by an fp32 scalar (cast to dtype internally).
void muls(const Tensor& self, float scalar, Tensor& out, aclrtStream stream);

// Mean over `dim` axes. If keep_dim is true, reduced axes are kept as size 1.
// Output dtype is inferred from input (kept the same).
void mean(const Tensor& self,
          const std::vector<int64_t>& dims,
          bool keep_dim,
          Tensor& out,
          aclrtStream stream);

}  // namespace minicpmv
