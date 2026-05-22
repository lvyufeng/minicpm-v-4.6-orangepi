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
// scratch: [5120] fp32 (8 blocks * 5 * head_dim temp buffers)
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

}  // namespace minicpmv
