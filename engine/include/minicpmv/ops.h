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

}  // namespace minicpmv
