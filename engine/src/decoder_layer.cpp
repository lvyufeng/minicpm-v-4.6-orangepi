#include "minicpmv/decoder_layer.h"

#include "minicpmv/acl_context.h"
#include "minicpmv/ops.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace minicpmv {
namespace {

uint16_t f32_to_f16_bits(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;
    if (exp <= 0) return static_cast<uint16_t>(sign);
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}

void check_ptr(const Tensor* t, const char* name) {
    if (t == nullptr) {
        throw std::runtime_error(std::string("missing decoder layer weight: ") + name);
    }
}

void copy_col_block(const Tensor& src, int64_t col_offset, Tensor& dst, aclrtStream stream) {
    const int64_t rows = src.shape()[0];
    const int64_t src_cols = src.shape()[1];
    const int64_t dst_cols = dst.shape()[1];
    const size_t elem = dtype_size(src.dtype());
    const size_t src_row_bytes = static_cast<size_t>(src_cols) * elem;
    const size_t dst_row_bytes = static_cast<size_t>(dst_cols) * elem;
    const size_t block_bytes = static_cast<size_t>(dst_cols) * elem;
    auto* s = static_cast<const uint8_t*>(src.data());
    auto* d = static_cast<uint8_t*>(dst.data());
    for (int64_t r = 0; r < rows; ++r) {
        check_acl(aclrtMemcpyAsync(d + static_cast<size_t>(r) * dst_row_bytes, block_bytes,
                                   s + static_cast<size_t>(r) * src_row_bytes + static_cast<size_t>(col_offset) * elem,
                                   block_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                  "copy_col_block");
    }
    check_acl(aclrtSynchronizeStream(stream), "copy_col_block sync");
}

void copy_head_to_seq(const Tensor& src_heads, int64_t head, int64_t heads_per_token,
                      Tensor& dst_seq, aclrtStream stream) {
    const int64_t tokens = dst_seq.shape()[0];
    const int64_t dim = dst_seq.shape()[1];
    const size_t row_bytes = static_cast<size_t>(dim) * dtype_size(src_heads.dtype());
    auto* s = static_cast<const uint8_t*>(src_heads.data());
    auto* d = static_cast<uint8_t*>(dst_seq.data());
    for (int64_t t = 0; t < tokens; ++t) {
        const int64_t src_row = t * heads_per_token + head;
        check_acl(aclrtMemcpyAsync(d + static_cast<size_t>(t) * row_bytes, row_bytes,
                                   s + static_cast<size_t>(src_row) * row_bytes, row_bytes,
                                   ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                  "copy_head_to_seq");
    }
    check_acl(aclrtSynchronizeStream(stream), "copy_head_to_seq sync");
}

void copy_seq_to_head_block(const Tensor& src_seq, Tensor& dst, int64_t col_offset,
                            aclrtStream stream) {
    const int64_t rows = src_seq.shape()[0];
    const int64_t dst_cols = dst.shape()[1];
    const int64_t src_cols = src_seq.shape()[1];
    const size_t elem = dtype_size(src_seq.dtype());
    const size_t src_row_bytes = static_cast<size_t>(src_cols) * elem;
    const size_t dst_row_bytes = static_cast<size_t>(dst_cols) * elem;
    auto* s = static_cast<const uint8_t*>(src_seq.data());
    auto* d = static_cast<uint8_t*>(dst.data());
    for (int64_t r = 0; r < rows; ++r) {
        check_acl(aclrtMemcpyAsync(d + static_cast<size_t>(r) * dst_row_bytes + static_cast<size_t>(col_offset) * elem,
                                   src_row_bytes,
                                   s + static_cast<size_t>(r) * src_row_bytes,
                                   src_row_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                  "copy_seq_to_head_block");
    }
    check_acl(aclrtSynchronizeStream(stream), "copy_seq_to_head_block sync");
}

void copy_heads_from_cols(const Tensor& src, int64_t heads, int64_t head_dim,
                          Tensor& dst, aclrtStream stream) {
    const int64_t rows = src.shape()[0];
    const int64_t src_cols = src.shape()[1];
    const size_t elem = dtype_size(src.dtype());
    const size_t src_row_bytes = static_cast<size_t>(src_cols) * elem;
    const size_t head_bytes = static_cast<size_t>(head_dim) * elem;
    auto* s = static_cast<const uint8_t*>(src.data());
    auto* d = static_cast<uint8_t*>(dst.data());
    for (int64_t t = 0; t < rows; ++t) {
        for (int64_t h = 0; h < heads; ++h) {
            check_acl(aclrtMemcpyAsync(d + static_cast<size_t>(t * heads + h) * head_bytes, head_bytes,
                                       s + static_cast<size_t>(t) * src_row_bytes + static_cast<size_t>(h * head_dim) * elem,
                                       head_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                      "copy_heads_from_cols");
        }
    }
    check_acl(aclrtSynchronizeStream(stream), "copy_heads_from_cols sync");
}

void split_q_gate(const Tensor& src, int64_t heads, int64_t head_dim,
                  Tensor& q_out, Tensor& gate_out, aclrtStream stream) {
    const int64_t rows = src.shape()[0];
    const size_t elem = dtype_size(src.dtype());
    const size_t src_row_bytes = static_cast<size_t>(src.shape()[1]) * elem;
    const size_t head_bytes = static_cast<size_t>(head_dim) * elem;
    const size_t q_row_bytes = static_cast<size_t>(q_out.shape()[1]) * elem;
    auto* s = static_cast<const uint8_t*>(src.data());
    auto* dq = static_cast<uint8_t*>(q_out.data());
    auto* dg = static_cast<uint8_t*>(gate_out.data());
    for (int64_t t = 0; t < rows; ++t) {
        for (int64_t h = 0; h < heads; ++h) {
            const size_t src_off = static_cast<size_t>(t) * src_row_bytes
                                 + static_cast<size_t>(h * 2 * head_dim) * elem;
            const size_t dst_off = static_cast<size_t>(t) * q_row_bytes
                                 + static_cast<size_t>(h * head_dim) * elem;
            check_acl(aclrtMemcpyAsync(dq + dst_off, head_bytes, s + src_off, head_bytes,
                                       ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                      "split_q_gate q");
            check_acl(aclrtMemcpyAsync(dg + dst_off, head_bytes, s + src_off + head_bytes, head_bytes,
                                       ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                      "split_q_gate gate");
        }
    }
    check_acl(aclrtSynchronizeStream(stream), "split_q_gate sync");
}


void pack_heads_to_row(const Tensor& src_heads, Tensor& dst_row, int64_t heads,
                       int64_t head_dim, aclrtStream stream) {
    const size_t elem = dtype_size(src_heads.dtype());
    const size_t head_bytes = static_cast<size_t>(head_dim) * elem;
    auto* s = static_cast<const uint8_t*>(src_heads.data());
    auto* d = static_cast<uint8_t*>(dst_row.data());
    for (int64_t h = 0; h < heads; ++h) {
        check_acl(aclrtMemcpyAsync(d + static_cast<size_t>(h) * head_bytes, head_bytes,
                                   s + static_cast<size_t>(h) * head_bytes, head_bytes,
                                   ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                  "pack_heads_to_row");
    }
    check_acl(aclrtSynchronizeStream(stream), "pack_heads_to_row sync");
}

void copy_tensor_to_cache_row(const Tensor& src, Tensor& cache, int64_t row, aclrtStream stream) {
    const size_t row_bytes = static_cast<size_t>(cache.shape()[1]) * dtype_size(cache.dtype());
    auto* d = static_cast<uint8_t*>(cache.data()) + static_cast<size_t>(row) * row_bytes;
    check_acl(aclrtMemcpyAsync(d, row_bytes, src.data(), row_bytes,
                               ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
              "copy_tensor_to_cache_row");
    check_acl(aclrtSynchronizeStream(stream), "copy_tensor_to_cache_row sync");
}

void copy_cache_head_to_seq(const Tensor& cache, int64_t head, int64_t heads_per_token,
                            int64_t head_dim, int64_t rows, Tensor& dst_seq,
                            aclrtStream stream) {
    const size_t elem = dtype_size(cache.dtype());
    const size_t cache_row_bytes = static_cast<size_t>(cache.shape()[1]) * elem;
    const size_t head_bytes = static_cast<size_t>(head_dim) * elem;
    auto* s = static_cast<const uint8_t*>(cache.data());
    auto* d = static_cast<uint8_t*>(dst_seq.data());
    for (int64_t r = 0; r < rows; ++r) {
        check_acl(aclrtMemcpyAsync(d + static_cast<size_t>(r) * head_bytes, head_bytes,
                                   s + static_cast<size_t>(r) * cache_row_bytes + static_cast<size_t>(head * head_dim) * elem,
                                   head_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                  "copy_cache_head_to_seq");
    }
    (void)heads_per_token;
    check_acl(aclrtSynchronizeStream(stream), "copy_cache_head_to_seq sync");
}

void copy_matrix_rows(const Tensor& src, int64_t src_row, Tensor& dst, int64_t dst_row,
                      int64_t rows, aclrtStream stream) {
    const size_t row_bytes = static_cast<size_t>(src.shape()[1]) * dtype_size(src.dtype());
    auto* s = static_cast<const uint8_t*>(src.data());
    auto* d = static_cast<uint8_t*>(dst.data());
    for (int64_t r = 0; r < rows; ++r) {
        check_acl(aclrtMemcpyAsync(d + static_cast<size_t>(dst_row + r) * row_bytes, row_bytes,
                                   s + static_cast<size_t>(src_row + r) * row_bytes, row_bytes,
                                   ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                  "copy_matrix_rows");
    }
    check_acl(aclrtSynchronizeStream(stream), "copy_matrix_rows sync");
}


void validate_shapes(const Tensor& hidden,
                     const FullAttentionDecoderLayerWeights& w,
                     const FullAttentionDecoderLayerConfig& c,
                     const Tensor& out) {
    check_ptr(w.input_norm_weight, "input_norm_weight");
    check_ptr(w.post_attention_norm_weight, "post_attention_norm_weight");
    check_ptr(w.q_proj_weight, "q_proj_weight");
    check_ptr(w.k_proj_weight, "k_proj_weight");
    check_ptr(w.v_proj_weight, "v_proj_weight");
    check_ptr(w.o_proj_weight, "o_proj_weight");
    check_ptr(w.q_norm_weight, "q_norm_weight");
    check_ptr(w.k_norm_weight, "k_norm_weight");
    check_ptr(w.gate_proj_weight, "gate_proj_weight");
    check_ptr(w.up_proj_weight, "up_proj_weight");
    check_ptr(w.down_proj_weight, "down_proj_weight");

    if (hidden.shape().size() != 2 || out.shape() != hidden.shape()) {
        throw std::runtime_error("decoder layer hidden/out must be [T, H] and same shape");
    }
    if (hidden.dtype() != DType::Float16 || out.dtype() != DType::Float16) {
        throw std::runtime_error("decoder layer requires fp16 hidden/out");
    }
    if (c.num_q_heads <= 0 || c.num_kv_heads <= 0 || c.head_dim <= 0 || c.rotary_dim <= 0) {
        throw std::runtime_error("decoder layer invalid config dims");
    }
    if (c.num_q_heads % c.num_kv_heads != 0) {
        throw std::runtime_error("decoder layer num_q_heads must be divisible by num_kv_heads");
    }
}

}  // namespace

void full_attention_decoder_layer(const Tensor& hidden,
                                  const FullAttentionDecoderLayerWeights& weights,
                                  const Tensor& cos_table,
                                  const Tensor& sin_table,
                                  const std::vector<int32_t>& row_to_t,
                                  const FullAttentionDecoderLayerConfig& config,
                                  Tensor& out,
                                  aclrtStream stream) {
    validate_shapes(hidden, weights, config, out);

    const int64_t T = hidden.shape()[0];
    const int64_t Hidden = hidden.shape()[1];
    const int64_t NumQHeads = config.num_q_heads;
    const int64_t NumKVHeads = config.num_kv_heads;
    const int64_t QPerKV = NumQHeads / NumKVHeads;
    const int64_t HeadDim = config.head_dim;
    const int64_t QMainDim = NumQHeads * HeadDim;
    const int64_t KVDim = NumKVHeads * HeadDim;
    const int64_t QProjOut = QMainDim * 2;
    const int64_t Intermediate = weights.gate_proj_weight->shape()[0];

    if (weights.input_norm_weight->shape() != std::vector<int64_t>{Hidden} ||
        weights.post_attention_norm_weight->shape() != std::vector<int64_t>{Hidden} ||
        weights.q_proj_weight->shape() != std::vector<int64_t>{QProjOut, Hidden} ||
        weights.k_proj_weight->shape() != std::vector<int64_t>{KVDim, Hidden} ||
        weights.v_proj_weight->shape() != std::vector<int64_t>{KVDim, Hidden} ||
        weights.o_proj_weight->shape() != std::vector<int64_t>{Hidden, QMainDim} ||
        weights.q_norm_weight->shape() != std::vector<int64_t>{HeadDim} ||
        weights.k_norm_weight->shape() != std::vector<int64_t>{HeadDim} ||
        weights.gate_proj_weight->shape() != std::vector<int64_t>{Intermediate, Hidden} ||
        weights.up_proj_weight->shape() != std::vector<int64_t>{Intermediate, Hidden} ||
        weights.down_proj_weight->shape() != std::vector<int64_t>{Hidden, Intermediate}) {
        throw std::runtime_error("decoder layer weight shape mismatch");
    }
    if (static_cast<int64_t>(row_to_t.size()) != T) {
        throw std::runtime_error("decoder layer row_to_t size must match sequence length");
    }

    Tensor normed({T, Hidden}, DType::Float16); normed.allocate();
    rms_norm(hidden, *weights.input_norm_weight, normed, config.rms_epsilon, stream);

    Tensor q_full({T, QProjOut}, DType::Float16); q_full.allocate();
    Tensor k_full({T, KVDim}, DType::Float16); k_full.allocate();
    Tensor v_full({T, KVDim}, DType::Float16); v_full.allocate();
    matmul_b_transposed(normed, *weights.q_proj_weight, q_full, stream);
    matmul_b_transposed(normed, *weights.k_proj_weight, k_full, stream);
    matmul_b_transposed(normed, *weights.v_proj_weight, v_full, stream);

    Tensor q_only({T, QMainDim}, DType::Float16); q_only.allocate();
    Tensor q_gate({T, QMainDim}, DType::Float16); q_gate.allocate();
    split_q_gate(q_full, NumQHeads, HeadDim, q_only, q_gate, stream);

    Tensor q_heads({T * NumQHeads, HeadDim}, DType::Float16); q_heads.allocate();
    Tensor k_heads({T * NumKVHeads, HeadDim}, DType::Float16); k_heads.allocate();
    copy_heads_from_cols(q_only, NumQHeads, HeadDim, q_heads, stream);
    copy_heads_from_cols(k_full, NumKVHeads, HeadDim, k_heads, stream);

    Tensor q_normed({T * NumQHeads, HeadDim}, DType::Float16); q_normed.allocate();
    Tensor k_normed({T * NumKVHeads, HeadDim}, DType::Float16); k_normed.allocate();
    rms_norm(q_heads, *weights.q_norm_weight, q_normed, config.rms_epsilon, stream);
    rms_norm(k_heads, *weights.k_norm_weight, k_normed, config.rms_epsilon, stream);

    std::vector<int32_t> q_row_to_t(T * NumQHeads);
    std::vector<int32_t> k_row_to_t(T * NumKVHeads);
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t h = 0; h < NumQHeads; ++h) q_row_to_t[t * NumQHeads + h] = row_to_t[t];
        for (int64_t h = 0; h < NumKVHeads; ++h) k_row_to_t[t * NumKVHeads + h] = row_to_t[t];
    }
    Tensor q_rope({T * NumQHeads, HeadDim}, DType::Float16); q_rope.allocate();
    Tensor k_rope({T * NumKVHeads, HeadDim}, DType::Float16); k_rope.allocate();
    apply_rope_partial(q_normed, cos_table, sin_table, q_row_to_t, config.rotary_dim, q_rope, stream);
    apply_rope_partial(k_normed, cos_table, sin_table, k_row_to_t, config.rotary_dim, k_rope, stream);

    Tensor scale({T, T}, DType::Float16);
    std::vector<uint16_t> scale_host(static_cast<size_t>(T * T), f32_to_f16_bits(1.0f / std::sqrt(static_cast<float>(HeadDim))));
    scale.copy_from_host(scale_host.data(), scale_host.size() * sizeof(uint16_t));

    std::vector<uint16_t> mask_host(static_cast<size_t>(T * T));
    for (int64_t r = 0; r < T; ++r) {
        for (int64_t c = 0; c < T; ++c) {
            mask_host[r * T + c] = f32_to_f16_bits(row_to_t[c] <= row_to_t[r] ? 0.0f : -65504.0f);
        }
    }
    Tensor causal_mask({T, T}, DType::Float16);
    causal_mask.copy_from_host(mask_host.data(), mask_host.size() * sizeof(uint16_t));

    Tensor attn_out({T, QMainDim}, DType::Float16); attn_out.allocate();
    Tensor q_seq({T, HeadDim}, DType::Float16); q_seq.allocate();
    Tensor k_seq({T, HeadDim}, DType::Float16); k_seq.allocate();
    Tensor v_seq({T, HeadDim}, DType::Float16); v_seq.allocate();
    Tensor scores({T, T}, DType::Float16); scores.allocate();
    Tensor scaled_scores({T, T}, DType::Float16); scaled_scores.allocate();
    Tensor masked_scores({T, T}, DType::Float16); masked_scores.allocate();
    Tensor probs({T, T}, DType::Float16); probs.allocate();
    Tensor ctx_seq({T, HeadDim}, DType::Float16); ctx_seq.allocate();

    for (int64_t qh = 0; qh < NumQHeads; ++qh) {
        const int64_t kvh = qh / QPerKV;
        copy_head_to_seq(q_rope, qh, NumQHeads, q_seq, stream);
        copy_head_to_seq(k_rope, kvh, NumKVHeads, k_seq, stream);
        copy_col_block(v_full, kvh * HeadDim, v_seq, stream);
        matmul_b_transposed(q_seq, k_seq, scores, stream);
        mul(scores, scale, scaled_scores, stream);
        add(scaled_scores, causal_mask, masked_scores, stream);
        softmax_last_dim(masked_scores, probs, stream);
        matmul(probs, v_seq, ctx_seq, stream);
        copy_seq_to_head_block(ctx_seq, attn_out, qh * HeadDim, stream);
    }

    Tensor attn_proj({T, Hidden}, DType::Float16); attn_proj.allocate();
    {
        Tensor gate_sig({T, QMainDim}, DType::Float16); gate_sig.allocate();
        sigmoid(q_gate, gate_sig, stream);
        Tensor attn_gated({T, QMainDim}, DType::Float16); attn_gated.allocate();
        mul(attn_out, gate_sig, attn_gated, stream);
        matmul_b_transposed(attn_gated, *weights.o_proj_weight, attn_proj, stream);
    }

    Tensor after_attn({T, Hidden}, DType::Float16); after_attn.allocate();
    add(hidden, attn_proj, after_attn, stream);

    Tensor mlp_in({T, Hidden}, DType::Float16); mlp_in.allocate();
    rms_norm(after_attn, *weights.post_attention_norm_weight, mlp_in, config.rms_epsilon, stream);

    Tensor gate({T, Intermediate}, DType::Float16); gate.allocate();
    Tensor up({T, Intermediate}, DType::Float16); up.allocate();
    Tensor gate_act({T, Intermediate}, DType::Float16); gate_act.allocate();
    Tensor gated({T, Intermediate}, DType::Float16); gated.allocate();
    Tensor mlp_out({T, Hidden}, DType::Float16); mlp_out.allocate();

    matmul_b_transposed(mlp_in, *weights.gate_proj_weight, gate, stream);
    matmul_b_transposed(mlp_in, *weights.up_proj_weight, up, stream);
    silu(gate, gate_act, stream);
    mul(gate_act, up, gated, stream);
    matmul_b_transposed(gated, *weights.down_proj_weight, mlp_out, stream);
    add(after_attn, mlp_out, out, stream);
}

void linear_attention_decoder_layer_stub(const Tensor& hidden,
                                         const LinearAttentionDecoderLayerStubWeights& weights,
                                         const LinearAttentionDecoderLayerConfig& config,
                                         Tensor& out,
                                         aclrtStream stream) {
    check_ptr(weights.input_norm_weight, "linear input_norm_weight");
    check_ptr(weights.post_attention_norm_weight, "linear post_attention_norm_weight");
    check_ptr(weights.gate_proj_weight, "linear gate_proj_weight");
    check_ptr(weights.up_proj_weight, "linear up_proj_weight");
    check_ptr(weights.down_proj_weight, "linear down_proj_weight");

    if (hidden.shape().size() != 2 || out.shape() != hidden.shape()) {
        throw std::runtime_error("linear decoder layer hidden/out must be [T, H] and same shape");
    }
    if (hidden.dtype() != DType::Float16 || out.dtype() != DType::Float16) {
        throw std::runtime_error("linear decoder layer requires fp16 hidden/out");
    }

    const int64_t T = hidden.shape()[0];
    const int64_t Hidden = hidden.shape()[1];
    const int64_t Intermediate = weights.gate_proj_weight->shape()[0];
    if (weights.input_norm_weight->shape() != std::vector<int64_t>{Hidden} ||
        weights.post_attention_norm_weight->shape() != std::vector<int64_t>{Hidden} ||
        weights.gate_proj_weight->shape() != std::vector<int64_t>{Intermediate, Hidden} ||
        weights.up_proj_weight->shape() != std::vector<int64_t>{Intermediate, Hidden} ||
        weights.down_proj_weight->shape() != std::vector<int64_t>{Hidden, Intermediate}) {
        throw std::runtime_error("linear decoder layer weight shape mismatch");
    }

    Tensor normed({T, Hidden}, DType::Float16); normed.allocate();
    rms_norm(hidden, *weights.input_norm_weight, normed, config.rms_epsilon, stream);

    Tensor after_attn({T, Hidden}, DType::Float16); after_attn.allocate();
    add(hidden, normed, after_attn, stream);

    Tensor mlp_in({T, Hidden}, DType::Float16); mlp_in.allocate();
    rms_norm(after_attn, *weights.post_attention_norm_weight, mlp_in, config.rms_epsilon, stream);

    Tensor gate({T, Intermediate}, DType::Float16); gate.allocate();
    Tensor up({T, Intermediate}, DType::Float16); up.allocate();
    Tensor gate_act({T, Intermediate}, DType::Float16); gate_act.allocate();
    Tensor gated({T, Intermediate}, DType::Float16); gated.allocate();
    Tensor mlp_out({T, Hidden}, DType::Float16); mlp_out.allocate();

    matmul_b_transposed(mlp_in, *weights.gate_proj_weight, gate, stream);
    matmul_b_transposed(mlp_in, *weights.up_proj_weight, up, stream);
    silu(gate, gate_act, stream);
    mul(gate_act, up, gated, stream);
    matmul_b_transposed(gated, *weights.down_proj_weight, mlp_out, stream);
    add(after_attn, mlp_out, out, stream);
}

namespace {

float h16_to_f32(uint16_t h) {
    uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            int32_t e = 1;
            while ((mant & 0x400u) == 0) { mant <<= 1; e--; }
            mant &= 0x3ffu;
            bits = sign | (static_cast<uint32_t>(e + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

float softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return std::exp(x);
    return std::log1p(std::exp(x));
}

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

}  // namespace

void linear_attention_decoder_layer(const Tensor& hidden,
                                    const LinearAttentionDecoderLayerWeights& weights,
                                    const LinearAttentionDecoderLayerConfig& config,
                                    Tensor& out,
                                    aclrtStream stream) {
    check_ptr(weights.input_norm_weight, "linear input_norm_weight");
    check_ptr(weights.post_attention_norm_weight, "linear post_attention_norm_weight");
    check_ptr(weights.in_proj_qkv_weight, "linear in_proj_qkv_weight");
    check_ptr(weights.in_proj_z_weight, "linear in_proj_z_weight");
    check_ptr(weights.in_proj_a_weight, "linear in_proj_a_weight");
    check_ptr(weights.in_proj_b_weight, "linear in_proj_b_weight");
    check_ptr(weights.conv1d_weight, "linear conv1d_weight");
    check_ptr(weights.dt_bias, "linear dt_bias");
    check_ptr(weights.a_log, "linear a_log");
    check_ptr(weights.gated_norm_weight, "linear gated_norm_weight");
    check_ptr(weights.out_proj_weight, "linear out_proj_weight");
    check_ptr(weights.gate_proj_weight, "linear gate_proj_weight");
    check_ptr(weights.up_proj_weight, "linear up_proj_weight");
    check_ptr(weights.down_proj_weight, "linear down_proj_weight");

    if (hidden.shape().size() != 2 || hidden.shape()[1] != 1024) {
        throw std::runtime_error("linear decoder layer hidden must be [T, 1024]");
    }
    if (out.shape() != hidden.shape() || out.dtype() != DType::Float16 || hidden.dtype() != DType::Float16) {
        throw std::runtime_error("linear decoder layer hidden/out must match shape and be fp16");
    }

    const int64_t T = hidden.shape()[0];
    const int64_t Hidden = 1024;
    const int64_t NumHeads = 16;
    const int64_t HeadDim = 128;
    const int64_t KeyDim = NumHeads * HeadDim;
    const int64_t ValueDim = NumHeads * HeadDim;
    const int64_t ConvDim = 2 * KeyDim + ValueDim;
    const int64_t Intermediate = weights.gate_proj_weight->shape()[0];

    if (weights.in_proj_qkv_weight->shape() != std::vector<int64_t>{ConvDim, Hidden} ||
        weights.in_proj_z_weight->shape() != std::vector<int64_t>{ValueDim, Hidden} ||
        weights.in_proj_a_weight->shape() != std::vector<int64_t>{NumHeads, Hidden} ||
        weights.in_proj_b_weight->shape() != std::vector<int64_t>{NumHeads, Hidden} ||
        weights.conv1d_weight->shape() != std::vector<int64_t>{ConvDim, 1, 4} ||
        weights.dt_bias->shape() != std::vector<int64_t>{NumHeads} ||
        weights.a_log->shape() != std::vector<int64_t>{NumHeads} ||
        weights.gated_norm_weight->shape() != std::vector<int64_t>{HeadDim} ||
        weights.out_proj_weight->shape() != std::vector<int64_t>{Hidden, ValueDim} ||
        weights.input_norm_weight->shape() != std::vector<int64_t>{Hidden} ||
        weights.post_attention_norm_weight->shape() != std::vector<int64_t>{Hidden} ||
        weights.gate_proj_weight->shape() != std::vector<int64_t>{Intermediate, Hidden} ||
        weights.up_proj_weight->shape() != std::vector<int64_t>{Intermediate, Hidden} ||
        weights.down_proj_weight->shape() != std::vector<int64_t>{Hidden, Intermediate}) {
        throw std::runtime_error("linear decoder layer weight shape mismatch");
    }

    Tensor normed({T, Hidden}, DType::Float16); normed.allocate();
    rms_norm(hidden, *weights.input_norm_weight, normed, config.rms_epsilon, stream);

    Tensor qkv({T, ConvDim}, DType::Float16); qkv.allocate();
    Tensor z({T, ValueDim}, DType::Float16); z.allocate();
    Tensor a({T, NumHeads}, DType::Float16); a.allocate();
    Tensor b({T, NumHeads}, DType::Float16); b.allocate();
    matmul_b_transposed(normed, *weights.in_proj_qkv_weight, qkv, stream);
    matmul_b_transposed(normed, *weights.in_proj_z_weight, z, stream);
    matmul_b_transposed(normed, *weights.in_proj_a_weight, a, stream);
    matmul_b_transposed(normed, *weights.in_proj_b_weight, b, stream);

    Tensor conv({T, ConvDim}, DType::Float16); conv.allocate();
    Tensor mixed({T, ConvDim}, DType::Float16); mixed.allocate();
    linear_causal_conv(qkv, *weights.conv1d_weight, conv, stream);
    silu(conv, mixed, stream);

    std::vector<uint16_t> a_host(static_cast<size_t>(T) * NumHeads);
    std::vector<uint16_t> b_host(static_cast<size_t>(T) * NumHeads);
    std::vector<uint16_t> dt_host(NumHeads);
    std::vector<uint16_t> a_log_host(NumHeads);
    a.copy_to_host(a_host.data(), a_host.size() * sizeof(uint16_t));
    b.copy_to_host(b_host.data(), b_host.size() * sizeof(uint16_t));
    weights.dt_bias->copy_to_host(dt_host.data(), dt_host.size() * sizeof(uint16_t));
    weights.a_log->copy_to_host(a_log_host.data(), a_log_host.size() * sizeof(uint16_t));

    std::vector<uint16_t> beta_h(static_cast<size_t>(T) * NumHeads);
    std::vector<uint16_t> decay_h(static_cast<size_t>(T) * NumHeads);
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t h = 0; h < NumHeads; ++h) {
            float bv = h16_to_f32(b_host[t * NumHeads + h]);
            float av = h16_to_f32(a_host[t * NumHeads + h]);
            float dtv = h16_to_f32(dt_host[h]);
            float alv = h16_to_f32(a_log_host[h]);
            float g = -std::exp(alv) * softplus(av + dtv);
            beta_h[t * NumHeads + h] = f32_to_f16_bits(sigmoid(bv));
            decay_h[t * NumHeads + h] = f32_to_f16_bits(std::exp(g));
        }
    }

    Tensor beta_dev({T, NumHeads}, DType::Float16);
    Tensor decay_dev({T, NumHeads}, DType::Float16);
    beta_dev.copy_from_host(beta_h.data(), beta_h.size() * sizeof(uint16_t));
    decay_dev.copy_from_host(decay_h.data(), decay_h.size() * sizeof(uint16_t));

    Tensor core_dev({T, ValueDim}, DType::Float16); core_dev.allocate();
    Tensor scratch({136192}, DType::Float32); scratch.allocate();
    linear_gated_delta_rule(mixed, beta_dev, decay_dev, scratch, core_dev, stream);

    Tensor z_silu({T, ValueDim}, DType::Float16); z_silu.allocate();
    silu(z, z_silu, stream);

    Tensor gated({T, ValueDim}, DType::Float16); gated.allocate();
    gated_rms_norm_z(core_dev, z_silu, *weights.gated_norm_weight, gated, stream);

    Tensor attn_proj({T, Hidden}, DType::Float16); attn_proj.allocate();
    matmul_b_transposed(gated, *weights.out_proj_weight, attn_proj, stream);

    Tensor after_attn({T, Hidden}, DType::Float16); after_attn.allocate();
    add(hidden, attn_proj, after_attn, stream);

    Tensor mlp_in({T, Hidden}, DType::Float16); mlp_in.allocate();
    rms_norm(after_attn, *weights.post_attention_norm_weight, mlp_in, config.rms_epsilon, stream);

    Tensor gate({T, Intermediate}, DType::Float16); gate.allocate();
    Tensor up({T, Intermediate}, DType::Float16); up.allocate();
    Tensor gate_act({T, Intermediate}, DType::Float16); gate_act.allocate();
    Tensor gated_mlp({T, Intermediate}, DType::Float16); gated_mlp.allocate();
    Tensor mlp_out({T, Hidden}, DType::Float16); mlp_out.allocate();

    matmul_b_transposed(mlp_in, *weights.gate_proj_weight, gate, stream);
    matmul_b_transposed(mlp_in, *weights.up_proj_weight, up, stream);
    silu(gate, gate_act, stream);
    mul(gate_act, up, gated_mlp, stream);
    matmul_b_transposed(gated_mlp, *weights.down_proj_weight, mlp_out, stream);
    add(after_attn, mlp_out, out, stream);
}

DecodeState make_decode_state(int64_t max_seq_len,
                              const std::vector<std::string>& layer_types,
                              const FullAttentionDecoderLayerConfig& full_config,
                              aclrtStream stream) {
    if (max_seq_len <= 0) {
        throw std::runtime_error("decode state max_seq_len must be positive");
    }
    DecodeState state;
    state.max_seq_len = max_seq_len;
    state.seq_len = 0;
    const int64_t kv_dim = full_config.num_kv_heads * full_config.head_dim;
    for (const auto& type : layer_types) {
        if (type == "full_attention") {
            FullAttentionLayerCache cache;
            cache.k_cache = Tensor({max_seq_len, kv_dim}, DType::Float16);
            cache.v_cache = Tensor({max_seq_len, kv_dim}, DType::Float16);
            cache.k_cache.allocate();
            cache.v_cache.allocate();
            check_acl(aclrtMemsetAsync(cache.k_cache.data(), cache.k_cache.size_bytes(), 0,
                                       cache.k_cache.size_bytes(), stream), "memset full k cache");
            check_acl(aclrtMemsetAsync(cache.v_cache.data(), cache.v_cache.size_bytes(), 0,
                                       cache.v_cache.size_bytes(), stream), "memset full v cache");
            state.full.push_back(std::move(cache));
        } else if (type == "linear_attention") {
            LinearAttentionLayerCache cache;
            cache.conv_buf = Tensor({3, 6144}, DType::Float16);
            cache.recurrent_state = Tensor({16, 128, 128}, DType::Float32);
            cache.conv_buf.allocate();
            cache.recurrent_state.allocate();
            check_acl(aclrtMemsetAsync(cache.conv_buf.data(), cache.conv_buf.size_bytes(), 0,
                                       cache.conv_buf.size_bytes(), stream), "memset linear conv cache");
            check_acl(aclrtMemsetAsync(cache.recurrent_state.data(), cache.recurrent_state.size_bytes(), 0,
                                       cache.recurrent_state.size_bytes(), stream), "memset linear recurrent cache");
            state.linear.push_back(std::move(cache));
        } else {
            throw std::runtime_error("unknown layer type for decode state: " + type);
        }
    }
    check_acl(aclrtSynchronizeStream(stream), "make_decode_state sync");
    return state;
}

void full_attention_decoder_layer_step(const Tensor& hidden,
                                       const FullAttentionDecoderLayerWeights& weights,
                                       const Tensor& cos_table,
                                       const Tensor& sin_table,
                                       int32_t pos,
                                       int64_t cache_len,
                                       const FullAttentionDecoderLayerConfig& config,
                                       FullAttentionLayerCache& cache,
                                       Tensor& out,
                                       aclrtStream stream) {
    validate_shapes(hidden, weights, config, out);
    if (hidden.shape()[0] != 1) {
        throw std::runtime_error("full_attention_decoder_layer_step hidden must be [1, H]");
    }
    if (cache_len < 0 || cache_len >= cache.k_cache.shape()[0]) {
        throw std::runtime_error("full_attention_decoder_layer_step cache_len out of range");
    }

    const int64_t Hidden = hidden.shape()[1];
    const int64_t NumQHeads = config.num_q_heads;
    const int64_t NumKVHeads = config.num_kv_heads;
    const int64_t QPerKV = NumQHeads / NumKVHeads;
    const int64_t HeadDim = config.head_dim;
    const int64_t QMainDim = NumQHeads * HeadDim;
    const int64_t KVDim = NumKVHeads * HeadDim;
    const int64_t QProjOut = QMainDim * 2;
    const int64_t Intermediate = weights.gate_proj_weight->shape()[0];
    const int64_t Context = cache_len + 1;

    if (cache.k_cache.shape() != std::vector<int64_t>{cache.k_cache.shape()[0], KVDim} ||
        cache.v_cache.shape() != cache.k_cache.shape()) {
        throw std::runtime_error("full_attention_decoder_layer_step cache shape mismatch");
    }

    Tensor normed({1, Hidden}, DType::Float16); normed.allocate();
    rms_norm(hidden, *weights.input_norm_weight, normed, config.rms_epsilon, stream);

    Tensor q_full({1, QProjOut}, DType::Float16); q_full.allocate();
    Tensor k_full({1, KVDim}, DType::Float16); k_full.allocate();
    Tensor v_full({1, KVDim}, DType::Float16); v_full.allocate();
    matmul_b_transposed(normed, *weights.q_proj_weight, q_full, stream);
    matmul_b_transposed(normed, *weights.k_proj_weight, k_full, stream);
    matmul_b_transposed(normed, *weights.v_proj_weight, v_full, stream);

    Tensor q_only({1, QMainDim}, DType::Float16); q_only.allocate();
    Tensor q_gate({1, QMainDim}, DType::Float16); q_gate.allocate();
    split_q_gate(q_full, NumQHeads, HeadDim, q_only, q_gate, stream);

    Tensor q_heads({NumQHeads, HeadDim}, DType::Float16); q_heads.allocate();
    Tensor k_heads({NumKVHeads, HeadDim}, DType::Float16); k_heads.allocate();
    copy_heads_from_cols(q_only, NumQHeads, HeadDim, q_heads, stream);
    copy_heads_from_cols(k_full, NumKVHeads, HeadDim, k_heads, stream);

    Tensor q_normed({NumQHeads, HeadDim}, DType::Float16); q_normed.allocate();
    Tensor k_normed({NumKVHeads, HeadDim}, DType::Float16); k_normed.allocate();
    rms_norm(q_heads, *weights.q_norm_weight, q_normed, config.rms_epsilon, stream);
    rms_norm(k_heads, *weights.k_norm_weight, k_normed, config.rms_epsilon, stream);

    std::vector<int32_t> q_row_to_t(NumQHeads, pos);
    std::vector<int32_t> k_row_to_t(NumKVHeads, pos);
    Tensor q_rope({NumQHeads, HeadDim}, DType::Float16); q_rope.allocate();
    Tensor k_rope({NumKVHeads, HeadDim}, DType::Float16); k_rope.allocate();
    apply_rope_partial(q_normed, cos_table, sin_table, q_row_to_t, config.rotary_dim, q_rope, stream);
    apply_rope_partial(k_normed, cos_table, sin_table, k_row_to_t, config.rotary_dim, k_rope, stream);

    Tensor k_row({1, KVDim}, DType::Float16); k_row.allocate();
    pack_heads_to_row(k_rope, k_row, NumKVHeads, HeadDim, stream);
    copy_tensor_to_cache_row(k_row, cache.k_cache, cache_len, stream);
    copy_tensor_to_cache_row(v_full, cache.v_cache, cache_len, stream);

    Tensor scale({1, Context}, DType::Float16);
    std::vector<uint16_t> scale_host(static_cast<size_t>(Context), f32_to_f16_bits(1.0f / std::sqrt(static_cast<float>(HeadDim))));
    scale.copy_from_host(scale_host.data(), scale_host.size() * sizeof(uint16_t));

    Tensor attn_out({1, QMainDim}, DType::Float16); attn_out.allocate();
    Tensor q_seq({1, HeadDim}, DType::Float16); q_seq.allocate();
    Tensor k_seq({Context, HeadDim}, DType::Float16); k_seq.allocate();
    Tensor v_seq({Context, HeadDim}, DType::Float16); v_seq.allocate();
    Tensor scores({1, Context}, DType::Float16); scores.allocate();
    Tensor scaled_scores({1, Context}, DType::Float16); scaled_scores.allocate();
    Tensor probs({1, Context}, DType::Float16); probs.allocate();
    Tensor ctx_seq({1, HeadDim}, DType::Float16); ctx_seq.allocate();

    for (int64_t qh = 0; qh < NumQHeads; ++qh) {
        const int64_t kvh = qh / QPerKV;
        copy_head_to_seq(q_rope, qh, NumQHeads, q_seq, stream);
        copy_cache_head_to_seq(cache.k_cache, kvh, NumKVHeads, HeadDim, Context, k_seq, stream);
        copy_cache_head_to_seq(cache.v_cache, kvh, NumKVHeads, HeadDim, Context, v_seq, stream);
        matmul_b_transposed(q_seq, k_seq, scores, stream);
        mul(scores, scale, scaled_scores, stream);
        softmax_last_dim(scaled_scores, probs, stream);
        matmul(probs, v_seq, ctx_seq, stream);
        copy_seq_to_head_block(ctx_seq, attn_out, qh * HeadDim, stream);
    }

    Tensor attn_proj({1, Hidden}, DType::Float16); attn_proj.allocate();
    {
        Tensor gate_sig({1, QMainDim}, DType::Float16); gate_sig.allocate();
        sigmoid(q_gate, gate_sig, stream);
        Tensor attn_gated({1, QMainDim}, DType::Float16); attn_gated.allocate();
        mul(attn_out, gate_sig, attn_gated, stream);
        matmul_b_transposed(attn_gated, *weights.o_proj_weight, attn_proj, stream);
    }

    Tensor after_attn({1, Hidden}, DType::Float16); after_attn.allocate();
    add(hidden, attn_proj, after_attn, stream);

    Tensor mlp_in({1, Hidden}, DType::Float16); mlp_in.allocate();
    rms_norm(after_attn, *weights.post_attention_norm_weight, mlp_in, config.rms_epsilon, stream);

    Tensor gate({1, Intermediate}, DType::Float16); gate.allocate();
    Tensor up({1, Intermediate}, DType::Float16); up.allocate();
    Tensor gate_act({1, Intermediate}, DType::Float16); gate_act.allocate();
    Tensor gated({1, Intermediate}, DType::Float16); gated.allocate();
    Tensor mlp_out({1, Hidden}, DType::Float16); mlp_out.allocate();

    matmul_b_transposed(mlp_in, *weights.gate_proj_weight, gate, stream);
    matmul_b_transposed(mlp_in, *weights.up_proj_weight, up, stream);
    silu(gate, gate_act, stream);
    mul(gate_act, up, gated, stream);
    matmul_b_transposed(gated, *weights.down_proj_weight, mlp_out, stream);
    add(after_attn, mlp_out, out, stream);
}

void linear_attention_decoder_layer_step(const Tensor& hidden,
                                         const LinearAttentionDecoderLayerWeights& weights,
                                         const LinearAttentionDecoderLayerConfig& config,
                                         LinearAttentionLayerCache& cache,
                                         Tensor& out,
                                         aclrtStream stream) {
    check_ptr(weights.input_norm_weight, "linear input_norm_weight");
    check_ptr(weights.post_attention_norm_weight, "linear post_attention_norm_weight");
    check_ptr(weights.in_proj_qkv_weight, "linear in_proj_qkv_weight");
    check_ptr(weights.in_proj_z_weight, "linear in_proj_z_weight");
    check_ptr(weights.in_proj_a_weight, "linear in_proj_a_weight");
    check_ptr(weights.in_proj_b_weight, "linear in_proj_b_weight");
    check_ptr(weights.conv1d_weight, "linear conv1d_weight");
    check_ptr(weights.dt_bias, "linear dt_bias");
    check_ptr(weights.a_log, "linear a_log");
    check_ptr(weights.gated_norm_weight, "linear gated_norm_weight");
    check_ptr(weights.out_proj_weight, "linear out_proj_weight");
    check_ptr(weights.gate_proj_weight, "linear gate_proj_weight");
    check_ptr(weights.up_proj_weight, "linear up_proj_weight");
    check_ptr(weights.down_proj_weight, "linear down_proj_weight");

    if (hidden.shape() != std::vector<int64_t>{1, 1024} || out.shape() != hidden.shape() ||
        hidden.dtype() != DType::Float16 || out.dtype() != DType::Float16) {
        throw std::runtime_error("linear_attention_decoder_layer_step hidden/out must be [1,1024] fp16");
    }
    if (cache.conv_buf.shape() != std::vector<int64_t>{3, 6144} || cache.recurrent_state.shape() != std::vector<int64_t>{16, 128, 128}) {
        throw std::runtime_error("linear_attention_decoder_layer_step cache shape mismatch");
    }

    const int64_t Hidden = 1024;
    const int64_t NumHeads = 16;
    const int64_t HeadDim = 128;
    const int64_t KeyDim = NumHeads * HeadDim;
    const int64_t ValueDim = NumHeads * HeadDim;
    const int64_t ConvDim = 2 * KeyDim + ValueDim;
    const int64_t Intermediate = weights.gate_proj_weight->shape()[0];

    Tensor normed({1, Hidden}, DType::Float16); normed.allocate();
    rms_norm(hidden, *weights.input_norm_weight, normed, config.rms_epsilon, stream);

    Tensor qkv({1, ConvDim}, DType::Float16); qkv.allocate();
    Tensor z({1, ValueDim}, DType::Float16); z.allocate();
    Tensor a({1, NumHeads}, DType::Float16); a.allocate();
    Tensor b({1, NumHeads}, DType::Float16); b.allocate();
    matmul_b_transposed(normed, *weights.in_proj_qkv_weight, qkv, stream);
    matmul_b_transposed(normed, *weights.in_proj_z_weight, z, stream);
    matmul_b_transposed(normed, *weights.in_proj_a_weight, a, stream);
    matmul_b_transposed(normed, *weights.in_proj_b_weight, b, stream);

    Tensor conv_input({4, ConvDim}, DType::Float16); conv_input.allocate();
    copy_matrix_rows(cache.conv_buf, 0, conv_input, 0, 3, stream);
    copy_matrix_rows(qkv, 0, conv_input, 3, 1, stream);
    Tensor conv_all({4, ConvDim}, DType::Float16); conv_all.allocate();
    linear_causal_conv(conv_input, *weights.conv1d_weight, conv_all, stream);
    Tensor conv_last({1, ConvDim}, DType::Float16); conv_last.allocate();
    copy_matrix_rows(conv_all, 3, conv_last, 0, 1, stream);
    copy_matrix_rows(conv_input, 1, cache.conv_buf, 0, 3, stream);

    Tensor mixed({1, ConvDim}, DType::Float16); mixed.allocate();
    silu(conv_last, mixed, stream);

    std::vector<uint16_t> a_host(NumHeads);
    std::vector<uint16_t> b_host(NumHeads);
    std::vector<uint16_t> dt_host(NumHeads);
    std::vector<uint16_t> a_log_host(NumHeads);
    a.copy_to_host(a_host.data(), a_host.size() * sizeof(uint16_t));
    b.copy_to_host(b_host.data(), b_host.size() * sizeof(uint16_t));
    weights.dt_bias->copy_to_host(dt_host.data(), dt_host.size() * sizeof(uint16_t));
    weights.a_log->copy_to_host(a_log_host.data(), a_log_host.size() * sizeof(uint16_t));

    std::vector<uint16_t> beta_h(NumHeads);
    std::vector<uint16_t> decay_h(NumHeads);
    for (int64_t h = 0; h < NumHeads; ++h) {
        float bv = h16_to_f32(b_host[h]);
        float av = h16_to_f32(a_host[h]);
        float dtv = h16_to_f32(dt_host[h]);
        float alv = h16_to_f32(a_log_host[h]);
        float g = -std::exp(alv) * softplus(av + dtv);
        beta_h[h] = f32_to_f16_bits(sigmoid(bv));
        decay_h[h] = f32_to_f16_bits(std::exp(g));
    }

    Tensor beta_dev({1, NumHeads}, DType::Float16);
    Tensor decay_dev({1, NumHeads}, DType::Float16);
    beta_dev.copy_from_host(beta_h.data(), beta_h.size() * sizeof(uint16_t));
    decay_dev.copy_from_host(decay_h.data(), decay_h.size() * sizeof(uint16_t));

    Tensor core_dev({1, ValueDim}, DType::Float16); core_dev.allocate();
    Tensor scratch({8 * 5 * 128}, DType::Float32); scratch.allocate();
    linear_gated_delta_rule_step(mixed, beta_dev, decay_dev, cache.recurrent_state, scratch, core_dev, stream);

    Tensor z_silu({1, ValueDim}, DType::Float16); z_silu.allocate();
    silu(z, z_silu, stream);

    Tensor gated({1, ValueDim}, DType::Float16); gated.allocate();
    gated_rms_norm_z(core_dev, z_silu, *weights.gated_norm_weight, gated, stream);

    Tensor attn_proj({1, Hidden}, DType::Float16); attn_proj.allocate();
    matmul_b_transposed(gated, *weights.out_proj_weight, attn_proj, stream);

    Tensor after_attn({1, Hidden}, DType::Float16); after_attn.allocate();
    add(hidden, attn_proj, after_attn, stream);

    Tensor mlp_in({1, Hidden}, DType::Float16); mlp_in.allocate();
    rms_norm(after_attn, *weights.post_attention_norm_weight, mlp_in, config.rms_epsilon, stream);

    Tensor gate({1, Intermediate}, DType::Float16); gate.allocate();
    Tensor up({1, Intermediate}, DType::Float16); up.allocate();
    Tensor gate_act({1, Intermediate}, DType::Float16); gate_act.allocate();
    Tensor gated_mlp({1, Intermediate}, DType::Float16); gated_mlp.allocate();
    Tensor mlp_out({1, Hidden}, DType::Float16); mlp_out.allocate();

    matmul_b_transposed(mlp_in, *weights.gate_proj_weight, gate, stream);
    matmul_b_transposed(mlp_in, *weights.up_proj_weight, up, stream);
    silu(gate, gate_act, stream);
    mul(gate_act, up, gated_mlp, stream);
    matmul_b_transposed(gated_mlp, *weights.down_proj_weight, mlp_out, stream);
    add(after_attn, mlp_out, out, stream);
}

}  // namespace minicpmv
