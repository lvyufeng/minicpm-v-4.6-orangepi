#include "minicpmv/decoder_layer.h"

#include "minicpmv/acl_context.h"
#include "minicpmv/ops.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
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

    Tensor q_heads({T * NumQHeads, HeadDim}, DType::Float16); q_heads.allocate();
    Tensor k_heads({T * NumKVHeads, HeadDim}, DType::Float16); k_heads.allocate();
    copy_col_block(q_full, 0, q_heads, stream);
    check_acl(aclrtMemcpyAsync(k_heads.data(), k_heads.size_bytes(), k_full.data(), k_full.size_bytes(),
                               ACL_MEMCPY_DEVICE_TO_DEVICE, stream), "copy k_heads");
    check_acl(aclrtSynchronizeStream(stream), "copy k_heads sync");

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

    Tensor attn_out({T, QMainDim}, DType::Float16); attn_out.allocate();
    Tensor q_seq({T, HeadDim}, DType::Float16); q_seq.allocate();
    Tensor k_seq({T, HeadDim}, DType::Float16); k_seq.allocate();
    Tensor v_seq({T, HeadDim}, DType::Float16); v_seq.allocate();
    Tensor scores({T, T}, DType::Float16); scores.allocate();
    Tensor scaled_scores({T, T}, DType::Float16); scaled_scores.allocate();
    Tensor probs({T, T}, DType::Float16); probs.allocate();
    Tensor ctx_seq({T, HeadDim}, DType::Float16); ctx_seq.allocate();

    for (int64_t qh = 0; qh < NumQHeads; ++qh) {
        const int64_t kvh = qh / QPerKV;
        copy_head_to_seq(q_rope, qh, NumQHeads, q_seq, stream);
        copy_head_to_seq(k_rope, kvh, NumKVHeads, k_seq, stream);
        copy_col_block(v_full, kvh * HeadDim, v_seq, stream);
        matmul_b_transposed(q_seq, k_seq, scores, stream);
        mul(scores, scale, scaled_scores, stream);
        softmax_last_dim(scaled_scores, probs, stream);
        matmul(probs, v_seq, ctx_seq, stream);
        copy_seq_to_head_block(ctx_seq, attn_out, qh * HeadDim, stream);
    }

    Tensor attn_proj({T, Hidden}, DType::Float16); attn_proj.allocate();
    matmul_b_transposed(attn_out, *weights.o_proj_weight, attn_proj, stream);

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
                                         const LinearAttentionDecoderLayerWeights& weights,
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

}  // namespace minicpmv
