#include "minicpmv/acl_context.h"
#include "minicpmv/ops.h"
#include "minicpmv/tensor.h"
#include "minicpmv/weights.h"

#include <acl/acl.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

using namespace minicpmv;

static uint16_t f2h(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;
    if (exp <= 0) return static_cast<uint16_t>(sign);
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}

static float h2f(uint16_t h) {
    uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) { out = sign; }
        else {
            int32_t e = 1;
            while ((mant & 0x400u) == 0) { mant <<= 1; e--; }
            mant &= 0x3ffu;
            out = sign | (static_cast<uint32_t>(e + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7f800000u | (mant << 13);
    } else {
        out = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}

static void copy_col_block(const Tensor& src, int64_t col_offset, Tensor& dst, aclrtStream stream) {
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

static void copy_head_to_seq(const Tensor& src_heads, int64_t head, int64_t heads_per_token,
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

static void copy_seq_to_head_block(const Tensor& src_seq, Tensor& dst, int64_t col_offset,
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

static bool all_finite_nonzero(const Tensor& t, float* first) {
    std::vector<uint16_t> host(t.numel());
    t.copy_to_host(host.data(), host.size() * sizeof(uint16_t));
    bool nonzero = false;
    for (size_t i = 0; i < host.size(); ++i) {
        float v = h2f(host[i]);
        if (i == 0 && first) *first = v;
        if (!std::isfinite(v)) return false;
        if (std::fabs(v) > 0.0f) nonzero = true;
    }
    return nonzero;
}

int main() {
    AclContext ctx(0);
    WeightsIndex index(default_safetensors_path());

    constexpr int64_t T = 4;
    constexpr int64_t Hidden = 1024;
    constexpr int64_t NumQHeads = 8;
    constexpr int64_t NumKVHeads = 2;
    constexpr int64_t QPerKV = NumQHeads / NumKVHeads;
    constexpr int64_t HeadDim = 256;
    constexpr int64_t QMainDim = NumQHeads * HeadDim;
    constexpr int64_t KVDim = NumKVHeads * HeadDim;
    constexpr int64_t QProjOut = QMainDim * 2;
    constexpr int64_t Rot = 64;
    constexpr int64_t HalfRot = Rot / 2;

    Tensor embed = index.load_to_device_as("model.language_model.embed_tokens.weight", DType::Float16);
    Tensor input_norm_w = index.load_to_device_as("model.language_model.layers.3.input_layernorm.weight", DType::Float16);
    Tensor q_w = index.load_to_device_as("model.language_model.layers.3.self_attn.q_proj.weight", DType::Float16);
    Tensor k_w = index.load_to_device_as("model.language_model.layers.3.self_attn.k_proj.weight", DType::Float16);
    Tensor v_w = index.load_to_device_as("model.language_model.layers.3.self_attn.v_proj.weight", DType::Float16);
    Tensor o_w = index.load_to_device_as("model.language_model.layers.3.self_attn.o_proj.weight", DType::Float16);
    Tensor q_norm_w = index.load_to_device_as("model.language_model.layers.3.self_attn.q_norm.weight", DType::Float16);
    Tensor k_norm_w = index.load_to_device_as("model.language_model.layers.3.self_attn.k_norm.weight", DType::Float16);

    if (q_w.shape() != std::vector<int64_t>{QProjOut, Hidden} ||
        k_w.shape() != std::vector<int64_t>{KVDim, Hidden} ||
        v_w.shape() != std::vector<int64_t>{KVDim, Hidden} ||
        o_w.shape() != std::vector<int64_t>{Hidden, QMainDim}) {
        std::cerr << "unexpected layer 3 attention weight shape" << std::endl;
        return 1;
    }

    std::vector<int32_t> ids = {1, 2, 10, 100};
    Tensor hidden({T, Hidden}, DType::Float16); hidden.allocate();
    embedding_lookup(embed, ids, hidden, ctx.stream());

    Tensor normed({T, Hidden}, DType::Float16); normed.allocate();
    rms_norm(hidden, input_norm_w, normed, 1e-6, ctx.stream());

    Tensor q_full({T, QProjOut}, DType::Float16); q_full.allocate();
    Tensor k_full({T, KVDim}, DType::Float16); k_full.allocate();
    Tensor v_full({T, KVDim}, DType::Float16); v_full.allocate();
    matmul_b_transposed(normed, q_w, q_full, ctx.stream());
    matmul_b_transposed(normed, k_w, k_full, ctx.stream());
    matmul_b_transposed(normed, v_w, v_full, ctx.stream());

    Tensor q_heads({T * NumQHeads, HeadDim}, DType::Float16); q_heads.allocate();
    Tensor k_heads({T * NumKVHeads, HeadDim}, DType::Float16); k_heads.allocate();
    copy_col_block(q_full, 0, q_heads, ctx.stream());
    check_acl(aclrtMemcpyAsync(k_heads.data(), k_heads.size_bytes(), k_full.data(), k_full.size_bytes(),
                               ACL_MEMCPY_DEVICE_TO_DEVICE, ctx.stream()), "copy k_heads");
    check_acl(aclrtSynchronizeStream(ctx.stream()), "copy k_heads sync");

    Tensor q_normed({T * NumQHeads, HeadDim}, DType::Float16); q_normed.allocate();
    Tensor k_normed({T * NumKVHeads, HeadDim}, DType::Float16); k_normed.allocate();
    rms_norm(q_heads, q_norm_w, q_normed, 1e-6, ctx.stream());
    rms_norm(k_heads, k_norm_w, k_normed, 1e-6, ctx.stream());

    std::vector<uint16_t> cos_host(T * HalfRot), sin_host(T * HalfRot);
    constexpr float RopeTheta = 10000000.0f;
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t i = 0; i < HalfRot; ++i) {
            float inv = std::pow(RopeTheta, -2.0f * static_cast<float>(i) / static_cast<float>(Rot));
            float theta = static_cast<float>(t) * inv;
            cos_host[t * HalfRot + i] = f2h(std::cos(theta));
            sin_host[t * HalfRot + i] = f2h(std::sin(theta));
        }
    }
    Tensor cos_t({T, HalfRot}, DType::Float16); cos_t.copy_from_host(cos_host.data(), cos_host.size() * sizeof(uint16_t));
    Tensor sin_t({T, HalfRot}, DType::Float16); sin_t.copy_from_host(sin_host.data(), sin_host.size() * sizeof(uint16_t));
    std::vector<int32_t> q_row_to_t(T * NumQHeads), k_row_to_t(T * NumKVHeads);
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t h = 0; h < NumQHeads; ++h) q_row_to_t[t * NumQHeads + h] = static_cast<int32_t>(t);
        for (int64_t h = 0; h < NumKVHeads; ++h) k_row_to_t[t * NumKVHeads + h] = static_cast<int32_t>(t);
    }
    Tensor q_rope({T * NumQHeads, HeadDim}, DType::Float16); q_rope.allocate();
    Tensor k_rope({T * NumKVHeads, HeadDim}, DType::Float16); k_rope.allocate();
    apply_rope_partial(q_normed, cos_t, sin_t, q_row_to_t, Rot, q_rope, ctx.stream());
    apply_rope_partial(k_normed, cos_t, sin_t, k_row_to_t, Rot, k_rope, ctx.stream());

    Tensor scale({T, T}, DType::Float16);
    std::vector<uint16_t> scale_host(T * T, f2h(1.0f / std::sqrt(static_cast<float>(HeadDim))));
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
        copy_head_to_seq(q_rope, qh, NumQHeads, q_seq, ctx.stream());
        copy_head_to_seq(k_rope, kvh, NumKVHeads, k_seq, ctx.stream());
        copy_col_block(v_full, kvh * HeadDim, v_seq, ctx.stream());
        matmul_b_transposed(q_seq, k_seq, scores, ctx.stream());
        mul(scores, scale, scaled_scores, ctx.stream());
        softmax_last_dim(scaled_scores, probs, ctx.stream());
        matmul(probs, v_seq, ctx_seq, ctx.stream());
        copy_seq_to_head_block(ctx_seq, attn_out, qh * HeadDim, ctx.stream());
    }

    Tensor out({T, Hidden}, DType::Float16); out.allocate();
    matmul_b_transposed(attn_out, o_w, out, ctx.stream());

    float a0 = 0.0f, o0 = 0.0f;
    if (!all_finite_nonzero(attn_out, &a0) || !all_finite_nonzero(out, &o0)) {
        std::cerr << "layer3 full attention smoke produced invalid output" << std::endl;
        return 2;
    }

    std::cout << "layer3 full attention smoke ok"
              << " attn_shape=[" << attn_out.shape()[0] << "," << attn_out.shape()[1] << "]"
              << " out_shape=[" << out.shape()[0] << "," << out.shape()[1] << "]"
              << " first=" << a0 << "," << o0 << std::endl;
    return 0;
}
