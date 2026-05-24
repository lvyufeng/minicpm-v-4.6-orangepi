#include "minicpmv/acl_context.h"
#include "minicpmv/ops.h"
#include "minicpmv/tensor.h"
#include "minicpmv/weights.h"

#include <acl/acl.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
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

static void copy_first_cols(const Tensor& src, int64_t src_cols,
                            Tensor& dst, int64_t dst_cols,
                            aclrtStream stream) {
    const size_t elem_size = dtype_size(src.dtype());
    const int64_t rows = src.shape()[0];
    const size_t src_row_bytes = static_cast<size_t>(src_cols) * elem_size;
    const size_t dst_row_bytes = static_cast<size_t>(dst_cols) * elem_size;
    auto* src_base = static_cast<const uint8_t*>(src.data());
    auto* dst_base = static_cast<uint8_t*>(dst.data());
    for (int64_t r = 0; r < rows; ++r) {
        check_acl(aclrtMemcpyAsync(dst_base + static_cast<size_t>(r) * dst_row_bytes,
                                   dst_row_bytes,
                                   src_base + static_cast<size_t>(r) * src_row_bytes,
                                   dst_row_bytes,
                                   ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                  "copy_first_cols");
    }
    check_acl(aclrtSynchronizeStream(stream), "copy_first_cols sync");
}

static bool all_finite_sample(const Tensor& t, float* first_value) {
    std::vector<uint16_t> host(t.numel());
    t.copy_to_host(host.data(), host.size() * sizeof(uint16_t));
    bool nonzero = false;
    float first = 0.0f;
    for (size_t i = 0; i < host.size(); ++i) {
        float v = h2f(host[i]);
        if (i == 0) first = v;
        if (!std::isfinite(v)) return false;
        if (std::fabs(v) > 0.0f) nonzero = true;
    }
    if (first_value) *first_value = first;
    return nonzero;
}

int main() {
    AclContext ctx(0);
    WeightsIndex index(default_safetensors_path());

    constexpr int64_t N = 4;
    constexpr int64_t Hidden = 1024;
    constexpr int64_t NumQHeads = 8;
    constexpr int64_t NumKVHeads = 2;
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
    Tensor q_norm_w = index.load_to_device_as("model.language_model.layers.3.self_attn.q_norm.weight", DType::Float16);
    Tensor k_norm_w = index.load_to_device_as("model.language_model.layers.3.self_attn.k_norm.weight", DType::Float16);

    if (embed.shape() != std::vector<int64_t>{248094, Hidden} ||
        input_norm_w.shape() != std::vector<int64_t>{Hidden} ||
        q_w.shape() != std::vector<int64_t>{QProjOut, Hidden} ||
        k_w.shape() != std::vector<int64_t>{KVDim, Hidden} ||
        v_w.shape() != std::vector<int64_t>{KVDim, Hidden} ||
        q_norm_w.shape() != std::vector<int64_t>{HeadDim} ||
        k_norm_w.shape() != std::vector<int64_t>{HeadDim}) {
        std::cerr << "unexpected layer 3 weight shape" << std::endl;
        return 1;
    }

    std::vector<int32_t> ids = {1, 2, 10, 100};
    Tensor hidden({N, Hidden}, DType::Float16);
    hidden.allocate();
    embedding_lookup(embed, ids, hidden, ctx.stream());

    Tensor normed({N, Hidden}, DType::Float16);
    normed.allocate();
    rms_norm(hidden, input_norm_w, normed, 1e-6, ctx.stream());

    Tensor q_full({N, QProjOut}, DType::Float16);
    Tensor k_full({N, KVDim}, DType::Float16);
    Tensor v_full({N, KVDim}, DType::Float16);
    q_full.allocate();
    k_full.allocate();
    v_full.allocate();
    matmul_b_transposed(normed, q_w, q_full, ctx.stream());
    matmul_b_transposed(normed, k_w, k_full, ctx.stream());
    matmul_b_transposed(normed, v_w, v_full, ctx.stream());

    Tensor q_heads({N * NumQHeads, HeadDim}, DType::Float16);
    Tensor k_heads({N * NumKVHeads, HeadDim}, DType::Float16);
    q_heads.allocate();
    k_heads.allocate();
    copy_first_cols(q_full, QProjOut, q_heads, QMainDim, ctx.stream());
    check_acl(aclrtMemcpyAsync(k_heads.data(), k_heads.size_bytes(),
                               k_full.data(), k_full.size_bytes(),
                               ACL_MEMCPY_DEVICE_TO_DEVICE, ctx.stream()),
              "copy k heads");
    check_acl(aclrtSynchronizeStream(ctx.stream()), "copy k heads sync");

    Tensor q_normed({N * NumQHeads, HeadDim}, DType::Float16);
    Tensor k_normed({N * NumKVHeads, HeadDim}, DType::Float16);
    q_normed.allocate();
    k_normed.allocate();
    rms_norm(q_heads, q_norm_w, q_normed, 1e-6, ctx.stream());
    rms_norm(k_heads, k_norm_w, k_normed, 1e-6, ctx.stream());

    std::vector<uint16_t> cos_host(N * HalfRot);
    std::vector<uint16_t> sin_host(N * HalfRot);
    constexpr float RopeTheta = 10000000.0f;
    for (int64_t t = 0; t < N; ++t) {
        for (int64_t i = 0; i < HalfRot; ++i) {
            float inv = std::pow(RopeTheta, -2.0f * static_cast<float>(i) / static_cast<float>(Rot));
            float theta = static_cast<float>(t) * inv;
            cos_host[t * HalfRot + i] = f2h(std::cos(theta));
            sin_host[t * HalfRot + i] = f2h(std::sin(theta));
        }
    }
    Tensor cos_t({N, HalfRot}, DType::Float16);
    Tensor sin_t({N, HalfRot}, DType::Float16);
    cos_t.copy_from_host(cos_host.data(), cos_host.size() * sizeof(uint16_t));
    sin_t.copy_from_host(sin_host.data(), sin_host.size() * sizeof(uint16_t));

    std::vector<int32_t> q_row_to_t(N * NumQHeads);
    std::vector<int32_t> k_row_to_t(N * NumKVHeads);
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t h = 0; h < NumQHeads; ++h) q_row_to_t[n * NumQHeads + h] = static_cast<int32_t>(n);
        for (int64_t h = 0; h < NumKVHeads; ++h) k_row_to_t[n * NumKVHeads + h] = static_cast<int32_t>(n);
    }

    Tensor q_rope({N * NumQHeads, HeadDim}, DType::Float16);
    Tensor k_rope({N * NumKVHeads, HeadDim}, DType::Float16);
    q_rope.allocate();
    k_rope.allocate();
    apply_rope_partial(q_normed, cos_t, sin_t, q_row_to_t, Rot, q_rope, ctx.stream());
    apply_rope_partial(k_normed, cos_t, sin_t, k_row_to_t, Rot, k_rope, ctx.stream());

    float q0 = 0.0f, k0 = 0.0f, v0 = 0.0f;
    if (!all_finite_sample(q_rope, &q0) || !all_finite_sample(k_rope, &k0) || !all_finite_sample(v_full, &v0)) {
        std::cerr << "layer3 full attention front smoke produced non-finite or all-zero output" << std::endl;
        return 2;
    }

    std::cout << "layer3 full attention front smoke ok"
              << " q_shape=[" << q_rope.shape()[0] << "," << q_rope.shape()[1] << "]"
              << " k_shape=[" << k_rope.shape()[0] << "," << k_rope.shape()[1] << "]"
              << " v_shape=[" << v_full.shape()[0] << "," << v_full.shape()[1] << "]"
              << " first=" << q0 << "," << k0 << "," << v0 << std::endl;
    return 0;
}
