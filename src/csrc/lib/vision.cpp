#include "minicpmv/vision.h"
#include "minicpmv/acl_context.h"
#include "minicpmv/ops.h"

#include <acl/acl_rt.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace minicpmv {

VisionConfig default_minicpmv46_vision_config() {
    return VisionConfig{};
}

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

Tensor load_w(WeightsIndex& index, const std::string& name) {
    return index.load_to_device_as(name, DType::Float16);
}

Tensor load_layer_w(WeightsIndex& index, int layer, const std::string& suffix) {
    return index.load_to_device_as(
        "model.vision_tower.encoder.layers." + std::to_string(layer) + "." + suffix,
        DType::Float16);
}

}  // namespace

VisionWeights load_vision_weights(WeightsIndex& index, const VisionConfig& cfg) {
    VisionWeights w;
    w.patch_embedding_w = load_w(index, "model.vision_tower.embeddings.patch_embedding.weight");
    w.patch_embedding_b = load_w(index, "model.vision_tower.embeddings.patch_embedding.bias");
    w.position_embedding = load_w(index, "model.vision_tower.embeddings.position_embedding.weight");

    w.post_layernorm_w = load_w(index, "model.vision_tower.post_layernorm.weight");
    w.post_layernorm_b = load_w(index, "model.vision_tower.post_layernorm.bias");

    w.layers.resize(cfg.num_hidden_layers);
    for (int64_t i = 0; i < cfg.num_hidden_layers; ++i) {
        auto& lw = w.layers[i];
        const int li = static_cast<int>(i);
        lw.layer_norm1_w = load_layer_w(index, li, "layer_norm1.weight");
        lw.layer_norm1_b = load_layer_w(index, li, "layer_norm1.bias");
        lw.q_w = load_layer_w(index, li, "self_attn.q_proj.weight");
        lw.q_b = load_layer_w(index, li, "self_attn.q_proj.bias");
        lw.k_w = load_layer_w(index, li, "self_attn.k_proj.weight");
        lw.k_b = load_layer_w(index, li, "self_attn.k_proj.bias");
        lw.v_w = load_layer_w(index, li, "self_attn.v_proj.weight");
        lw.v_b = load_layer_w(index, li, "self_attn.v_proj.bias");
        lw.out_proj_w = load_layer_w(index, li, "self_attn.out_proj.weight");
        lw.out_proj_b = load_layer_w(index, li, "self_attn.out_proj.bias");
        lw.layer_norm2_w = load_layer_w(index, li, "layer_norm2.weight");
        lw.layer_norm2_b = load_layer_w(index, li, "layer_norm2.bias");
        lw.fc1_w = load_layer_w(index, li, "mlp.fc1.weight");
        lw.fc1_b = load_layer_w(index, li, "mlp.fc1.bias");
        lw.fc2_w = load_layer_w(index, li, "mlp.fc2.weight");
        lw.fc2_b = load_layer_w(index, li, "mlp.fc2.bias");
    }

    auto load_merger = [&](const std::string& p) {
        VitMergerWeights m;
        m.layer_norm1_w = load_w(index, p + ".layer_norm1.weight");
        m.layer_norm1_b = load_w(index, p + ".layer_norm1.bias");
        m.q_w = load_w(index, p + ".self_attn.q_proj.weight");
        m.q_b = load_w(index, p + ".self_attn.q_proj.bias");
        m.k_w = load_w(index, p + ".self_attn.k_proj.weight");
        m.k_b = load_w(index, p + ".self_attn.k_proj.bias");
        m.v_w = load_w(index, p + ".self_attn.v_proj.weight");
        m.v_b = load_w(index, p + ".self_attn.v_proj.bias");
        m.out_proj_w = load_w(index, p + ".self_attn.out_proj.weight");
        m.out_proj_b = load_w(index, p + ".self_attn.out_proj.bias");
        m.pre_norm_w = load_w(index, p + ".pre_norm.weight");
        m.pre_norm_b = load_w(index, p + ".pre_norm.bias");
        m.linear_1_w = load_w(index, p + ".linear_1.weight");
        m.linear_1_b = load_w(index, p + ".linear_1.bias");
        m.linear_2_w = load_w(index, p + ".linear_2.weight");
        m.linear_2_b = load_w(index, p + ".linear_2.bias");
        return m;
    };
    w.vit_merger = load_merger("model.vision_tower.vit_merger");

    // Outer merger: model.merger.mlp.{i}.{linear_1, linear_2, pre_norm}.{weight,bias}
    // For MiniCPM-V 4.6 merger_times=1 → one downsample MLP that projects to LLM.
    DownsampleMlpWeights mlp;
    mlp.pre_norm_w = load_w(index, "model.merger.mlp.0.pre_norm.weight");
    mlp.pre_norm_b = load_w(index, "model.merger.mlp.0.pre_norm.bias");
    mlp.linear_1_w = load_w(index, "model.merger.mlp.0.linear_1.weight");
    mlp.linear_1_b = load_w(index, "model.merger.mlp.0.linear_1.bias");
    mlp.linear_2_w = load_w(index, "model.merger.mlp.0.linear_2.weight");
    mlp.linear_2_b = load_w(index, "model.merger.mlp.0.linear_2.bias");
    w.merger_mlp.push_back(std::move(mlp));

    return w;
}

void vision_patch_embed(const Tensor& pixel_values,
                        const Tensor& weight,
                        const Tensor& bias,
                        int64_t patch_size,
                        Tensor& out,
                        aclrtStream stream) {
    if (pixel_values.shape().size() != 4 || pixel_values.shape()[1] != 3) {
        throw std::runtime_error("patch_embed expects [1, 3, H, W] pixel_values");
    }
    const int64_t H = pixel_values.shape()[2];
    const int64_t W = pixel_values.shape()[3];
    if (H % patch_size != 0 || W % patch_size != 0) {
        throw std::runtime_error("patch_embed H/W must be divisible by patch_size");
    }
    const int64_t hidden = weight.shape()[0];
    const int64_t Hp = H / patch_size;
    const int64_t Wp = W / patch_size;
    const int64_t P = Hp * Wp;
    if (out.shape() != std::vector<int64_t>{1, P, hidden}) {
        throw std::runtime_error("patch_embed out shape must be [1, P, hidden]");
    }

    // Conv2d → [1, hidden, Hp, Wp]
    Tensor conv_out({1, hidden, Hp, Wp}, DType::Float16); conv_out.allocate();
    conv2d(pixel_values, weight, &bias, {patch_size, patch_size}, {0, 0}, conv_out, stream);

    // Flatten [hidden, Hp*Wp] then transpose → [P, hidden]. Equivalent to
    // patch_embeds.flatten(2).transpose(1, 2) on the HF side. We just need
    // to reinterpret the [1, hidden, P] buffer as [P, hidden] which is a
    // permute (1, 0). Use a strided copy via aclrtMemcpy with manual layout
    // — for the initial implementation do this on host (P is small ~1024).
    std::vector<uint16_t> conv_host(static_cast<size_t>(hidden * P));
    conv_out.copy_to_host(conv_host.data(), conv_host.size() * sizeof(uint16_t));
    std::vector<uint16_t> trans_host(static_cast<size_t>(P * hidden));
    for (int64_t p = 0; p < P; ++p) {
        for (int64_t h = 0; h < hidden; ++h) {
            trans_host[p * hidden + h] = conv_host[h * P + p];
        }
    }
    out.copy_from_host(trans_host.data(), trans_host.size() * sizeof(uint16_t));
}

void vision_position_embed(const Tensor& position_table,
                           int64_t target_h,
                           int64_t target_w,
                           int64_t num_per_side,
                           Tensor& out,
                           aclrtStream stream) {
    if (position_table.shape().size() != 2 || position_table.shape()[0] != num_per_side * num_per_side) {
        throw std::runtime_error("position_table shape must be [num_per_side^2, hidden]");
    }
    const int64_t hidden = position_table.shape()[1];
    const int64_t total = target_h * target_w;
    if (out.shape() != std::vector<int64_t>{total, hidden}) {
        throw std::runtime_error("position_embed out shape must be [target_h*target_w, hidden]");
    }

    // Reproduce HF's bucketize indexing exactly. boundaries =
    //   [1/N, 2/N, ..., (N-1)/N]
    // where N = num_per_side. For target axis with target_n bins,
    //   fractional_coords = [0, 1/target_n, 2/target_n, ..., (target_n-1)/target_n]
    //   bucket = bucketize(fractional_coords, boundaries, right=True)
    // which gives bucket[i] = #(boundaries <= fractional_coords[i]).
    // Then pos_id[i, j] = bucket_h[i] * N + bucket_w[j].
    auto bucket_axis = [&](int64_t target_n) {
        std::vector<int64_t> b(target_n);
        for (int64_t i = 0; i < target_n; ++i) {
            const double frac = static_cast<double>(i) / static_cast<double>(target_n);
            // count boundaries that are <= frac. boundaries[k] = (k+1)/num_per_side.
            // (k+1)/N <= frac  ⇔  k <= frac*N - 1  ⇔  k < frac*N.
            // Hence count = floor(frac * N + tiny_eps) clipped to [0, N-1] equivalent.
            int64_t cnt = static_cast<int64_t>(std::floor(frac * num_per_side + 1e-9));
            if (cnt < 0) cnt = 0;
            if (cnt > num_per_side - 1) cnt = num_per_side - 1;
            b[i] = cnt;
        }
        return b;
    };
    std::vector<int64_t> bh = bucket_axis(target_h);
    std::vector<int64_t> bw = bucket_axis(target_w);

    std::vector<int32_t> pos_ids;
    pos_ids.reserve(total);
    for (int64_t i = 0; i < target_h; ++i) {
        for (int64_t j = 0; j < target_w; ++j) {
            pos_ids.push_back(static_cast<int32_t>(bh[i] * num_per_side + bw[j]));
        }
    }
    embedding_lookup(position_table, pos_ids, out, stream);
    (void)f32_to_f16_bits;
}

void vision_encoder_layer(const Tensor& hidden,
                          const VisionLayerWeights& w,
                          int64_t num_heads,
                          double layer_norm_eps,
                          Tensor& out,
                          aclrtStream stream) {
    if (hidden.shape().size() != 3) throw std::runtime_error("encoder layer expects [1, T, H]");
    if (hidden.shape() != out.shape()) throw std::runtime_error("encoder layer shape mismatch");

    const int64_t B = hidden.shape()[0];
    const int64_t T = hidden.shape()[1];
    const int64_t H = hidden.shape()[2];
    if (B != 1) throw std::runtime_error("encoder layer batch must be 1");
    if (H % num_heads != 0) throw std::runtime_error("hidden must divide num_heads");
    const int64_t D = H / num_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    // -- pre-attn LN on [1, T, H]; layer_norm normalizes the last dim.
    Tensor ln1({B, T, H}, DType::Float16); ln1.allocate();
    layer_norm(hidden, w.layer_norm1_w, w.layer_norm1_b, ln1, layer_norm_eps, stream);

    // Reinterpret ln1 [1, T, H] as [T, H] for matmuls.
    Tensor ln1_2d({T, H}, DType::Float16); ln1_2d.allocate();
    aclrtMemcpyAsync(ln1_2d.data(), ln1_2d.size_bytes(),
                     ln1.data(), ln1.size_bytes(),
                     ACL_MEMCPY_DEVICE_TO_DEVICE, stream);

    Tensor q({T, H}, DType::Float16); q.allocate();
    Tensor k({T, H}, DType::Float16); k.allocate();
    Tensor v({T, H}, DType::Float16); v.allocate();
    linear_bias(ln1_2d, w.q_w, &w.q_b, q, stream);
    linear_bias(ln1_2d, w.k_w, &w.k_b, k, stream);
    linear_bias(ln1_2d, w.v_w, &w.v_b, v, stream);

    Tensor q3({T, num_heads, D}, DType::Float16);
    Tensor k3({T, num_heads, D}, DType::Float16);
    Tensor v3({T, num_heads, D}, DType::Float16);
    q3.allocate(); aclrtMemcpyAsync(q3.data(), q3.size_bytes(), q.data(), q.size_bytes(),
                                    ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
    k3.allocate(); aclrtMemcpyAsync(k3.data(), k3.size_bytes(), k.data(), k.size_bytes(),
                                    ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
    v3.allocate(); aclrtMemcpyAsync(v3.data(), v3.size_bytes(), v.data(), v.size_bytes(),
                                    ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
    Tensor qh({num_heads, T, D}, DType::Float16); qh.allocate();
    Tensor kh({num_heads, T, D}, DType::Float16); kh.allocate();
    Tensor vh({num_heads, T, D}, DType::Float16); vh.allocate();
    permute(q3, {1, 0, 2}, qh, stream);
    permute(k3, {1, 0, 2}, kh, stream);
    permute(v3, {1, 0, 2}, vh, stream);

    Tensor qh_scaled({num_heads, T, D}, DType::Float16); qh_scaled.allocate();
    muls(qh, scale, qh_scaled, stream);
    Tensor kh_t({num_heads, D, T}, DType::Float16); kh_t.allocate();
    permute(kh, {0, 2, 1}, kh_t, stream);
    Tensor scores({num_heads, T, T}, DType::Float16); scores.allocate();
    batch_matmul(qh_scaled, kh_t, scores, stream);

    Tensor probs({num_heads, T, T}, DType::Float16); probs.allocate();
    softmax_last_dim(scores, probs, stream);

    Tensor attn_h({num_heads, T, D}, DType::Float16); attn_h.allocate();
    batch_matmul(probs, vh, attn_h, stream);
    Tensor attn_thd({T, num_heads, D}, DType::Float16); attn_thd.allocate();
    permute(attn_h, {1, 0, 2}, attn_thd, stream);
    Tensor attn_th({T, H}, DType::Float16); attn_th.allocate();
    aclrtMemcpyAsync(attn_th.data(), attn_th.size_bytes(),
                     attn_thd.data(), attn_thd.size_bytes(),
                     ACL_MEMCPY_DEVICE_TO_DEVICE, stream);

    Tensor attn_out({T, H}, DType::Float16); attn_out.allocate();
    linear_bias(attn_th, w.out_proj_w, &w.out_proj_b, attn_out, stream);

    Tensor after_attn({B, T, H}, DType::Float16); after_attn.allocate();
    Tensor attn_out_3d({B, T, H}, DType::Float16); attn_out_3d.allocate();
    aclrtMemcpyAsync(attn_out_3d.data(), attn_out_3d.size_bytes(),
                     attn_out.data(), attn_out.size_bytes(),
                     ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
    add(hidden, attn_out_3d, after_attn, stream);

    Tensor ln2({B, T, H}, DType::Float16); ln2.allocate();
    layer_norm(after_attn, w.layer_norm2_w, w.layer_norm2_b, ln2, layer_norm_eps, stream);

    Tensor ln2_2d({T, H}, DType::Float16); ln2_2d.allocate();
    aclrtMemcpyAsync(ln2_2d.data(), ln2_2d.size_bytes(),
                     ln2.data(), ln2.size_bytes(),
                     ACL_MEMCPY_DEVICE_TO_DEVICE, stream);

    const int64_t IM = w.fc1_w.shape()[0];
    Tensor fc1_out({T, IM}, DType::Float16); fc1_out.allocate();
    linear_bias(ln2_2d, w.fc1_w, &w.fc1_b, fc1_out, stream);

    Tensor fc1_act({T, IM}, DType::Float16); fc1_act.allocate();
    gelu(fc1_out, /*tanh_approx=*/true, fc1_act, stream);

    Tensor fc2_out({T, H}, DType::Float16); fc2_out.allocate();
    linear_bias(fc1_act, w.fc2_w, &w.fc2_b, fc2_out, stream);

    Tensor fc2_out_3d({B, T, H}, DType::Float16); fc2_out_3d.allocate();
    aclrtMemcpyAsync(fc2_out_3d.data(), fc2_out_3d.size_bytes(),
                     fc2_out.data(), fc2_out.size_bytes(),
                     ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
    add(after_attn, fc2_out_3d, out, stream);
}

}  // namespace minicpmv
