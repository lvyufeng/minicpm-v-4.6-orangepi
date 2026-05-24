// Validate vision_patch_embed + vision_position_embed + vision_encoder_layer
// against a CPU torch reference dumped by /tmp/dump_vision_ref.py.
#include "minicpmv/acl_context.h"
#include "minicpmv/ops.h"
#include "minicpmv/tensor.h"
#include "minicpmv/vision.h"
#include "minicpmv/weights.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

using namespace minicpmv;

static float h2f(uint16_t h) {
    uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    uint32_t out;
    if (exp == 0) { out = sign; }
    else if (exp == 31) { out = sign | 0x7f800000u | (mant << 13); }
    else { out = sign | ((exp + 127 - 15) << 23) | (mant << 13); }
    float f; std::memcpy(&f, &out, sizeof(f)); return f;
}

static std::vector<uint16_t> read_f16(const std::string& path, size_t expect_bytes) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "open failed: %s\n", path.c_str()); std::exit(2); }
    std::vector<uint16_t> v(expect_bytes / 2);
    f.read(reinterpret_cast<char*>(v.data()), expect_bytes);
    if (!f) { std::fprintf(stderr, "short read: %s\n", path.c_str()); std::exit(2); }
    return v;
}

static void compare(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b,
                    const char* label, float tol) {
    float maxAbs = 0.0f;
    int big = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        float d = std::fabs(h2f(a[i]) - h2f(b[i]));
        if (d > maxAbs) maxAbs = d;
        if (d > tol) ++big;
    }
    std::printf("[%s] max_abs_diff=%.4f  large(>%g)=%d/%zu\n", label, maxAbs, tol, big, a.size());
}

int main() {
    AclContext ctx(0);
    const int64_t H = 14, W = 14336, P = 1024;
    const int64_t Hidden = 1152, Heads = 16;
    const int64_t Patch = 14;
    const int64_t N = 70;
    const int64_t target_h = 32, target_w = 32;

    auto pix_host = read_f16("/tmp/vision_ref_pixel_values.f16", 1ull * 3 * H * W * 2);
    Tensor pix({1, 3, H, W}, DType::Float16); pix.allocate();
    pix.copy_from_host(pix_host.data(), pix_host.size() * 2);

    WeightsIndex index(default_safetensors_path());
    VisionConfig cfg = default_minicpmv46_vision_config();

    // Load patch_embed + position_embedding + the first 7 encoder layers
    // (HF applies vit_merger AFTER layer 6, i.e. after running layers 0..6).
    VisionWeights vw;
    vw.patch_embedding_w = index.load_to_device_as("model.vision_tower.embeddings.patch_embedding.weight", DType::Float16);
    vw.patch_embedding_b = index.load_to_device_as("model.vision_tower.embeddings.patch_embedding.bias", DType::Float16);
    vw.position_embedding = index.load_to_device_as("model.vision_tower.embeddings.position_embedding.weight", DType::Float16);
    vw.layers.resize(7);
    for (int li = 0; li < 7; ++li) {
        auto& lw = vw.layers[li];
        auto L = [&](const char* s) {
            return index.load_to_device_as(
                std::string("model.vision_tower.encoder.layers.") + std::to_string(li) + "." + s,
                DType::Float16);
        };
        lw.layer_norm1_w = L("layer_norm1.weight");
        lw.layer_norm1_b = L("layer_norm1.bias");
        lw.q_w = L("self_attn.q_proj.weight");
        lw.q_b = L("self_attn.q_proj.bias");
        lw.k_w = L("self_attn.k_proj.weight");
        lw.k_b = L("self_attn.k_proj.bias");
        lw.v_w = L("self_attn.v_proj.weight");
        lw.v_b = L("self_attn.v_proj.bias");
        lw.out_proj_w = L("self_attn.out_proj.weight");
        lw.out_proj_b = L("self_attn.out_proj.bias");
        lw.layer_norm2_w = L("layer_norm2.weight");
        lw.layer_norm2_b = L("layer_norm2.bias");
        lw.fc1_w = L("mlp.fc1.weight");
        lw.fc1_b = L("mlp.fc1.bias");
        lw.fc2_w = L("mlp.fc2.weight");
        lw.fc2_b = L("mlp.fc2.bias");
    }
    auto WL = [&](const char* s) {
        return index.load_to_device_as(std::string("model.vision_tower.vit_merger.") + s, DType::Float16);
    };
    vw.vit_merger.layer_norm1_w = WL("layer_norm1.weight");
    vw.vit_merger.layer_norm1_b = WL("layer_norm1.bias");
    vw.vit_merger.q_w = WL("self_attn.q_proj.weight");
    vw.vit_merger.q_b = WL("self_attn.q_proj.bias");
    vw.vit_merger.k_w = WL("self_attn.k_proj.weight");
    vw.vit_merger.k_b = WL("self_attn.k_proj.bias");
    vw.vit_merger.v_w = WL("self_attn.v_proj.weight");
    vw.vit_merger.v_b = WL("self_attn.v_proj.bias");
    vw.vit_merger.out_proj_w = WL("self_attn.out_proj.weight");
    vw.vit_merger.out_proj_b = WL("self_attn.out_proj.bias");
    vw.vit_merger.pre_norm_w = WL("pre_norm.weight");
    vw.vit_merger.pre_norm_b = WL("pre_norm.bias");
    vw.vit_merger.linear_1_w = WL("linear_1.weight");
    vw.vit_merger.linear_1_b = WL("linear_1.bias");
    vw.vit_merger.linear_2_w = WL("linear_2.weight");
    vw.vit_merger.linear_2_b = WL("linear_2.bias");

    Tensor patch_out({1, P, Hidden}, DType::Float16); patch_out.allocate();
    vision_patch_embed(pix, vw.patch_embedding_w, vw.patch_embedding_b, Patch, patch_out, ctx.stream());
    {
        std::vector<uint16_t> ho(static_cast<size_t>(P * Hidden));
        patch_out.copy_to_host(ho.data(), ho.size() * 2);
        auto ref = read_f16("/tmp/vision_ref_patch_embed.f16", ho.size() * 2);
        compare(ho, ref, "patch_embed", 0.05f);
    }

    Tensor pe({target_h * target_w, Hidden}, DType::Float16); pe.allocate();
    vision_position_embed(vw.position_embedding, target_h, target_w, N, pe, ctx.stream());

    // Build full input embeddings = patch_out + pe (broadcast pe to [1, P, H])
    Tensor pe_3d({1, P, Hidden}, DType::Float16); pe_3d.allocate();
    aclrtMemcpyAsync(pe_3d.data(), pe_3d.size_bytes(), pe.data(), pe.size_bytes(),
                     ACL_MEMCPY_DEVICE_TO_DEVICE, ctx.stream());
    aclrtSynchronizeStream(ctx.stream());
    Tensor full({1, P, Hidden}, DType::Float16); full.allocate();
    add(patch_out, pe_3d, full, ctx.stream());
    {
        std::vector<uint16_t> ho(static_cast<size_t>(P * Hidden));
        full.copy_to_host(ho.data(), ho.size() * 2);
        auto ref = read_f16("/tmp/vision_ref_embeddings.f16", ho.size() * 2);
        compare(ho, ref, "embeddings", 0.05f);
    }

    // Layer 0
    Tensor layer0_out({1, P, Hidden}, DType::Float16); layer0_out.allocate();
    vision_encoder_layer(full, vw.layers[0], Heads, cfg.layer_norm_eps, layer0_out, ctx.stream());
    {
        std::vector<uint16_t> ho(static_cast<size_t>(P * Hidden));
        layer0_out.copy_to_host(ho.data(), ho.size() * 2);
        auto ref = read_f16("/tmp/vision_ref_layer0.f16", ho.size() * 2);
        compare(ho, ref, "encoder_layer0", 0.2f);
    }

    // Layers 1..6 stacked
    Tensor cur({1, P, Hidden}, DType::Float16); cur.allocate();
    aclrtMemcpyAsync(cur.data(), cur.size_bytes(), layer0_out.data(), layer0_out.size_bytes(),
                     ACL_MEMCPY_DEVICE_TO_DEVICE, ctx.stream());
    aclrtSynchronizeStream(ctx.stream());
    for (int li = 1; li < 7; ++li) {
        Tensor next({1, P, Hidden}, DType::Float16); next.allocate();
        vision_encoder_layer(cur, vw.layers[li], Heads, cfg.layer_norm_eps, next, ctx.stream());
        aclrtMemcpyAsync(cur.data(), cur.size_bytes(), next.data(), next.size_bytes(),
                         ACL_MEMCPY_DEVICE_TO_DEVICE, ctx.stream());
        aclrtSynchronizeStream(ctx.stream());
    }
    {
        std::vector<uint16_t> ho(static_cast<size_t>(P * Hidden));
        cur.copy_to_host(ho.data(), ho.size() * 2);
        auto ref = read_f16("/tmp/vision_ref_layer6.f16", ho.size() * 2);
        compare(ho, ref, "encoder_layer6_stack", 0.5f);
    }

    // vit_merger: reduces P=1024 → P/4=256
    const int64_t Wm = 4;
    const int64_t Pm = P / Wm;
    Tensor vm_out({1, Pm, Hidden}, DType::Float16); vm_out.allocate();
    vision_vit_merger(cur, target_h, target_w, cfg.window_h, cfg.window_w, Heads,
                      vw.vit_merger, cfg.layer_norm_eps, vm_out, ctx.stream());
    {
        std::vector<uint16_t> ho(static_cast<size_t>(Pm * Hidden));
        vm_out.copy_to_host(ho.data(), ho.size() * 2);
        auto ref = read_f16("/tmp/vision_ref_vit_merger.f16", ho.size() * 2);
        compare(ho, ref, "vit_merger", 1.0f);
    }
    return 0;
}

