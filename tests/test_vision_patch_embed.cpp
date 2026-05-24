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

    // Load full vision tower weights via the existing helper.
    VisionWeights vw = load_vision_weights(index, cfg);

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

    // Run remaining 20 encoder layers (7..26) on the downsampled sequence.
    Tensor cur2({1, Pm, Hidden}, DType::Float16); cur2.allocate();
    aclrtMemcpyAsync(cur2.data(), cur2.size_bytes(), vm_out.data(), vm_out.size_bytes(),
                     ACL_MEMCPY_DEVICE_TO_DEVICE, ctx.stream());
    aclrtSynchronizeStream(ctx.stream());
    for (int li = 7; li < 27; ++li) {
        Tensor next({1, Pm, Hidden}, DType::Float16); next.allocate();
        vision_encoder_layer(cur2, vw.layers[li], Heads, cfg.layer_norm_eps, next, ctx.stream());
        aclrtMemcpyAsync(cur2.data(), cur2.size_bytes(), next.data(), next.size_bytes(),
                         ACL_MEMCPY_DEVICE_TO_DEVICE, ctx.stream());
        aclrtSynchronizeStream(ctx.stream());
    }
    {
        std::vector<uint16_t> ho(static_cast<size_t>(Pm * Hidden));
        cur2.copy_to_host(ho.data(), ho.size() * 2);
        auto ref = read_f16("/tmp/vision_ref_encoder_final.f16", ho.size() * 2);
        compare(ho, ref, "encoder_layer26_stack", 5.0f);
    }

    // post_layernorm
    Tensor post_out({1, Pm, Hidden}, DType::Float16); post_out.allocate();
    layer_norm(cur2, vw.post_layernorm_w, vw.post_layernorm_b, post_out, cfg.layer_norm_eps, ctx.stream());
    {
        std::vector<uint16_t> ho(static_cast<size_t>(Pm * Hidden));
        post_out.copy_to_host(ho.data(), ho.size() * 2);
        auto ref = read_f16("/tmp/vision_ref_post_ln.f16", ho.size() * 2);
        compare(ho, ref, "post_layernorm", 5.0f);
    }

    // merger.mlp: P=256 → P/(2*2)=64 tokens at LM hidden = 1024.
    const int64_t Pf = Pm / (cfg.merge_h * cfg.merge_w);
    const int64_t LLM = cfg.llm_hidden_size;
    Tensor image_features({1, Pf, LLM}, DType::Float16); image_features.allocate();
    vision_merger_mlp(post_out, target_h / cfg.window_h, target_w / cfg.window_w,
                      cfg.merge_h, cfg.merge_w, vw.merger_mlp[0], cfg.layer_norm_eps,
                      image_features, ctx.stream());
    {
        std::vector<uint16_t> ho(static_cast<size_t>(Pf * LLM));
        image_features.copy_to_host(ho.data(), ho.size() * 2);
        auto ref = read_f16("/tmp/vision_ref_image_features.f16", ho.size() * 2);
        compare(ho, ref, "image_features", 5.0f);
    }
    return 0;
}

