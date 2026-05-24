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

    // Load just patch_embed + position_embedding + layer-0 weights to keep
    // NPU memory tiny (full 27-layer load is ~1.5 GB and may interact with
    // kernel-cache eviction during the encoder forward).
    VisionWeights vw;
    vw.patch_embedding_w = index.load_to_device_as("model.vision_tower.embeddings.patch_embedding.weight", DType::Float16);
    vw.patch_embedding_b = index.load_to_device_as("model.vision_tower.embeddings.patch_embedding.bias", DType::Float16);
    vw.position_embedding = index.load_to_device_as("model.vision_tower.embeddings.position_embedding.weight", DType::Float16);
    vw.layers.resize(1);
    auto& l0 = vw.layers[0];
    auto L = [&](const char* s) {
        return index.load_to_device_as(std::string("model.vision_tower.encoder.layers.0.") + s, DType::Float16);
    };
    l0.layer_norm1_w = L("layer_norm1.weight");
    l0.layer_norm1_b = L("layer_norm1.bias");
    l0.q_w = L("self_attn.q_proj.weight");
    l0.q_b = L("self_attn.q_proj.bias");
    l0.k_w = L("self_attn.k_proj.weight");
    l0.k_b = L("self_attn.k_proj.bias");
    l0.v_w = L("self_attn.v_proj.weight");
    l0.v_b = L("self_attn.v_proj.bias");
    l0.out_proj_w = L("self_attn.out_proj.weight");
    l0.out_proj_b = L("self_attn.out_proj.bias");
    l0.layer_norm2_w = L("layer_norm2.weight");
    l0.layer_norm2_b = L("layer_norm2.bias");
    l0.fc1_w = L("mlp.fc1.weight");
    l0.fc1_b = L("mlp.fc1.bias");
    l0.fc2_w = L("mlp.fc2.weight");
    l0.fc2_b = L("mlp.fc2.bias");

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
    return 0;
}

