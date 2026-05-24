#pragma once

#include "minicpmv/tensor.h"
#include "minicpmv/weights.h"

#include <acl/acl.h>

#include <cstdint>
#include <vector>

namespace minicpmv {

struct VisionConfig {
    int64_t hidden_size{1152};           // SigLIP-so400m vision embed dim
    int64_t intermediate_size{4304};     // SigLIP MLP intermediate
    int64_t num_hidden_layers{27};
    int64_t num_attention_heads{16};     // 1152 / 16 = 72 head_dim
    int64_t patch_size{14};
    int64_t image_size{980};             // max image edge (preprocessor target)
    int64_t num_positions{4900};         // 70x70 = 4900 (image_size/patch_size = 980/14 = 70)
    int64_t insert_layer_id{6};          // vit_merger runs after this many encoder layers
    int64_t window_h{2};                 // vit_merger 2x2 spatial window
    int64_t window_w{2};
    int64_t window_hidden_size{4608};    // 1152 * 4
    int64_t window_intermediate_size{17216};
    int64_t merge_h{2};                  // merger.mlp 2x2 spatial reduce
    int64_t merge_w{2};
    int64_t llm_hidden_size{1024};       // final projection to LM space
    double layer_norm_eps{1e-6};
};

VisionConfig default_minicpmv46_vision_config();

struct VisionLayerWeights {
    Tensor layer_norm1_w;       // [1152]
    Tensor layer_norm1_b;       // [1152]
    Tensor q_w;                 // [1152, 1152]
    Tensor q_b;                 // [1152]
    Tensor k_w;
    Tensor k_b;
    Tensor v_w;
    Tensor v_b;
    Tensor out_proj_w;
    Tensor out_proj_b;
    Tensor layer_norm2_w;
    Tensor layer_norm2_b;
    Tensor fc1_w;               // [4304, 1152]
    Tensor fc1_b;               // [4304]
    Tensor fc2_w;               // [1152, 4304]
    Tensor fc2_b;               // [1152]
};

struct VitMergerWeights {
    Tensor layer_norm1_w;       // [1152]
    Tensor layer_norm1_b;
    Tensor q_w, q_b, k_w, k_b, v_w, v_b, out_proj_w, out_proj_b;
    Tensor pre_norm_w;          // [4608]
    Tensor pre_norm_b;
    Tensor linear_1_w;          // [17216, 4608]
    Tensor linear_1_b;
    Tensor linear_2_w;          // [1152, 17216]
    Tensor linear_2_b;
};

struct DownsampleMlpWeights {
    // pre_norm on 4*hidden, linear_1: 4*hidden → 4*hidden, linear_2: 4*hidden → out
    Tensor pre_norm_w;
    Tensor pre_norm_b;
    Tensor linear_1_w;
    Tensor linear_1_b;
    Tensor linear_2_w;
    Tensor linear_2_b;
};

struct VisionWeights {
    // embeddings
    Tensor patch_embedding_w;   // [1152, 3, 14, 14]
    Tensor patch_embedding_b;   // [1152]
    Tensor position_embedding;  // [4900, 1152]
    // encoder
    std::vector<VisionLayerWeights> layers;
    // post-encoder
    Tensor post_layernorm_w;
    Tensor post_layernorm_b;
    // merger inside the encoder (between layers insert_layer_id-1 and insert_layer_id)
    VitMergerWeights vit_merger;
    // outer merger (projects to LLM)
    std::vector<DownsampleMlpWeights> merger_mlp;
};

VisionWeights load_vision_weights(WeightsIndex& index, const VisionConfig& cfg);

// Patch embedding via Conv2d 3→hidden, kernel=patch, stride=patch.
//   pixel_values: [1, 3, H, W] fp16 (H, W must each be divisible by patch_size)
//   out: [1, P, hidden] fp16 where P = (H/patch)*(W/patch)
// Internally runs Conv2d then flatten+transpose.
void vision_patch_embed(const Tensor& pixel_values,
                        const Tensor& weight,   // [hidden, 3, patch, patch]
                        const Tensor& bias,     // [hidden]
                        int64_t patch_size,
                        Tensor& out,             // [1, P, hidden]
                        aclrtStream stream);

// Build position embeddings for a given grid by bucketizing the 70x70 table.
//   position_table: [4900, hidden]
//   target_h, target_w: actual patch grid (e.g. 32, 32 for a 448x448 image)
//   num_per_side: 70 (= image_size/patch_size)
//   out: [target_h * target_w, hidden]
void vision_position_embed(const Tensor& position_table,
                           int64_t target_h,
                           int64_t target_w,
                           int64_t num_per_side,
                           Tensor& out,
                           aclrtStream stream);

// Run one SigLIP transformer block: pre-LayerNorm self-attention + residual,
// pre-LayerNorm GELU MLP + residual.
//   hidden:  [1, T, H] fp16, modified in-place via the out param
//   out:     [1, T, H] fp16
//   scratch: scratch tensors that the caller owns. Required because we
//            can't allocate inside the encoder loop (NPU malloc adds latency).
void vision_encoder_layer(const Tensor& hidden,
                          const VisionLayerWeights& w,
                          int64_t num_heads,
                          double layer_norm_eps,
                          Tensor& out,
                          aclrtStream stream);

// Vit merger — windowed self-attention over 2x2 patch windows + 4-patch
// spatial concat + MLP. Reduces token count by window_h*window_w.
//   hidden:    [1, P, H] fp16    (P = target_h * target_w)
//   out:       [1, P/(wh*ww), H] fp16  (e.g. 1024 → 256 for 32x32 grid)
void vision_vit_merger(const Tensor& hidden,
                       int64_t target_h,
                       int64_t target_w,
                       int64_t window_h,
                       int64_t window_w,
                       int64_t num_heads,
                       const VitMergerWeights& w,
                       double layer_norm_eps,
                       Tensor& out,
                       aclrtStream stream);

}  // namespace minicpmv
