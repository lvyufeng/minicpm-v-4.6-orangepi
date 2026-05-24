#include "minicpmv/acl_context.h"
#include "minicpmv/weights.h"

#include <iostream>
#include <vector>

int main() {
    try {
        minicpmv::AclContext ctx(0);
        minicpmv::WeightsIndex index(minicpmv::default_safetensors_path());
        std::cout << "tensor_count=" << index.size() << "\n";

        const char* names[] = {
            "model.language_model.embed_tokens.weight",
            "model.language_model.norm.weight",
            "model.language_model.layers.0.input_layernorm.weight",
            "model.language_model.layers.0.post_attention_layernorm.weight",
            "model.language_model.layers.23.input_layernorm.weight",
            "model.language_model.layers.23.post_attention_layernorm.weight",
            "model.vision_tower.encoder.layers.0.layer_norm1.weight",
            "model.vision_tower.encoder.layers.26.layer_norm2.weight",
        };

        for (const char* name : names) {
            if (!index.contains(name)) {
                std::cerr << "missing tensor: " << name << "\n";
                return 2;
            }
            auto tensor = index.load_to_device(name);
            std::vector<uint16_t> host(std::min<size_t>(tensor.numel(), 8));
            tensor.copy_to_host(host.data(), host.size() * sizeof(uint16_t));
            std::cout << name << " numel=" << tensor.numel() << " bytes=" << tensor.size_bytes() << " first8=";
            for (auto v : host) {
                std::cout << std::hex << v << ",";
            }
            std::cout << std::dec << "\n";
        }

        if (index.contains("lm_head.weight")) {
            std::cerr << "warning: lm_head.weight present despite tie_word_embeddings=true\n";
        } else {
            std::cout << "ok: lm_head is tied to embed_tokens.weight (no separate tensor)\n";
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
