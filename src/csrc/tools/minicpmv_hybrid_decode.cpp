#include "minicpmv/acl_context.h"
#include "minicpmv/decoder_layer.h"
#include "minicpmv/language_model.h"
#include "minicpmv/ops.h"
#include "minicpmv/tensor.h"
#include "minicpmv/vision.h"
#include "minicpmv/weights.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace minicpmv;

// Minimal JSON value reader for the small flat schema we emit on the Python
// side. Supports integers, doubles, strings, lists of integers, lists of
// strings.
namespace {

void skip_ws(const std::string& s, size_t& i) {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
}

bool parse_key(const std::string& s, const std::string& key, size_t& pos) {
    std::string pat = "\"" + key + "\"";
    size_t found = s.find(pat);
    if (found == std::string::npos) return false;
    pos = found + pat.size();
    skip_ws(s, pos);
    if (pos >= s.size() || s[pos] != ':') return false;
    ++pos;
    skip_ws(s, pos);
    return true;
}

int64_t parse_int(const std::string& s, size_t& pos) {
    skip_ws(s, pos);
    size_t start = pos;
    if (pos < s.size() && s[pos] == '-') ++pos;
    while (pos < s.size() && std::isdigit(static_cast<unsigned char>(s[pos]))) ++pos;
    if (pos == start) throw std::runtime_error("expected integer in bundle");
    return std::stoll(s.substr(start, pos - start));
}

std::string parse_string(const std::string& s, size_t& pos) {
    skip_ws(s, pos);
    if (pos >= s.size() || s[pos] != '"') throw std::runtime_error("expected string in bundle");
    ++pos;
    std::string out;
    while (pos < s.size() && s[pos] != '"') {
        if (s[pos] == '\\' && pos + 1 < s.size()) {
            out += s[pos + 1];
            pos += 2;
        } else {
            out += s[pos++];
        }
    }
    if (pos >= s.size()) throw std::runtime_error("unterminated string");
    ++pos;
    return out;
}

std::vector<int64_t> parse_int_list(const std::string& s, size_t& pos) {
    skip_ws(s, pos);
    if (pos >= s.size() || s[pos] != '[') throw std::runtime_error("expected list in bundle");
    ++pos;
    std::vector<int64_t> out;
    while (true) {
        skip_ws(s, pos);
        if (pos < s.size() && s[pos] == ']') { ++pos; break; }
        out.push_back(parse_int(s, pos));
        skip_ws(s, pos);
        if (pos < s.size() && s[pos] == ',') { ++pos; continue; }
        if (pos < s.size() && s[pos] == ']') { ++pos; break; }
    }
    return out;
}

std::vector<std::string> parse_string_list(const std::string& s, size_t& pos) {
    skip_ws(s, pos);
    if (pos >= s.size() || s[pos] != '[') throw std::runtime_error("expected list in bundle");
    ++pos;
    std::vector<std::string> out;
    while (true) {
        skip_ws(s, pos);
        if (pos < s.size() && s[pos] == ']') { ++pos; break; }
        out.push_back(parse_string(s, pos));
        skip_ws(s, pos);
        if (pos < s.size() && s[pos] == ',') { ++pos; continue; }
        if (pos < s.size() && s[pos] == ']') { ++pos; break; }
    }
    return out;
}

// Parse list-of-int-pairs: [[h0,w0], [h1,w1], ...]
std::vector<std::pair<int64_t, int64_t>> parse_int_pair_list(const std::string& s, size_t& pos) {
    skip_ws(s, pos);
    if (pos >= s.size() || s[pos] != '[') throw std::runtime_error("expected list in bundle");
    ++pos;
    std::vector<std::pair<int64_t, int64_t>> out;
    while (true) {
        skip_ws(s, pos);
        if (pos < s.size() && s[pos] == ']') { ++pos; break; }
        if (pos >= s.size() || s[pos] != '[') throw std::runtime_error("expected inner pair list");
        ++pos;
        const int64_t a = parse_int(s, pos);
        skip_ws(s, pos);
        if (pos >= s.size() || s[pos] != ',') throw std::runtime_error("expected , in pair");
        ++pos;
        const int64_t b = parse_int(s, pos);
        skip_ws(s, pos);
        if (pos >= s.size() || s[pos] != ']') throw std::runtime_error("expected ] closing pair");
        ++pos;
        out.emplace_back(a, b);
        skip_ws(s, pos);
        if (pos < s.size() && s[pos] == ',') { ++pos; continue; }
        if (pos < s.size() && s[pos] == ']') { ++pos; break; }
    }
    return out;
}

struct Bundle {
    int64_t seq_len;
    int64_t hidden_size;
    int64_t max_seq_len;
    int64_t max_new_tokens;
    int64_t vocab_size;
    std::vector<int64_t> eos_token_ids;
    std::vector<std::string> layer_types;
    std::string embeddings_dtype;
    std::string embeddings_endianness;
    std::string embeddings_path;  // optional: empty = use engine-side embedding lookup on input_ids
    std::string weights_path;
    std::vector<int64_t> input_ids;  // required for text-only path; used as fallback for image path
    // Optional vision inputs. When image_pixels_path is non-empty the engine
    // runs the SigLIP vision tower + merger and scatters the resulting image
    // features into prompt_hidden at every input_ids position equal to
    // image_token_id. With multi-slice the HF processor splits one image
    // into N sub-images; pixel_values are concatenated along W and
    // image_slice_sizes lists each sub-image's [target_h, target_w] patch
    // grid.
    std::string image_pixels_path;
    std::vector<std::pair<int64_t, int64_t>> image_slice_sizes;  // [[h0,w0], ...]
    int64_t image_token_id{248056};
};

Bundle read_bundle(const std::string& json_path) {
    std::ifstream f(json_path);
    if (!f) throw std::runtime_error("failed to open bundle JSON: " + json_path);
    std::stringstream ss;
    ss << f.rdbuf();
    std::string s = ss.str();
    size_t pos = 0;
    Bundle b{};
    if (!parse_key(s, "seq_len", pos)) throw std::runtime_error("bundle missing seq_len");
    b.seq_len = parse_int(s, pos);
    if (!parse_key(s, "hidden_size", pos)) throw std::runtime_error("bundle missing hidden_size");
    b.hidden_size = parse_int(s, pos);
    if (!parse_key(s, "max_seq_len", pos)) throw std::runtime_error("bundle missing max_seq_len");
    b.max_seq_len = parse_int(s, pos);
    if (!parse_key(s, "max_new_tokens", pos)) throw std::runtime_error("bundle missing max_new_tokens");
    b.max_new_tokens = parse_int(s, pos);
    if (!parse_key(s, "vocab_size", pos)) throw std::runtime_error("bundle missing vocab_size");
    b.vocab_size = parse_int(s, pos);
    if (!parse_key(s, "eos_token_ids", pos)) throw std::runtime_error("bundle missing eos_token_ids");
    b.eos_token_ids = parse_int_list(s, pos);
    if (!parse_key(s, "layer_types", pos)) throw std::runtime_error("bundle missing layer_types");
    b.layer_types = parse_string_list(s, pos);
    if (!parse_key(s, "embeddings_dtype", pos)) throw std::runtime_error("bundle missing embeddings_dtype");
    b.embeddings_dtype = parse_string(s, pos);
    if (!parse_key(s, "embeddings_endianness", pos)) throw std::runtime_error("bundle missing embeddings_endianness");
    b.embeddings_endianness = parse_string(s, pos);
    if (!parse_key(s, "embeddings_path", pos)) throw std::runtime_error("bundle missing embeddings_path");
    b.embeddings_path = parse_string(s, pos);
    if (!parse_key(s, "weights_path", pos)) throw std::runtime_error("bundle missing weights_path");
    b.weights_path = parse_string(s, pos);
    if (parse_key(s, "input_ids", pos)) {
        b.input_ids = parse_int_list(s, pos);
    }
    if (parse_key(s, "image_pixels_path", pos)) {
        b.image_pixels_path = parse_string(s, pos);
    }
    if (parse_key(s, "image_slice_sizes", pos)) {
        b.image_slice_sizes = parse_int_pair_list(s, pos);
    } else {
        // Backward-compat: single-slice with image_target_h/w as flat ints.
        int64_t single_h = 0, single_w = 0;
        if (parse_key(s, "image_target_h", pos)) single_h = parse_int(s, pos);
        if (parse_key(s, "image_target_w", pos)) single_w = parse_int(s, pos);
        if (single_h > 0 && single_w > 0) {
            b.image_slice_sizes.emplace_back(single_h, single_w);
        }
    }
    if (parse_key(s, "image_token_id", pos)) {
        b.image_token_id = parse_int(s, pos);
    }
    return b;
}

}  // namespace

// Process one bundle: read the bundle JSON + its embeddings file, run prefill
// from the embeddings, greedy-decode up to b.max_new_tokens or EOS, and stream
// each token id on its own line to stdout. Emits `# done reason=... steps=N`
// when finished. Throws on bundle/IO/shape errors.
static void process_bundle(const std::string& bundle_path,
                           const LanguageModelWeights& w,
                           const LanguageModelConfig& cfg,
                           const VisionWeights* vw,
                           const VisionConfig& vcfg,
                           const Tensor& cos_t,
                           const Tensor& sin_t,
                           aclrtStream stream) {
    Bundle b = read_bundle(bundle_path);
    if (b.hidden_size != cfg.hidden_size) throw std::runtime_error("hidden_size mismatch");
    if (b.vocab_size != cfg.vocab_size) throw std::runtime_error("vocab_size mismatch");
    if (!b.layer_types.empty() && b.layer_types != cfg.layer_types) {
        throw std::runtime_error("layer_types mismatch with server-loaded weights");
    }
    if (b.max_seq_len > cos_t.shape()[0]) {
        throw std::runtime_error("bundle max_seq_len exceeds pre-built rope table size");
    }

    Tensor prompt_hidden({b.seq_len, cfg.hidden_size}, DType::Float16);
    if (b.embeddings_path.empty()) {
        // Text-only path: skip the Python torch_npu embedding lookup (which
        // pays a 30-50s JIT compile per unique seq_len), do it here on the
        // engine side using the already-loaded w.embed.
        if (static_cast<int64_t>(b.input_ids.size()) != b.seq_len) {
            throw std::runtime_error("bundle input_ids count != seq_len in text-only mode");
        }
        std::vector<int32_t> ids;
        ids.reserve(b.input_ids.size());
        for (auto v : b.input_ids) ids.push_back(static_cast<int32_t>(v));
        prompt_hidden.allocate();
        embedding_lookup(w.embed, ids, prompt_hidden, stream);
    } else {
        if (b.embeddings_dtype != "float16") throw std::runtime_error("only float16 embeddings supported");
        if (b.embeddings_endianness != "little") throw std::runtime_error("only little-endian supported");
        std::vector<uint16_t> embeds_host(static_cast<size_t>(b.seq_len * cfg.hidden_size));
        {
            std::ifstream ef(b.embeddings_path, std::ios::binary);
            if (!ef) throw std::runtime_error("failed to open embeddings: " + b.embeddings_path);
            ef.read(reinterpret_cast<char*>(embeds_host.data()),
                    static_cast<std::streamsize>(embeds_host.size() * sizeof(uint16_t)));
            if (!ef) throw std::runtime_error("short read on embeddings file");
        }
        prompt_hidden.copy_from_host(embeds_host.data(), embeds_host.size() * sizeof(uint16_t));
    }

    // ---- optional vision path: run the SigLIP tower once per slice and
    // scatter each slice's image_features into the i-th run of
    // image_token_id positions in input_ids.
    if (!b.image_pixels_path.empty()) {
        if (vw == nullptr) throw std::runtime_error("bundle has image but engine started without vision weights");
        if (b.image_slice_sizes.empty()) {
            throw std::runtime_error("bundle has image_pixels_path but no image_slice_sizes");
        }
        const int64_t patch = vcfg.patch_size;

        // Compute total pixel buffer size: sum of P_i across slices.
        int64_t total_P = 0;
        for (auto& s : b.image_slice_sizes) total_P += s.first * s.second;
        const int64_t img_h = patch;                // 14
        const int64_t img_w = total_P * patch;      // concat along W
        std::vector<uint16_t> pix_host(static_cast<size_t>(3 * img_h * img_w));
        {
            std::ifstream pf(b.image_pixels_path, std::ios::binary);
            if (!pf) throw std::runtime_error("failed to open image pixels: " + b.image_pixels_path);
            pf.read(reinterpret_cast<char*>(pix_host.data()),
                    static_cast<std::streamsize>(pix_host.size() * sizeof(uint16_t)));
            if (!pf) throw std::runtime_error("short read on image pixels file");
        }

        // Locate image-token runs in input_ids — one per slice.
        std::vector<std::pair<int64_t, int64_t>> runs;  // (start, len)
        {
            int64_t run_start = -1;
            int64_t run_len = 0;
            for (int64_t i = 0; i < static_cast<int64_t>(b.input_ids.size()); ++i) {
                if (b.input_ids[i] == b.image_token_id) {
                    if (run_start < 0) { run_start = i; run_len = 1; }
                    else { run_len += 1; }
                } else if (run_start >= 0) {
                    runs.emplace_back(run_start, run_len);
                    run_start = -1; run_len = 0;
                }
            }
            if (run_start >= 0) runs.emplace_back(run_start, run_len);
        }
        if (runs.size() != b.image_slice_sizes.size()) {
            throw std::runtime_error("image-token runs in input_ids (" +
                std::to_string(runs.size()) + ") != slice count (" +
                std::to_string(b.image_slice_sizes.size()) + ")");
        }

        // Loop slices. Each contributes Pf_i = P_i / (window*merge) features
        // (= 16 for the default 2x2 window + 2x2 merge), which scatter into
        // the i-th image-token run in input_ids.
        const size_t row_bytes = static_cast<size_t>(cfg.hidden_size) * sizeof(uint16_t);
        int64_t w_offset = 0;   // running offset into the patch-strip width axis
        for (size_t si = 0; si < b.image_slice_sizes.size(); ++si) {
            const int64_t target_h = b.image_slice_sizes[si].first;
            const int64_t target_w = b.image_slice_sizes[si].second;
            const int64_t P = target_h * target_w;
            const int64_t slice_w = P * patch;

            // Extract this slice's [1, 3, 14, slice_w] subregion. Source is
            // [3, 14, img_w] laid out C-major: for each channel c and row r,
            // copy slice_w contiguous fp16 starting at offset w_offset.
            Tensor pixels({1, 3, img_h, slice_w}, DType::Float16); pixels.allocate();
            std::vector<uint16_t> slice_pix(static_cast<size_t>(3 * img_h * slice_w));
            for (int64_t c = 0; c < 3; ++c) {
                for (int64_t r = 0; r < img_h; ++r) {
                    const size_t src_off = static_cast<size_t>(c * img_h * img_w + r * img_w + w_offset);
                    const size_t dst_off = static_cast<size_t>(c * img_h * slice_w + r * slice_w);
                    std::memcpy(&slice_pix[dst_off], &pix_host[src_off],
                                static_cast<size_t>(slice_w) * sizeof(uint16_t));
                }
            }
            pixels.copy_from_host(slice_pix.data(), slice_pix.size() * sizeof(uint16_t));
            w_offset += slice_w;

            Tensor patch_out({1, P, vcfg.hidden_size}, DType::Float16); patch_out.allocate();
            vision_patch_embed(pixels, vw->patch_embedding_w, vw->patch_embedding_b,
                               patch, patch_out, stream);
            Tensor pe({P, vcfg.hidden_size}, DType::Float16); pe.allocate();
            vision_position_embed(vw->position_embedding,
                                  target_h, target_w,
                                  vcfg.image_size / patch,
                                  pe, stream);
            Tensor pe_3d({1, P, vcfg.hidden_size}, DType::Float16); pe_3d.allocate();
            aclrtMemcpyAsync(pe_3d.data(), pe_3d.size_bytes(), pe.data(), pe.size_bytes(),
                             ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
            Tensor full({1, P, vcfg.hidden_size}, DType::Float16); full.allocate();
            add(patch_out, pe_3d, full, stream);

            Tensor cur({1, P, vcfg.hidden_size}, DType::Float16); cur.allocate();
            aclrtMemcpyAsync(cur.data(), cur.size_bytes(), full.data(), full.size_bytes(),
                             ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
            for (int li = 0; li <= vcfg.insert_layer_id; ++li) {
                Tensor next({1, P, vcfg.hidden_size}, DType::Float16); next.allocate();
                vision_encoder_layer(cur, vw->layers[li], vcfg.num_attention_heads,
                                     vcfg.layer_norm_eps, next, stream);
                aclrtMemcpyAsync(cur.data(), cur.size_bytes(), next.data(), next.size_bytes(),
                                 ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
            }
            const int64_t Pm = P / (vcfg.window_h * vcfg.window_w);
            Tensor vm({1, Pm, vcfg.hidden_size}, DType::Float16); vm.allocate();
            vision_vit_merger(cur, target_h, target_w,
                              vcfg.window_h, vcfg.window_w,
                              vcfg.num_attention_heads, vw->vit_merger, vcfg.layer_norm_eps,
                              vm, stream);
            Tensor cur2({1, Pm, vcfg.hidden_size}, DType::Float16); cur2.allocate();
            aclrtMemcpyAsync(cur2.data(), cur2.size_bytes(), vm.data(), vm.size_bytes(),
                             ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
            for (int li = vcfg.insert_layer_id + 1; li < vcfg.num_hidden_layers; ++li) {
                Tensor next({1, Pm, vcfg.hidden_size}, DType::Float16); next.allocate();
                vision_encoder_layer(cur2, vw->layers[li], vcfg.num_attention_heads,
                                     vcfg.layer_norm_eps, next, stream);
                aclrtMemcpyAsync(cur2.data(), cur2.size_bytes(), next.data(), next.size_bytes(),
                                 ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
            }
            Tensor post({1, Pm, vcfg.hidden_size}, DType::Float16); post.allocate();
            layer_norm(cur2, vw->post_layernorm_w, vw->post_layernorm_b,
                       post, vcfg.layer_norm_eps, stream);
            const int64_t Pf = Pm / (vcfg.merge_h * vcfg.merge_w);
            Tensor image_features({1, Pf, vcfg.llm_hidden_size}, DType::Float16);
            image_features.allocate();
            vision_merger_mlp(post,
                              target_h / vcfg.window_h,
                              target_w / vcfg.window_w,
                              vcfg.merge_h, vcfg.merge_w,
                              vw->merger_mlp[0], vcfg.layer_norm_eps,
                              image_features, stream);

            // Scatter Pf features into the i-th image-token run.
            const int64_t run_start = runs[si].first;
            const int64_t run_len = runs[si].second;
            if (run_len != Pf) {
                throw std::runtime_error("slice " + std::to_string(si) +
                    " expected " + std::to_string(Pf) +
                    " image tokens in input_ids but got run of " + std::to_string(run_len));
            }
            for (int64_t k = 0; k < Pf; ++k) {
                void* dst = static_cast<uint8_t*>(prompt_hidden.data()) +
                            static_cast<size_t>(run_start + k) * row_bytes;
                const void* src = static_cast<const uint8_t*>(image_features.data()) +
                                  static_cast<size_t>(k) * row_bytes;
                check_acl(aclrtMemcpyAsync(dst, row_bytes, src, row_bytes,
                                           ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                          "image scatter");
            }
        }
        check_acl(aclrtSynchronizeStream(stream), "image scatter sync");
    }

    DecodeState state = make_decode_state(b.max_seq_len,
                                          cfg.layer_types,
                                          FullAttentionDecoderLayerConfig{cfg.num_q_heads, cfg.num_kv_heads,
                                                                          cfg.head_dim, cfg.rotary_dim, cfg.rms_epsilon},
                                          stream);

    Tensor last_hidden = prefill_from_embeddings(prompt_hidden, w, cfg, cos_t, sin_t, state, stream);
    int64_t next_tok = lm_head_greedy(last_hidden, w, cfg, stream);
    std::cout << next_tok << std::endl;
    std::cout.flush();

    const char* reason = "max";
    int64_t emitted = 1;
    if (is_eos(next_tok, b.eos_token_ids)) {
        reason = "eos";
    } else {
        for (int64_t step = 1; step < b.max_new_tokens; ++step) {
            if (state.seq_len >= state.max_seq_len) { reason = "max_seq_len"; break; }
            next_tok = decode_step_greedy(static_cast<int32_t>(next_tok), w, cfg,
                                          cos_t, sin_t, state, stream);
            std::cout << next_tok << std::endl;
            std::cout.flush();
            ++emitted;
            if (is_eos(next_tok, b.eos_token_ids)) { reason = "eos"; break; }
        }
    }
    std::cout << "# done reason=" << reason << " steps=" << emitted << std::endl;
    std::cout.flush();
}

int main(int argc, char** argv) {
    bool server_mode = false;
    bool with_vision = false;
    std::string bundle_path;
    std::string weights_path_arg;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--server") server_mode = true;
        else if (a == "--with-vision") with_vision = true;
        else if (a == "--bundle" && i + 1 < argc) bundle_path = argv[++i];
        else if (a == "--weights" && i + 1 < argc) weights_path_arg = argv[++i];
    }
    if (!server_mode && bundle_path.empty()) {
        std::cerr << "usage:\n"
                  << "  minicpmv_hybrid_decode --bundle <prefix>.json [--with-vision]   (single-shot)\n"
                  << "  minicpmv_hybrid_decode --server --weights <path> [--with-vision]\n"
                  << std::endl;
        return 2;
    }
    try {
        AclContext ctx(0);
        LanguageModelConfig cfg = default_minicpmv46_lm_config();
        VisionConfig vcfg = default_minicpmv46_vision_config();

        std::string weights_path;
        if (!weights_path_arg.empty()) {
            weights_path = weights_path_arg;
        } else {
            Bundle b = read_bundle(bundle_path);
            weights_path = b.weights_path;
            if (!b.layer_types.empty()) cfg.layer_types = b.layer_types;
            // Auto-enable vision if the bundle is image-bearing.
            if (!b.image_pixels_path.empty()) with_vision = true;
        }

        std::cerr << "# loading weights..." << std::endl;
        WeightsIndex index(weights_path);
        LanguageModelWeights w = load_language_model_weights(index, cfg);

        std::unique_ptr<VisionWeights> vw;
        if (with_vision) {
            std::cerr << "# loading vision weights..." << std::endl;
            vw = std::make_unique<VisionWeights>(load_vision_weights(index, vcfg));
        }

        // Build rope tables once for a generous max sequence — request bundles
        // are checked against this in process_bundle().
        constexpr int64_t kMaxRopeSeq = 8192;
        Tensor cos_t, sin_t;
        build_rope_tables(kMaxRopeSeq, cfg, cos_t, sin_t);

        if (server_mode) {
            std::cerr << "# server_ready" << std::endl;
            std::string line;
            while (std::getline(std::cin, line)) {
                while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
                if (line.empty()) continue;
                if (line == "quit" || line == "exit") break;
                try {
                    process_bundle(line, w, cfg, vw.get(), vcfg, cos_t, sin_t, ctx.stream());
                } catch (const std::exception& e) {
                    std::cout << "# error: " << e.what() << std::endl;
                    std::cout.flush();
                }
            }
        } else {
            process_bundle(bundle_path, w, cfg, vw.get(), vcfg, cos_t, sin_t, ctx.stream());
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "minicpmv_hybrid_decode error: " << e.what() << std::endl;
        return 1;
    }
}
