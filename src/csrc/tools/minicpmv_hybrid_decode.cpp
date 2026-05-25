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
#include <unordered_map>
#include <utility>
#include <vector>

using namespace minicpmv;

// Minimal JSON value reader for the small flat schema we emit on the Python
// side. Supports integers, strings, lists of integers, lists of strings, and
// list-of-int-pairs ([[h, w], ...]).
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
    int64_t seq_len{0};
    int64_t hidden_size{0};
    int64_t max_seq_len{0};
    int64_t max_new_tokens{0};
    int64_t vocab_size{0};
    std::vector<int64_t> eos_token_ids;
    std::vector<std::string> layer_types;
    std::string embeddings_dtype;
    std::string embeddings_endianness;
    std::string embeddings_path;
    std::string weights_path;
    std::vector<int64_t> input_ids;
    std::string image_pixels_path;
    std::vector<std::pair<int64_t, int64_t>> image_slice_sizes;
    int64_t image_token_id{248056};
    std::string conversation_id;
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
    if (parse_key(s, "input_ids", pos)) b.input_ids = parse_int_list(s, pos);
    if (parse_key(s, "image_pixels_path", pos)) b.image_pixels_path = parse_string(s, pos);
    if (parse_key(s, "image_slice_sizes", pos)) {
        b.image_slice_sizes = parse_int_pair_list(s, pos);
    } else {
        int64_t single_h = 0, single_w = 0;
        if (parse_key(s, "image_target_h", pos)) single_h = parse_int(s, pos);
        if (parse_key(s, "image_target_w", pos)) single_w = parse_int(s, pos);
        if (single_h > 0 && single_w > 0) b.image_slice_sizes.emplace_back(single_h, single_w);
    }
    if (parse_key(s, "image_token_id", pos)) b.image_token_id = parse_int(s, pos);
    if (parse_key(s, "conversation_id", pos)) b.conversation_id = parse_string(s, pos);
    return b;
}

struct ConversationState {
    DecodeState state;
    std::vector<int32_t> prefix_input_ids;
    std::vector<int64_t> eos_token_ids;
    int64_t max_seq_len{0};
    Tensor last_hidden_1xH;
};

int64_t longest_common_prefix(const std::vector<int32_t>& a,
                              const std::vector<int32_t>& b) {
    const int64_t n = std::min<int64_t>(a.size(), b.size());
    int64_t i = 0;
    while (i < n && a[i] == b[i]) ++i;
    return i;
}

}  // namespace

static Tensor build_prompt_hidden(const Bundle& b,
                                  const LanguageModelWeights& w,
                                  const LanguageModelConfig& cfg,
                                  const VisionWeights* vw,
                                  const VisionConfig& vcfg,
                                  aclrtStream stream) {
    Tensor prompt_hidden({b.seq_len, cfg.hidden_size}, DType::Float16);
    if (b.embeddings_path.empty()) {
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
        std::ifstream ef(b.embeddings_path, std::ios::binary);
        if (!ef) throw std::runtime_error("failed to open embeddings: " + b.embeddings_path);
        ef.read(reinterpret_cast<char*>(embeds_host.data()),
                static_cast<std::streamsize>(embeds_host.size() * sizeof(uint16_t)));
        if (!ef) throw std::runtime_error("short read on embeddings file");
        prompt_hidden.copy_from_host(embeds_host.data(), embeds_host.size() * sizeof(uint16_t));
    }

    if (!b.image_pixels_path.empty()) {
        if (vw == nullptr) throw std::runtime_error("bundle has image but engine started without vision weights");
        if (b.image_slice_sizes.empty()) {
            throw std::runtime_error("bundle has image_pixels_path but no image_slice_sizes");
        }
        const int64_t patch = vcfg.patch_size;
        int64_t total_P = 0;
        for (auto& s : b.image_slice_sizes) total_P += s.first * s.second;
        const int64_t img_h = patch;
        const int64_t img_w = total_P * patch;
        std::vector<uint16_t> pix_host(static_cast<size_t>(3 * img_h * img_w));
        std::ifstream pf(b.image_pixels_path, std::ios::binary);
        if (!pf) throw std::runtime_error("failed to open image pixels: " + b.image_pixels_path);
        pf.read(reinterpret_cast<char*>(pix_host.data()),
                static_cast<std::streamsize>(pix_host.size() * sizeof(uint16_t)));
        if (!pf) throw std::runtime_error("short read on image pixels file");

        std::vector<std::pair<int64_t, int64_t>> runs;
        {
            int64_t run_start = -1, run_len = 0;
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
            throw std::runtime_error("image-token runs in input_ids (" + std::to_string(runs.size()) + ") != slice count (" + std::to_string(b.image_slice_sizes.size()) + ")");
        }

        const size_t row_bytes = static_cast<size_t>(cfg.hidden_size) * sizeof(uint16_t);
        int64_t w_offset = 0;
        for (size_t si = 0; si < b.image_slice_sizes.size(); ++si) {
            const int64_t target_h = b.image_slice_sizes[si].first;
            const int64_t target_w = b.image_slice_sizes[si].second;
            const int64_t P = target_h * target_w;
            const int64_t slice_w = P * patch;

            Tensor pixels({1, 3, img_h, slice_w}, DType::Float16); pixels.allocate();
            std::vector<uint16_t> slice_pix(static_cast<size_t>(3 * img_h * slice_w));
            for (int64_t c = 0; c < 3; ++c) {
                for (int64_t r = 0; r < img_h; ++r) {
                    const size_t src_off = static_cast<size_t>(c * img_h * img_w + r * img_w + w_offset);
                    const size_t dst_off = static_cast<size_t>(c * img_h * slice_w + r * slice_w);
                    std::memcpy(&slice_pix[dst_off], &pix_host[src_off], static_cast<size_t>(slice_w) * sizeof(uint16_t));
                }
            }
            pixels.copy_from_host(slice_pix.data(), slice_pix.size() * sizeof(uint16_t));
            w_offset += slice_w;

            Tensor patch_out({1, P, vcfg.hidden_size}, DType::Float16); patch_out.allocate();
            vision_patch_embed(pixels, vw->patch_embedding_w, vw->patch_embedding_b, patch, patch_out, stream);
            Tensor pe({P, vcfg.hidden_size}, DType::Float16); pe.allocate();
            vision_position_embed(vw->position_embedding, target_h, target_w, vcfg.image_size / patch, pe, stream);
            Tensor pe_3d({1, P, vcfg.hidden_size}, DType::Float16); pe_3d.allocate();
            check_acl(aclrtMemcpyAsync(pe_3d.data(), pe_3d.size_bytes(), pe.data(), pe.size_bytes(), ACL_MEMCPY_DEVICE_TO_DEVICE, stream), "vision pe expand");
            Tensor full({1, P, vcfg.hidden_size}, DType::Float16); full.allocate();
            add(patch_out, pe_3d, full, stream);

            Tensor cur({1, P, vcfg.hidden_size}, DType::Float16); cur.allocate();
            check_acl(aclrtMemcpyAsync(cur.data(), cur.size_bytes(), full.data(), full.size_bytes(), ACL_MEMCPY_DEVICE_TO_DEVICE, stream), "vision full->cur");
            for (int li = 0; li <= vcfg.insert_layer_id; ++li) {
                Tensor next({1, P, vcfg.hidden_size}, DType::Float16); next.allocate();
                vision_encoder_layer(cur, vw->layers[li], vcfg.num_attention_heads, vcfg.layer_norm_eps, next, stream);
                check_acl(aclrtMemcpyAsync(cur.data(), cur.size_bytes(), next.data(), next.size_bytes(), ACL_MEMCPY_DEVICE_TO_DEVICE, stream), "vision encoder pre-merge copy");
            }
            const int64_t Pm = P / (vcfg.window_h * vcfg.window_w);
            Tensor vm({1, Pm, vcfg.hidden_size}, DType::Float16); vm.allocate();
            vision_vit_merger(cur, target_h, target_w, vcfg.window_h, vcfg.window_w,
                              vcfg.num_attention_heads, vw->vit_merger, vcfg.layer_norm_eps,
                              vm, stream);
            Tensor cur2({1, Pm, vcfg.hidden_size}, DType::Float16); cur2.allocate();
            check_acl(aclrtMemcpyAsync(cur2.data(), cur2.size_bytes(), vm.data(), vm.size_bytes(), ACL_MEMCPY_DEVICE_TO_DEVICE, stream), "vision vm->cur2");
            for (int li = vcfg.insert_layer_id + 1; li < vcfg.num_hidden_layers; ++li) {
                Tensor next({1, Pm, vcfg.hidden_size}, DType::Float16); next.allocate();
                vision_encoder_layer(cur2, vw->layers[li], vcfg.num_attention_heads, vcfg.layer_norm_eps, next, stream);
                check_acl(aclrtMemcpyAsync(cur2.data(), cur2.size_bytes(), next.data(), next.size_bytes(), ACL_MEMCPY_DEVICE_TO_DEVICE, stream), "vision encoder post-merge copy");
            }
            Tensor post({1, Pm, vcfg.hidden_size}, DType::Float16); post.allocate();
            layer_norm(cur2, vw->post_layernorm_w, vw->post_layernorm_b, post, vcfg.layer_norm_eps, stream);
            const int64_t Pf = Pm / (vcfg.merge_h * vcfg.merge_w);
            Tensor image_features({1, Pf, vcfg.llm_hidden_size}, DType::Float16); image_features.allocate();
            vision_merger_mlp(post,
                              target_h / vcfg.window_h,
                              target_w / vcfg.window_w,
                              vcfg.merge_h, vcfg.merge_w,
                              vw->merger_mlp[0], vcfg.layer_norm_eps,
                              image_features, stream);

            const int64_t run_start = runs[si].first;
            const int64_t run_len = runs[si].second;
            if (run_len != Pf) {
                throw std::runtime_error("slice " + std::to_string(si) + " expected " + std::to_string(Pf) + " image tokens in input_ids but got run of " + std::to_string(run_len));
            }
            for (int64_t k = 0; k < Pf; ++k) {
                void* dst = static_cast<uint8_t*>(prompt_hidden.data()) + static_cast<size_t>(run_start + k) * row_bytes;
                const void* src = static_cast<const uint8_t*>(image_features.data()) + static_cast<size_t>(k) * row_bytes;
                check_acl(aclrtMemcpyAsync(dst, row_bytes, src, row_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream), "image scatter");
            }
        }
        check_acl(aclrtSynchronizeStream(stream), "image scatter sync");
    }
    return prompt_hidden;
}

static Tensor advance_suffix_tokens(const std::vector<int32_t>& suffix_tokens,
                                    const LanguageModelWeights& w,
                                    const LanguageModelConfig& cfg,
                                    const Tensor& cos_t,
                                    const Tensor& sin_t,
                                    ConversationState& conv,
                                    aclrtStream stream) {
    Tensor last_hidden({1, cfg.hidden_size}, DType::Float16); last_hidden.allocate();
    bool saw_any = false;
    for (int32_t tok : suffix_tokens) {
        if (conv.state.seq_len >= conv.state.max_seq_len) {
            throw std::runtime_error("prefix cache state full while replaying suffix");
        }
        Tensor hidden({1, cfg.hidden_size}, DType::Float16); hidden.allocate();
        Tensor next({1, cfg.hidden_size}, DType::Float16); next.allocate();
        embedding_lookup(w.embed, {tok}, hidden, stream);

        int full_i = 0;
        int linear_i = 0;
        for (int64_t layer = 0; layer < cfg.num_layers; ++layer) {
            const auto& lw = w.layers[layer];
            if (cfg.layer_types[layer] == "linear_attention") {
                LinearAttentionDecoderLayerConfig lcfg{cfg.rms_epsilon};
                LinearAttentionDecoderLayerWeights ww{
                    &lw.input_norm_w, &lw.post_norm_w, &lw.qkv_w, &lw.z_w, &lw.a_w,
                    &lw.b_w, &lw.conv_w, &lw.dt_bias, &lw.a_log, &lw.gated_norm_w,
                    &lw.out_proj_w, &lw.gate_w, &lw.up_w, &lw.down_w,
                    &lw.conv_w_step_t,
                };
                linear_attention_decoder_layer_step(hidden, ww, lcfg, conv.state.linear[linear_i], next, stream);
                ++linear_i;
            } else {
                FullAttentionDecoderLayerConfig fcfg{cfg.num_q_heads, cfg.num_kv_heads,
                                                     cfg.head_dim, cfg.rotary_dim, cfg.rms_epsilon};
                FullAttentionDecoderLayerWeights ww{
                    &lw.input_norm_w, &lw.post_norm_w, &lw.q_w, &lw.k_w, &lw.v_w,
                    &lw.o_w, &lw.q_norm_w, &lw.k_norm_w, &lw.gate_w, &lw.up_w, &lw.down_w,
                };
                full_attention_decoder_layer_step(hidden, ww, cos_t, sin_t,
                                                  static_cast<int32_t>(conv.state.seq_len),
                                                  conv.state.seq_len, fcfg,
                                                  conv.state.full[full_i], next, stream);
                ++full_i;
            }
            check_acl(aclrtMemcpyAsync(hidden.data(), hidden.size_bytes(), next.data(), next.size_bytes(),
                                       ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                      "prefix-cache suffix advance copy");
        }
        ++conv.state.seq_len;
        conv.prefix_input_ids.push_back(tok);
        check_acl(aclrtMemcpyAsync(last_hidden.data(), last_hidden.size_bytes(), hidden.data(), hidden.size_bytes(),
                                   ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                  "prefix-cache save last hidden");
        saw_any = true;
    }
    if (!saw_any) {
        if (conv.last_hidden_1xH.data() == nullptr) {
            throw std::runtime_error("no cached last_hidden available for exact-prefix cache hit");
        }
        check_acl(aclrtMemcpyAsync(last_hidden.data(), last_hidden.size_bytes(),
                                   conv.last_hidden_1xH.data(), conv.last_hidden_1xH.size_bytes(),
                                   ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                  "prefix-cache load cached last hidden");
    }
    check_acl(aclrtSynchronizeStream(stream), "prefix-cache suffix sync");
    return last_hidden;
}

static void stream_generation_from_hidden(const Tensor& last_hidden,
                                          const Bundle& b,
                                          const LanguageModelWeights& w,
                                          const LanguageModelConfig& cfg,
                                          const Tensor& cos_t,
                                          const Tensor& sin_t,
                                          ConversationState* conv,
                                          DecodeState* state,
                                          aclrtStream stream) {
    int64_t next_tok = lm_head_greedy(last_hidden, w, cfg, stream);
    std::cout << next_tok << std::endl;
    std::cout.flush();
    if (conv != nullptr) conv->prefix_input_ids.push_back(static_cast<int32_t>(next_tok));

    const char* reason = "max";
    int64_t emitted = 1;
    if (is_eos(next_tok, b.eos_token_ids)) {
        reason = "eos";
    } else {
        for (int64_t step = 1; step < b.max_new_tokens; ++step) {
            if (state->seq_len >= state->max_seq_len) { reason = "max_seq_len"; break; }
            next_tok = decode_step_greedy(static_cast<int32_t>(next_tok), w, cfg,
                                          cos_t, sin_t, *state, stream);
            if (conv != nullptr) conv->prefix_input_ids.push_back(static_cast<int32_t>(next_tok));
            std::cout << next_tok << std::endl;
            std::cout.flush();
            ++emitted;
            if (is_eos(next_tok, b.eos_token_ids)) { reason = "eos"; break; }
        }
    }
    std::cout << "# done reason=" << reason << " steps=" << emitted << std::endl;
    std::cout.flush();
}

static void process_bundle(const std::string& bundle_path,
                           const LanguageModelWeights& w,
                           const LanguageModelConfig& cfg,
                           const VisionWeights* vw,
                           const VisionConfig& vcfg,
                           const Tensor& cos_t,
                           const Tensor& sin_t,
                           std::unordered_map<std::string, ConversationState>& conversations,
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

    if (b.conversation_id.empty()) {
        Tensor prompt_hidden = build_prompt_hidden(b, w, cfg, vw, vcfg, stream);
        DecodeState state = make_decode_state(
            b.max_seq_len, cfg.layer_types,
            FullAttentionDecoderLayerConfig{cfg.num_q_heads, cfg.num_kv_heads,
                                            cfg.head_dim, cfg.rotary_dim, cfg.rms_epsilon},
            stream);
        Tensor last_hidden = prefill_from_embeddings(prompt_hidden, w, cfg, cos_t, sin_t, state, stream);
        stream_generation_from_hidden(last_hidden, b, w, cfg, cos_t, sin_t, nullptr, &state, stream);
        return;
    }

    std::vector<int32_t> request_ids;
    request_ids.reserve(b.input_ids.size());
    for (auto v : b.input_ids) request_ids.push_back(static_cast<int32_t>(v));

    auto rebuild = [&]() -> ConversationState& {
        conversations.erase(b.conversation_id);
        Tensor prompt_hidden = build_prompt_hidden(b, w, cfg, vw, vcfg, stream);
        ConversationState fresh;
        fresh.max_seq_len = b.max_seq_len;
        fresh.eos_token_ids = b.eos_token_ids;
        fresh.state = make_decode_state(
            b.max_seq_len, cfg.layer_types,
            FullAttentionDecoderLayerConfig{cfg.num_q_heads, cfg.num_kv_heads,
                                            cfg.head_dim, cfg.rotary_dim, cfg.rms_epsilon},
            stream);
        fresh.last_hidden_1xH = Tensor({1, cfg.hidden_size}, DType::Float16);
        fresh.last_hidden_1xH.allocate();
        Tensor last_hidden = prefill_from_embeddings(prompt_hidden, w, cfg, cos_t, sin_t, fresh.state, stream);
        check_acl(aclrtMemcpyAsync(fresh.last_hidden_1xH.data(), fresh.last_hidden_1xH.size_bytes(),
                                   last_hidden.data(), last_hidden.size_bytes(),
                                   ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                  "cache rebuild save last_hidden");
        check_acl(aclrtSynchronizeStream(stream), "cache rebuild save last_hidden sync");
        fresh.prefix_input_ids = request_ids;
        auto [it, _] = conversations.emplace(b.conversation_id, std::move(fresh));
        return it->second;
    };

    auto it = conversations.find(b.conversation_id);
    if (it == conversations.end()) {
        ConversationState& conv = rebuild();
        stream_generation_from_hidden(conv.last_hidden_1xH, b, w, cfg, cos_t, sin_t, &conv, &conv.state, stream);
        return;
    }

    ConversationState& conv = it->second;
    if (b.max_seq_len > conv.max_seq_len) {
        ConversationState& fresh = rebuild();
        stream_generation_from_hidden(fresh.last_hidden_1xH, b, w, cfg, cos_t, sin_t, &fresh, &fresh.state, stream);
        return;
    }

    const int64_t lcp = longest_common_prefix(conv.prefix_input_ids, request_ids);
    const bool monotonic_extend = (lcp == static_cast<int64_t>(conv.prefix_input_ids.size()));
    if (!monotonic_extend) {
        ConversationState& fresh = rebuild();
        stream_generation_from_hidden(fresh.last_hidden_1xH, b, w, cfg, cos_t, sin_t, &fresh, &fresh.state, stream);
        return;
    }

    const int64_t old_len = static_cast<int64_t>(conv.prefix_input_ids.size());

    std::vector<int32_t> suffix(request_ids.begin() + old_len, request_ids.end());
    if (static_cast<int64_t>(suffix.size()) + conv.state.seq_len > conv.state.max_seq_len) {
        ConversationState& fresh = rebuild();
        stream_generation_from_hidden(fresh.last_hidden_1xH, b, w, cfg, cos_t, sin_t, &fresh, &fresh.state, stream);
        return;
    }

    Tensor last_hidden = advance_suffix_tokens(suffix, w, cfg, cos_t, sin_t, conv, stream);
    check_acl(aclrtMemcpyAsync(conv.last_hidden_1xH.data(), conv.last_hidden_1xH.size_bytes(),
                               last_hidden.data(), last_hidden.size_bytes(),
                               ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
              "cache suffix save last_hidden");
    check_acl(aclrtSynchronizeStream(stream), "cache suffix save last_hidden sync");
    stream_generation_from_hidden(conv.last_hidden_1xH, b, w, cfg, cos_t, sin_t, &conv, &conv.state, stream);
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

        constexpr int64_t kMaxRopeSeq = 8192;
        Tensor cos_t, sin_t;
        build_rope_tables(kMaxRopeSeq, cfg, cos_t, sin_t);

        std::unordered_map<std::string, ConversationState> conversations;

        if (server_mode) {
            std::cerr << "# server_ready" << std::endl;
            std::string line;
            while (std::getline(std::cin, line)) {
                while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
                if (line.empty()) continue;
                if (line == "quit" || line == "exit") break;
                try {
                    process_bundle(line, w, cfg, vw.get(), vcfg, cos_t, sin_t, conversations, ctx.stream());
                } catch (const std::exception& e) {
                    std::cout << "# error: " << e.what() << std::endl;
                    std::cout.flush();
                }
            }
        } else {
            process_bundle(bundle_path, w, cfg, vw.get(), vcfg, cos_t, sin_t, conversations, ctx.stream());
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "minicpmv_hybrid_decode error: " << e.what() << std::endl;
        return 1;
    }
}
