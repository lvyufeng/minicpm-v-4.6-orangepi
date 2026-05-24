#include "minicpmv/acl_context.h"
#include "minicpmv/decoder_layer.h"
#include "minicpmv/language_model.h"
#include "minicpmv/ops.h"
#include "minicpmv/tensor.h"
#include "minicpmv/weights.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
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
    std::string embeddings_path;
    std::string weights_path;
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
    return b;
}

}  // namespace

int main(int argc, char** argv) {
    std::string bundle_path;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--bundle" && i + 1 < argc) bundle_path = argv[++i];
    }
    if (bundle_path.empty()) {
        std::cerr << "usage: minicpmv_hybrid_decode --bundle <prefix>.json" << std::endl;
        return 2;
    }
    try {
        Bundle b = read_bundle(bundle_path);
        if (b.embeddings_dtype != "float16") throw std::runtime_error("only float16 embeddings supported");
        if (b.embeddings_endianness != "little") throw std::runtime_error("only little-endian supported");

        AclContext ctx(0);
        WeightsIndex index(b.weights_path);
        LanguageModelConfig cfg = default_minicpmv46_lm_config();
        if (b.hidden_size != cfg.hidden_size) throw std::runtime_error("hidden_size mismatch");
        if (b.vocab_size != cfg.vocab_size) throw std::runtime_error("vocab_size mismatch");
        if (!b.layer_types.empty()) cfg.layer_types = b.layer_types;
        LanguageModelWeights w = load_language_model_weights(index, cfg);

        std::vector<uint16_t> embeds_host(static_cast<size_t>(b.seq_len * cfg.hidden_size));
        {
            std::ifstream ef(b.embeddings_path, std::ios::binary);
            if (!ef) throw std::runtime_error("failed to open embeddings: " + b.embeddings_path);
            ef.read(reinterpret_cast<char*>(embeds_host.data()),
                    static_cast<std::streamsize>(embeds_host.size() * sizeof(uint16_t)));
            if (!ef) throw std::runtime_error("short read on embeddings file");
        }
        Tensor prompt_hidden({b.seq_len, cfg.hidden_size}, DType::Float16);
        prompt_hidden.copy_from_host(embeds_host.data(), embeds_host.size() * sizeof(uint16_t));

        Tensor cos_t, sin_t;
        build_rope_tables(b.max_seq_len, cfg, cos_t, sin_t);

        DecodeState state = make_decode_state(b.max_seq_len,
                                              cfg.layer_types,
                                              FullAttentionDecoderLayerConfig{cfg.num_q_heads, cfg.num_kv_heads,
                                                                              cfg.head_dim, cfg.rotary_dim, cfg.rms_epsilon},
                                              ctx.stream());

        Tensor last_hidden = prefill_from_embeddings(prompt_hidden, w, cfg, cos_t, sin_t, state, ctx.stream());
        int64_t next_tok = lm_head_greedy(last_hidden, w, cfg, ctx.stream());
        std::cout << next_tok << std::endl;
        std::cout.flush();

        const char* reason = "max";
        int64_t emitted = 1;
        if (is_eos(next_tok, b.eos_token_ids)) {
            reason = "eos";
        } else {
            for (int64_t step = 1; step < b.max_new_tokens; ++step) {
                if (state.seq_len >= state.max_seq_len) {
                    reason = "max_seq_len";
                    break;
                }
                next_tok = decode_step_greedy(static_cast<int32_t>(next_tok), w, cfg,
                                              cos_t, sin_t, state, ctx.stream());
                std::cout << next_tok << std::endl;
                std::cout.flush();
                ++emitted;
                if (is_eos(next_tok, b.eos_token_ids)) {
                    reason = "eos";
                    break;
                }
            }
        }

        std::cout << "# done reason=" << reason << " steps=" << emitted << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "minicpmv_hybrid_decode error: " << e.what() << std::endl;
        return 1;
    }
}
