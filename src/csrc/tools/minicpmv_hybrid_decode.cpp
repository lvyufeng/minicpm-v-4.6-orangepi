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
    std::string embeddings_path;  // optional: empty = use engine-side embedding lookup on input_ids
    std::string weights_path;
    std::vector<int64_t> input_ids;  // required for text-only path; used as fallback for image path
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
    std::string bundle_path;
    std::string weights_path_arg;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--server") server_mode = true;
        else if (a == "--bundle" && i + 1 < argc) bundle_path = argv[++i];
        else if (a == "--weights" && i + 1 < argc) weights_path_arg = argv[++i];
    }
    if (!server_mode && bundle_path.empty()) {
        std::cerr << "usage:\n"
                  << "  minicpmv_hybrid_decode --bundle <prefix>.json     (single-shot)\n"
                  << "  minicpmv_hybrid_decode --server --weights <path>  (server mode: one bundle path per stdin line)\n"
                  << std::endl;
        return 2;
    }
    try {
        AclContext ctx(0);
        LanguageModelConfig cfg = default_minicpmv46_lm_config();

        std::string weights_path;
        if (!weights_path_arg.empty()) {
            weights_path = weights_path_arg;
        } else {
            // Single-shot mode: take weights_path + layer_types from the bundle so
            // existing callers don't need to pass them explicitly.
            Bundle b = read_bundle(bundle_path);
            weights_path = b.weights_path;
            if (!b.layer_types.empty()) cfg.layer_types = b.layer_types;
        }

        std::cerr << "# loading weights..." << std::endl;
        WeightsIndex index(weights_path);
        LanguageModelWeights w = load_language_model_weights(index, cfg);

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
                    process_bundle(line, w, cfg, cos_t, sin_t, ctx.stream());
                } catch (const std::exception& e) {
                    std::cout << "# error: " << e.what() << std::endl;
                    std::cout.flush();
                }
            }
        } else {
            process_bundle(bundle_path, w, cfg, cos_t, sin_t, ctx.stream());
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "minicpmv_hybrid_decode error: " << e.what() << std::endl;
        return 1;
    }
}
