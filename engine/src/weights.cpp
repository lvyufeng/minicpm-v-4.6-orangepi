#include "minicpmv/weights.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace minicpmv {
namespace {

DType parse_dtype(const std::string& s) {
    if (s == "F16") return DType::Float16;
    if (s == "BF16") return DType::BFloat16;
    if (s == "F32") return DType::Float32;
    if (s == "I32") return DType::Int32;
    if (s == "I64") return DType::Int64;
    if (s == "U8") return DType::UInt8;
    throw std::runtime_error("unsupported safetensors dtype: " + s);
}

uint16_t bf16_bits_to_f16_bits(uint16_t bf) {
    uint32_t bits = static_cast<uint32_t>(bf) << 16;
    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = bits & 0x7fffffu;
    if (exp <= 0) return static_cast<uint16_t>(sign);
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}

void skip_ws(const std::string& s, size_t& i) {
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) {
        ++i;
    }
}

void expect(const std::string& s, size_t& i, char ch) {
    skip_ws(s, i);
    if (i >= s.size() || s[i] != ch) {
        std::ostringstream oss;
        oss << "expected '" << ch << "' at position " << i;
        throw std::runtime_error(oss.str());
    }
    ++i;
}

std::string parse_string(const std::string& s, size_t& i) {
    skip_ws(s, i);
    expect(s, i, '"');
    std::string out;
    while (i < s.size()) {
        char c = s[i++];
        if (c == '"') {
            return out;
        }
        if (c == '\\') {
            if (i >= s.size()) throw std::runtime_error("bad escape in json string");
            char esc = s[i++];
            switch (esc) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                default: throw std::runtime_error("unsupported json escape");
            }
        } else {
            out.push_back(c);
        }
    }
    throw std::runtime_error("unterminated json string");
}

uint64_t parse_uint(const std::string& s, size_t& i) {
    skip_ws(s, i);
    if (i >= s.size() || !std::isdigit(static_cast<unsigned char>(s[i]))) {
        throw std::runtime_error("expected unsigned integer");
    }
    uint64_t value = 0;
    while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) {
        value = value * 10 + static_cast<uint64_t>(s[i] - '0');
        ++i;
    }
    return value;
}

std::vector<int64_t> parse_int64_array(const std::string& s, size_t& i) {
    std::vector<int64_t> out;
    expect(s, i, '[');
    skip_ws(s, i);
    if (i < s.size() && s[i] == ']') {
        ++i;
        return out;
    }
    while (true) {
        out.push_back(static_cast<int64_t>(parse_uint(s, i)));
        skip_ws(s, i);
        if (i < s.size() && s[i] == ',') {
            ++i;
            continue;
        }
        break;
    }
    expect(s, i, ']');
    return out;
}

std::vector<uint64_t> parse_uint64_array(const std::string& s, size_t& i) {
    std::vector<uint64_t> out;
    expect(s, i, '[');
    skip_ws(s, i);
    if (i < s.size() && s[i] == ']') {
        ++i;
        return out;
    }
    while (true) {
        out.push_back(parse_uint(s, i));
        skip_ws(s, i);
        if (i < s.size() && s[i] == ',') {
            ++i;
            continue;
        }
        break;
    }
    expect(s, i, ']');
    return out;
}

void skip_json_value(const std::string& s, size_t& i);

void skip_json_object(const std::string& s, size_t& i) {
    expect(s, i, '{');
    skip_ws(s, i);
    if (i < s.size() && s[i] == '}') {
        ++i;
        return;
    }
    while (true) {
        (void)parse_string(s, i);
        expect(s, i, ':');
        skip_json_value(s, i);
        skip_ws(s, i);
        if (i < s.size() && s[i] == ',') {
            ++i;
            continue;
        }
        break;
    }
    expect(s, i, '}');
}

void skip_json_array(const std::string& s, size_t& i) {
    expect(s, i, '[');
    skip_ws(s, i);
    if (i < s.size() && s[i] == ']') {
        ++i;
        return;
    }
    while (true) {
        skip_json_value(s, i);
        skip_ws(s, i);
        if (i < s.size() && s[i] == ',') {
            ++i;
            continue;
        }
        break;
    }
    expect(s, i, ']');
}

void skip_json_value(const std::string& s, size_t& i) {
    skip_ws(s, i);
    if (i >= s.size()) throw std::runtime_error("unexpected end of json");
    char c = s[i];
    if (c == '{') {
        skip_json_object(s, i);
    } else if (c == '[') {
        skip_json_array(s, i);
    } else if (c == '"') {
        (void)parse_string(s, i);
    } else {
        while (i < s.size() && s[i] != ',' && s[i] != '}' && s[i] != ']') {
            ++i;
        }
    }
}

}  // namespace

WeightsIndex::WeightsIndex(const std::string& safetensors_path) : path_(safetensors_path) {
    std::ifstream in(path_, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open safetensors file: " + path_);
    }
    in.seekg(0, std::ios::end);
    auto size = static_cast<size_t>(in.tellg());
    in.seekg(0, std::ios::beg);
    file_bytes_.resize(size);
    in.read(reinterpret_cast<char*>(file_bytes_.data()), static_cast<std::streamsize>(size));
    parse();
}

const TensorInfo& WeightsIndex::at(const std::string& name) const {
    return tensors_.at(name);
}

bool WeightsIndex::contains(const std::string& name) const {
    return tensors_.find(name) != tensors_.end();
}

std::vector<std::string> WeightsIndex::names() const {
    std::vector<std::string> out;
    out.reserve(tensors_.size());
    for (const auto& kv : tensors_) {
        out.push_back(kv.first);
    }
    std::sort(out.begin(), out.end());
    return out;
}

Tensor WeightsIndex::load_to_device(const std::string& name) const {
    const auto& info = at(name);
    Tensor tensor(info.shape, info.dtype);
    tensor.allocate();
    const uint8_t* base = file_bytes_.data();
    tensor.copy_from_host(base + info.data_begin, info.data_end - info.data_begin);
    return tensor;
}

Tensor WeightsIndex::load_to_device_as(const std::string& name, DType target_dtype) const {
    const auto& info = at(name);
    if (info.dtype == target_dtype) {
        return load_to_device(name);
    }
    if (!(info.dtype == DType::BFloat16 && target_dtype == DType::Float16)) {
        throw std::runtime_error("unsupported dtype conversion in load_to_device_as");
    }

    const size_t numel = [&]() {
        size_t n = 1;
        for (auto d : info.shape) n *= static_cast<size_t>(d);
        return n;
    }();

    std::vector<uint16_t> converted(numel);
    const uint16_t* src = reinterpret_cast<const uint16_t*>(file_bytes_.data() + info.data_begin);
    for (size_t i = 0; i < numel; ++i) {
        converted[i] = bf16_bits_to_f16_bits(src[i]);
    }

    Tensor tensor(info.shape, target_dtype);
    tensor.allocate();
    tensor.copy_from_host(converted.data(), converted.size() * sizeof(uint16_t));
    return tensor;
}

void WeightsIndex::parse() {
    if (file_bytes_.size() < 8) {
        throw std::runtime_error("invalid safetensors file");
    }
    uint64_t header_len = 0;
    std::memcpy(&header_len, file_bytes_.data(), sizeof(uint64_t));
    if (8 + header_len > file_bytes_.size()) {
        throw std::runtime_error("invalid safetensors header length");
    }
    std::string header(reinterpret_cast<const char*>(file_bytes_.data() + 8), static_cast<size_t>(header_len));
    uint64_t data_base = 8 + header_len;

    size_t i = 0;
    expect(header, i, '{');
    skip_ws(header, i);
    if (i < header.size() && header[i] == '}') {
        ++i;
        return;
    }

    while (true) {
        std::string tensor_name = parse_string(header, i);
        expect(header, i, ':');
        if (tensor_name == "__metadata__") {
            skip_json_object(header, i);
        } else {
            expect(header, i, '{');
            TensorInfo info{};
            bool have_dtype = false, have_shape = false, have_offsets = false;
            while (true) {
                std::string key = parse_string(header, i);
                expect(header, i, ':');
                if (key == "dtype") {
                    info.dtype = parse_dtype(parse_string(header, i));
                    have_dtype = true;
                } else if (key == "shape") {
                    info.shape = parse_int64_array(header, i);
                    have_shape = true;
                } else if (key == "data_offsets") {
                    auto offsets = parse_uint64_array(header, i);
                    if (offsets.size() != 2) {
                        throw std::runtime_error("data_offsets size must be 2");
                    }
                    info.data_begin = data_base + offsets[0];
                    info.data_end = data_base + offsets[1];
                    have_offsets = true;
                } else {
                    skip_json_value(header, i);
                }
                skip_ws(header, i);
                if (i < header.size() && header[i] == ',') {
                    ++i;
                    continue;
                }
                break;
            }
            expect(header, i, '}');
            if (!have_dtype || !have_shape || !have_offsets) {
                throw std::runtime_error("incomplete tensor entry for " + tensor_name);
            }
            tensors_.emplace(std::move(tensor_name), std::move(info));
        }
        skip_ws(header, i);
        if (i < header.size() && header[i] == ',') {
            ++i;
            continue;
        }
        break;
    }
    expect(header, i, '}');
}

}  // namespace minicpmv
