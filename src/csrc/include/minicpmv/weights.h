#pragma once

#include "minicpmv/tensor.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace minicpmv {

struct TensorInfo {
    DType dtype;
    std::vector<int64_t> shape;
    uint64_t data_begin;
    uint64_t data_end;
};

class WeightsIndex {
public:
    explicit WeightsIndex(const std::string& safetensors_path);

    const TensorInfo& at(const std::string& name) const;
    bool contains(const std::string& name) const;
    size_t size() const { return tensors_.size(); }
    std::vector<std::string> names() const;

    Tensor load_to_device(const std::string& name) const;
    Tensor load_to_device_as(const std::string& name, DType target_dtype) const;

private:
    std::string path_;
    std::vector<uint8_t> file_bytes_;
    std::unordered_map<std::string, TensorInfo> tensors_;

    void parse();
};

// Resolve the path to the MiniCPM-V 4.6 safetensors file used by tests and
// benches. Honors the MINICPMV_MODEL_PATH environment variable; falls back to
// `./MiniCPM-V-4.6/model.safetensors` relative to the binary's CWD.
std::string default_safetensors_path();

}  // namespace minicpmv
