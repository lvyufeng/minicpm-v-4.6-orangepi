#pragma once

#include "minicpmv/tensor.h"
#include "minicpmv/weights.h"

#include <cstdint>
#include <string>

namespace minicpmv {

struct W4A16QuantizedWeight {
    Tensor w_int8;
    Tensor scales;
    int64_t K{0};
    int64_t N{0};
    int64_t group_size{128};
};

struct W8A8QuantizedWeight {
    Tensor w_int8;
    Tensor w_scale;
    int64_t K{0};
    int64_t N{0};
};

bool has_w4a16_quantized_weight(const WeightsIndex& index, const std::string& base);
W4A16QuantizedWeight load_w4a16_quantized_weight(const WeightsIndex& index, const std::string& base);
W8A8QuantizedWeight quantize_dense_weight_w8a8(const Tensor& weight_kn);

}  // namespace minicpmv
