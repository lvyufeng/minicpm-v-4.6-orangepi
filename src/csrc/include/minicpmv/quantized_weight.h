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

bool has_w4a16_quantized_weight(const WeightsIndex& index, const std::string& base);
W4A16QuantizedWeight load_w4a16_quantized_weight(const WeightsIndex& index, const std::string& base);

}  // namespace minicpmv
