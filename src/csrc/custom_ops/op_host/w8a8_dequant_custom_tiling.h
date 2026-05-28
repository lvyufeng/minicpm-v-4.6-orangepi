#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(W8a8DequantCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalLen);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(W8a8DequantCustom, W8a8DequantCustomTilingData)
}  // namespace optiling
