#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(W8a8QuantizeCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, K);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(W8a8QuantizeCustom, W8a8QuantizeCustomTilingData)
}  // namespace optiling
