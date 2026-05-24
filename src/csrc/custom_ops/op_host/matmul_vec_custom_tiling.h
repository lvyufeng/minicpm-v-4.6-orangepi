#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MatmulVecCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, N);
    TILING_DATA_FIELD_DEF(uint32_t, K);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MatmulVecCustom, MatmulVecCustomTilingData)
}  // namespace optiling
