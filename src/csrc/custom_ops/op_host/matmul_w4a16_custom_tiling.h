#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MatmulW4a16CustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, K);
    TILING_DATA_FIELD_DEF(uint32_t, N);
    TILING_DATA_FIELD_DEF(uint32_t, blockLen);
    TILING_DATA_FIELD_DEF(uint32_t, tileLen);
    TILING_DATA_FIELD_DEF(uint32_t, tilesPerBlock);
    TILING_DATA_FIELD_DEF(uint32_t, groupSize);  // always 128 for this model
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MatmulW4a16Custom, MatmulW4a16CustomTilingData)
}  // namespace optiling
