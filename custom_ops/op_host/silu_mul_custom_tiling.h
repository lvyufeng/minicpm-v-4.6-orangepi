#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SiluMulCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalLen);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SiluMulCustom, SiluMulCustomTilingData)
}  // namespace optiling
