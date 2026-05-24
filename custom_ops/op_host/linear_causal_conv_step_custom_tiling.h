#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(LinearCausalConvStepCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, channels);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LinearCausalConvStepCustom, LinearCausalConvStepCustomTilingData)
}  // namespace optiling
