#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(LinearGatedDeltaRuleCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, seqLen);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LinearGatedDeltaRuleCustom, LinearGatedDeltaRuleCustomTilingData)
}  // namespace optiling
