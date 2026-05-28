#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MatmulW8a8I32CustomTilingData)
    TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cubeTiling);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MatmulW8a8I32Custom, MatmulW8a8I32CustomTilingData)
}  // namespace optiling
