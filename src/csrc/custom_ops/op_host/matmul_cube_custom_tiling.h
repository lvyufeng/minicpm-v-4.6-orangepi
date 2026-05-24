#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
// Embed the standard TCubeTiling as the op's tiling data. The kernel reinterprets
// the raw bytes as a TCubeTiling and feeds it into MatmulImpl::Init.
BEGIN_TILING_DATA_DEF(MatmulCubeCustomTilingData)
    TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cubeTiling);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MatmulCubeCustom, MatmulCubeCustomTilingData)
}  // namespace optiling
