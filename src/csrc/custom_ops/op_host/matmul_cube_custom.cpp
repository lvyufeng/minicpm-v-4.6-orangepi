#include "matmul_cube_custom_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "register/op_def_registry.h"

using namespace matmul_tiling;

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    // A: [M, K]. B: [K, N] (natural, NOT transposed — callers must pre-transpose).
    const gert::StorageShape* aShape = context->GetInputShape(0);
    const gert::StorageShape* bShape = context->GetInputShape(1);
    const int32_t M = static_cast<int32_t>(aShape->GetStorageShape().GetDim(0));
    const int32_t K = static_cast<int32_t>(aShape->GetStorageShape().GetDim(1));
    const int32_t N = static_cast<int32_t>(bShape->GetStorageShape().GetDim(1));

    auto* tilingDataRaw = context->GetRawTilingData();
    // Write TCubeTiling directly to the raw buffer rather than through a
    // wrapper struct — avoids any alignment/padding surprises.

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    MultiCoreMatmulTiling api(ascendcPlatform);
    api.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16, /*transpose=*/false);
    api.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16, /*transpose=*/false);
    api.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16);
    api.SetBiasType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16);
    api.SetOrgShape(M, N, K);
    api.SetShape(M, N, K);
    api.SetBias(false);
    api.SetBufferSpace(-1, -1, -1);

    int32_t blockDim = 1;
    if (ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND310B) {
        const int32_t aivNum = static_cast<int32_t>(ascendcPlatform.GetCoreNumAiv());
        const int32_t numCores = aivNum > 0 ? aivNum : 8;
        int32_t singleN = (N + numCores - 1) / numCores;
        constexpr int32_t kAlign = 16;
        singleN = ((singleN + kAlign - 1) / kAlign) * kAlign;
        const int32_t nBlocks = (N + singleN - 1) / singleN;
        api.SetSingleShape(M, singleN, -1);
        blockDim = nBlocks;
        api.SetDim(blockDim);
    } else {
        const int32_t aicNum = static_cast<int32_t>(ascendcPlatform.GetCoreNumAic());
        blockDim = aicNum > 0 ? aicNum : 1;
        api.SetDim(blockDim);
    }

    optiling::TCubeTiling cubeTiling;
    if (api.GetTiling(cubeTiling) == -1) {
        return ge::GRAPH_FAILED;
    }
    cubeTiling.SaveToBuffer(tilingDataRaw->GetData(), tilingDataRaw->GetCapacity());
    tilingDataRaw->SetDataSize(cubeTiling.GetDataSize());
    context->SetBlockDim(static_cast<uint32_t>(blockDim));
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
    const gert::Shape* aShape = context->GetInputShape(0);
    const gert::Shape* bShape = context->GetInputShape(1);
    gert::Shape* outShape = context->GetOutputShape(0);
    outShape->SetDimNum(2);
    outShape->SetDim(0, aShape->GetDim(0));  // M
    outShape->SetDim(1, bShape->GetDim(1));  // N (B is [K, N])
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class MatmulCubeCustom : public OpDef {
public:
    explicit MatmulCubeCustom(const char* name) : OpDef(name) {
        this->Input("a")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("b")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("c")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");
    }
};

OP_ADD(MatmulCubeCustom);
}  // namespace ops
