#include "matmul_w8a8_i32_custom_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "register/op_def_registry.h"

using namespace matmul_tiling;

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    const gert::StorageShape* xShape = context->GetInputShape(0);
    const gert::StorageShape* wShape = context->GetInputShape(1);
    const int32_t M = static_cast<int32_t>(xShape->GetStorageShape().GetDim(0));
    const int32_t K = static_cast<int32_t>(xShape->GetStorageShape().GetDim(1));
    const int32_t N = static_cast<int32_t>(wShape->GetStorageShape().GetDim(1));

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    MultiCoreMatmulTiling api(ascendcPlatform);
    api.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_INT8, /*transpose=*/false);
    api.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_INT8, /*transpose=*/false);
    api.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_INT32);
    api.SetBiasType(TPosition::GM, CubeFormat::ND, DataType::DT_INT32);
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
        blockDim = (N + singleN - 1) / singleN;
        api.SetSingleShape(M, singleN, -1);
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
    cubeTiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                            context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(cubeTiling.GetDataSize());
    context->SetBlockDim(static_cast<uint32_t>(blockDim));
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
    const gert::Shape* x_shape = context->GetInputShape(0);
    const gert::Shape* w_shape = context->GetInputShape(1);
    gert::Shape* out_shape = context->GetOutputShape(0);
    out_shape->SetDimNum(2);
    out_shape->SetDim(0, x_shape->GetDim(0));
    out_shape->SetDim(1, w_shape->GetDim(1));
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
    context->SetOutputDataType(0, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class MatmulW8a8I32Custom : public OpDef {
public:
    explicit MatmulW8a8I32Custom(const char* name) : OpDef(name) {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("w")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");
    }
};

OP_ADD(MatmulW8a8I32Custom);
}  // namespace ops
