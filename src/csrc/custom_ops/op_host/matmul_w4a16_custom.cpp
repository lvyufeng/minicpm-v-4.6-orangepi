#include "matmul_w4a16_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    MatmulW4a16CustomTilingData tiling;
    // x is [1, K] fp16; w is packed [K/128*8*tilesPerBlock*128, tileLen] int8; out is [1, N].
    const gert::StorageShape* x_shape = context->GetInputShape(0);
    const gert::StorageShape* w_shape = context->GetInputShape(1);
    const gert::StorageShape* out_shape = context->GetOutputShape(0);
    const uint32_t K = static_cast<uint32_t>(x_shape->GetStorageShape().GetDim(1));
    const uint32_t N = static_cast<uint32_t>(out_shape->GetStorageShape().GetDim(1));
    const uint32_t tileLen = static_cast<uint32_t>(w_shape->GetStorageShape().GetDim(1));
    const uint32_t blockLen = (N + 7) / 8;
    const uint32_t tilesPerBlock = (blockLen + tileLen - 1) / tileLen;
    tiling.set_K(K);
    tiling.set_N(N);
    tiling.set_blockLen(blockLen);
    tiling.set_tileLen(tileLen);
    tiling.set_tilesPerBlock(tilesPerBlock);
    tiling.set_groupSize(128);
    context->SetBlockDim(8);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
    const gert::Shape* scales_shape = context->GetInputShape(2);
    gert::Shape* out_shape = context->GetOutputShape(0);
    out_shape->SetDimNum(2);
    out_shape->SetDim(0, 1);
    out_shape->SetDim(1, scales_shape->GetDim(1));
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
    context->SetOutputDataType(0, ge::DT_FLOAT16);
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class MatmulW4a16Custom : public OpDef {
public:
    explicit MatmulW4a16Custom(const char* name) : OpDef(name) {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("w")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("scales")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");
    }
};

OP_ADD(MatmulW4a16Custom);
}  // namespace ops
