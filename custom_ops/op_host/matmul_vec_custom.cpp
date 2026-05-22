#include "matmul_vec_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    MatmulVecCustomTilingData tiling;

    // x [K] fp16, W [N, K] fp16, out [N] fp16.
    const gert::StorageShape* xShape = context->GetInputShape(0);
    const gert::StorageShape* wShape = context->GetInputShape(1);
    int64_t xTotal = 1;
    const auto& xStorageShape = xShape->GetStorageShape();
    for (size_t i = 0; i < xStorageShape.GetDimNum(); ++i) {
        xTotal *= xStorageShape.GetDim(i);
    }
    const int64_t K = xTotal;
    const int64_t N = wShape->GetStorageShape().GetDim(0);

    tiling.set_N(static_cast<uint32_t>(N));
    tiling.set_K(static_cast<uint32_t>(K));

    context->SetBlockDim(8);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
    const gert::Shape* wShape = context->GetInputShape(1);
    gert::Shape* outShape = context->GetOutputShape(0);
    outShape->SetDimNum(1);
    outShape->SetDim(0, wShape->GetDim(0));
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class MatmulVecCustom : public OpDef {
public:
    explicit MatmulVecCustom(const char* name) : OpDef(name) {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("w")
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

OP_ADD(MatmulVecCustom);
}  // namespace ops
