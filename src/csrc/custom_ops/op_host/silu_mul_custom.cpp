#include "silu_mul_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    SiluMulCustomTilingData tiling;

    const gert::StorageShape* gateShape = context->GetInputShape(0);
    int64_t total = 1;
    const auto& gateStorageShape = gateShape->GetStorageShape();
    for (size_t i = 0; i < gateStorageShape.GetDimNum(); ++i) {
        total *= gateStorageShape.GetDim(i);
    }
    tiling.set_totalLen(static_cast<uint32_t>(total));

    context->SetBlockDim(8);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
    const gert::Shape* gateShape = context->GetInputShape(0);
    gert::Shape* outShape = context->GetOutputShape(0);
    *outShape = *gateShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class SiluMulCustom : public OpDef {
public:
    explicit SiluMulCustom(const char* name) : OpDef(name) {
        this->Input("gate")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("up")
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

OP_ADD(SiluMulCustom);
}  // namespace ops
