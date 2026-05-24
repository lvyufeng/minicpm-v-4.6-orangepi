#include "linear_causal_conv_step_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    LinearCausalConvStepCustomTilingData tiling;
    const gert::StorageShape* x_shape = context->GetInputShape(0);
    // x is [4, C] fp16; weight_t is [4, C] fp16 (pre-transposed by host).
    const uint32_t channels = static_cast<uint32_t>(x_shape->GetStorageShape().GetDim(1));
    tiling.set_channels(channels);
    context->SetBlockDim(8);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
    const gert::Shape* x_shape = context->GetInputShape(0);
    gert::Shape* out_shape = context->GetOutputShape(0);
    out_shape->SetDimNum(2);
    out_shape->SetDim(0, 1);
    out_shape->SetDim(1, x_shape->GetDim(1));
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class LinearCausalConvStepCustom : public OpDef {
public:
    explicit LinearCausalConvStepCustom(const char* name) : OpDef(name) {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("weight_t")
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

OP_ADD(LinearCausalConvStepCustom);
}  // namespace ops
