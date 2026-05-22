#include "linear_causal_conv_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    LinearCausalConvCustomTilingData tiling;
    const gert::StorageShape* x_shape = context->GetInputShape(0);
    const uint32_t seq_len = static_cast<uint32_t>(x_shape->GetStorageShape().GetDim(0));
    const uint32_t channels = static_cast<uint32_t>(x_shape->GetStorageShape().GetDim(1));
    tiling.set_totalLength(seq_len * channels);
    tiling.set_seqLen(seq_len);
    tiling.set_channels(channels);
    context->SetBlockDim(8);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
    const gert::Shape* x_shape = context->GetInputShape(0);
    gert::Shape* out_shape = context->GetOutputShape(0);
    *out_shape = *x_shape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class LinearCausalConvCustom : public OpDef {
public:
    explicit LinearCausalConvCustom(const char* name) : OpDef(name) {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("weight")
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

OP_ADD(LinearCausalConvCustom);
}  // namespace ops
