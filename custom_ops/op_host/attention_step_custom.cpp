#include "attention_step_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    AttentionStepCustomTilingData tiling;

    // Inputs: query [num_q_heads*head_dim] (flat or 2D), k_cache [max_seq, num_kv_heads*head_dim].
    const gert::StorageShape* qShape = context->GetInputShape(0);
    const gert::StorageShape* kShape = context->GetInputShape(1);
    int64_t qTotal = 1;
    const auto& qStorageShape = qShape->GetStorageShape();
    for (size_t i = 0; i < qStorageShape.GetDimNum(); ++i) {
        qTotal *= qStorageShape.GetDim(i);
    }
    const int64_t maxSeq = kShape->GetStorageShape().GetDim(0);
    const int64_t kvCols = kShape->GetStorageShape().GetDim(kShape->GetStorageShape().GetDimNum() - 1);

    // Pull dynamic attrs (context, num_q_heads, num_kv_heads, scale).
    const auto* attrs = context->GetAttrs();
    const int64_t* ctxAttr  = attrs->GetAttrPointer<int64_t>(0);
    const int64_t* nqAttr   = attrs->GetAttrPointer<int64_t>(1);
    const int64_t* nkvAttr  = attrs->GetAttrPointer<int64_t>(2);
    const float*   scaleAttr = attrs->GetAttrPointer<float>(3);

    const uint32_t numQHeads  = static_cast<uint32_t>(*nqAttr);
    const uint32_t numKvHeads = static_cast<uint32_t>(*nkvAttr);
    const uint32_t headDim    = static_cast<uint32_t>(qTotal / numQHeads);
    (void)kvCols;

    tiling.set_context(static_cast<uint32_t>(*ctxAttr));
    tiling.set_numQHeads(numQHeads);
    tiling.set_numKvHeads(numKvHeads);
    tiling.set_headDim(headDim);
    tiling.set_maxSeq(static_cast<uint32_t>(maxSeq));
    tiling.set_scale(*scaleAttr);

    context->SetBlockDim(numQHeads);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
    const gert::Shape* qShape = context->GetInputShape(0);
    gert::Shape* outShape = context->GetOutputShape(0);
    *outShape = *qShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class AttentionStepCustom : public OpDef {
public:
    explicit AttentionStepCustom(const char* name) : OpDef(name) {
        this->Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("k_cache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("v_cache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("context").Int();
        this->Attr("num_q_heads").Int();
        this->Attr("num_kv_heads").Int();
        this->Attr("scale").Float();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");
    }
};

OP_ADD(AttentionStepCustom);
}  // namespace ops
