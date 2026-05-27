// Probe what dtypes/shapes aclnnWeightQuantBatchMatmulV2 accepts on 310B.
//
// We try: x[M,K] fp16 @ weight[K,N] {int8 or int4-packed} with per-group scale
// [K/group, N] fp16.  Shape: M=1, K=1024, N=3584, group=128.
#include "minicpmv/acl_context.h"
#include "minicpmv/tensor.h"

#include <aclnnop/aclnn_weight_quant_batch_matmul_v2.h>

#include <cstdio>
#include <cstring>
#include <vector>

using namespace minicpmv;

static int try_call(const std::vector<int64_t>& w_shape, aclDataType w_dtype, const char* tag,
                    int M, int K, int N, int group) {
    AclContext ctx(0);
    // x [M, K] fp16
    Tensor x({M, K}, DType::Float16); x.allocate();
    std::vector<uint16_t> hx(M * K, 0x3c00);
    x.copy_from_host(hx.data(), hx.size() * 2);

    // weight tensor with the requested shape/dtype, filled with zeros (just probing the API)
    int64_t w_total = 1;
    for (auto d : w_shape) w_total *= d;
    DType w_d;
    size_t w_bytes;
    switch (w_dtype) {
        case ACL_INT8:  w_d = DType::Int8;  w_bytes = w_total; break;
        case ACL_INT32: w_d = DType::Int32; w_bytes = w_total * 4; break;
        default: w_d = DType::Int8; w_bytes = w_total;
    }
    Tensor w(w_shape, w_d); w.allocate();
    std::vector<uint8_t> hw(w_bytes, 0);
    w.copy_from_host(hw.data(), hw.size());

    // antiquantScale [K/group, N] fp16
    Tensor s({K / group, N}, DType::Float16); s.allocate();
    std::vector<uint16_t> hs(s.numel(), 0x3c00);
    s.copy_from_host(hs.data(), hs.size() * 2);

    // y [M, N] fp16
    Tensor y({M, N}, DType::Float16); y.allocate();

    aclTensor *ax, *aw, *as, *ay;
    {
        std::vector<int64_t> xs = {M, K}, xss = {M, K};
        std::vector<int64_t> str = {K, 1};
        ax = aclCreateTensor(xs.data(), 2, ACL_FLOAT16, str.data(), 0, ACL_FORMAT_ND,
                             xss.data(), 2, x.data());
    }
    {
        std::vector<int64_t> ws = w.shape();
        std::vector<int64_t> wss = ws;
        std::vector<int64_t> str(ws.size(), 1);
        for (int i = ws.size() - 2; i >= 0; --i) str[i] = str[i + 1] * ws[i + 1];
        aw = aclCreateTensor(ws.data(), ws.size(), w_dtype, str.data(), 0, ACL_FORMAT_ND,
                             wss.data(), wss.size(), w.data());
    }
    {
        std::vector<int64_t> ss = s.shape();
        std::vector<int64_t> sss = ss;
        std::vector<int64_t> str = {ss[1], 1};
        as = aclCreateTensor(ss.data(), 2, ACL_FLOAT16, str.data(), 0, ACL_FORMAT_ND,
                             sss.data(), 2, s.data());
    }
    {
        std::vector<int64_t> ys = y.shape();
        std::vector<int64_t> yss = ys;
        std::vector<int64_t> str = {N, 1};
        ay = aclCreateTensor(ys.data(), 2, ACL_FLOAT16, str.data(), 0, ACL_FORMAT_ND,
                             yss.data(), 2, y.data());
    }

    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnWeightQuantBatchMatmulV2GetWorkspaceSize(
        ax, aw, as, /*antiquantOffsetOpt*/nullptr,
        /*quantScaleOpt*/nullptr, /*quantOffsetOpt*/nullptr,
        /*biasOpt*/nullptr, group, ay, &ws_size, &executor);
    std::printf("[probe] %-30s w_shape=[", tag);
    for (size_t i = 0; i < w_shape.size(); ++i) std::printf("%ld%s", w_shape[i], i+1<w_shape.size()?",":"");
    std::printf("] w_dtype=%d  → GetWorkspaceSize ret=%d", (int)w_dtype, (int)ret);
    if (ret == 0) std::printf(" ws=%lu", ws_size);
    std::printf("\n");

    aclDestroyTensor(ax); aclDestroyTensor(aw); aclDestroyTensor(as); aclDestroyTensor(ay);
    return ret;
}

int main() {
    const int M = 1, K = 1024, N = 3584, group = 128;
    // The HF GPTQ qweight has shape [K/8, N] int32 (8 nibbles packed along K).
    // It's also worth testing the unpacked [K, N] int8 form.
    try_call({K, N},       ACL_INT8,  "weight [K,N] int8",         M, K, N, group);
    try_call({N, K},       ACL_INT8,  "weight [N,K] int8 (transp)",M, K, N, group);
    try_call({K / 8, N},   ACL_INT32, "weight [K/8,N] int32 pack", M, K, N, group);
    try_call({K, N / 8},   ACL_INT32, "weight [K,N/8] int32 pack", M, K, N, group);
    // ACL_INT4 (29) is the proper 4-bit element type.
    try_call({K, N},       ACL_INT4,  "weight [K,N] int4 raw",     M, K, N, group);
    try_call({N, K},       ACL_INT4,  "weight [N,K] int4 transp",  M, K, N, group);
    return 0;
}
