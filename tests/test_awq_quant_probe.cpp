// Probe CANN native quant matmul APIs for AWQ-like W4A16 shapes on 310B.
//
// AWQ MiniCPM-V-4.6 stores representative language weights as:
//   qweight [K, N/8] I32, qzeros [K/128, N/8] I32, scales [K/128, N] F16.
// This probe checks whether CANN accepts either that packed shape directly or
// an unpacked int8 [K,N]/[N,K] form with per-group scales.
#include "minicpmv/acl_context.h"
#include "minicpmv/tensor.h"

#include <aclnnop/aclnn_quant_matmul_dequant.h>
#include <aclnnop/aclnn_weight_quant_batch_matmul_v3.h>

#include <cstdio>
#include <string>
#include <vector>

using namespace minicpmv;

struct AclTensorGuard {
    aclTensor* t{nullptr};
    ~AclTensorGuard() { if (t) aclDestroyTensor(t); }
};

static AclTensorGuard make_tensor(const Tensor& src, aclDataType dtype) {
    AclTensorGuard g;
    std::vector<int64_t> shape = src.shape();
    std::vector<int64_t> storage = shape;
    std::vector<int64_t> stride(shape.size(), 1);
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        stride[i] = stride[i + 1] * shape[i + 1];
    }
    g.t = aclCreateTensor(shape.data(), shape.size(), dtype, stride.data(), 0,
                          ACL_FORMAT_ND, storage.data(), storage.size(), src.data());
    return g;
}

static Tensor make_device_tensor(const std::vector<int64_t>& shape, DType dtype) {
    Tensor t(shape, dtype);
    t.allocate();
    std::vector<uint8_t> zeros(t.size_bytes(), 0);
    t.copy_from_host(zeros.data(), zeros.size());
    return t;
}

static int try_weight_quant_v3(const std::vector<int64_t>& w_shape,
                               DType w_dtype,
                               aclDataType w_acl_dtype,
                               const char* tag,
                               int M, int K, int N, int group) {
    AclContext ctx(0);
    Tensor x = make_device_tensor({M, K}, DType::Float16);
    Tensor w = make_device_tensor(w_shape, w_dtype);
    Tensor scale = make_device_tensor({K / group, N}, DType::Float16);
    Tensor y = make_device_tensor({M, N}, DType::Float16);

    auto ax = make_tensor(x, ACL_FLOAT16);
    auto aw = make_tensor(w, w_acl_dtype);
    auto as = make_tensor(scale, ACL_FLOAT16);
    auto ay = make_tensor(y, ACL_FLOAT16);

    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    auto ret = aclnnWeightQuantBatchMatmulV3GetWorkspaceSize(
        ax.t, aw.t, as.t,
        /*antiquantOffsetOptional*/nullptr,
        /*quantScaleOptional*/nullptr,
        /*quantOffsetOptional*/nullptr,
        /*biasOptional*/nullptr,
        group,
        /*innerPrecise*/0,
        ay.t,
        &ws,
        &exec);
    std::printf("[WQBMv3] %-28s w_shape=[", tag);
    for (size_t i = 0; i < w_shape.size(); ++i) std::printf("%ld%s", w_shape[i], i + 1 < w_shape.size() ? "," : "");
    std::printf("] dtype=%d ret=%d", static_cast<int>(w_acl_dtype), static_cast<int>(ret));
    if (ret == 0) std::printf(" ws=%lu", ws);
    std::printf("\n");
    return ret;
}

static int try_quant_dequant(const std::vector<int64_t>& w_shape,
                             DType w_dtype,
                             aclDataType w_acl_dtype,
                             bool transpose_weight,
                             const char* tag,
                             int M, int K, int N, int group) {
    AclContext ctx(0);
    Tensor x = make_device_tensor({M, K}, DType::Float16);
    Tensor w = make_device_tensor(w_shape, w_dtype);
    Tensor scale = make_device_tensor({K / group, N}, DType::Float32);
    Tensor y = make_device_tensor({M, N}, DType::Float16);

    auto ax = make_tensor(x, ACL_FLOAT16);
    auto aw = make_tensor(w, w_acl_dtype);
    auto as = make_tensor(scale, ACL_FLOAT);
    auto ay = make_tensor(y, ACL_FLOAT16);

    char mode[] = "static";
    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    auto ret = aclnnQuantMatmulDequantGetWorkspaceSize(
        ax.t, aw.t, as.t,
        /*biasOptional*/nullptr,
        /*xScaleOptional*/nullptr,
        /*xOffsetOptional*/nullptr,
        /*smoothScaleOptional*/nullptr,
        mode,
        transpose_weight,
        ay.t,
        &ws,
        &exec);
    std::printf("[QMDQ]   %-28s w_shape=[", tag);
    for (size_t i = 0; i < w_shape.size(); ++i) std::printf("%ld%s", w_shape[i], i + 1 < w_shape.size() ? "," : "");
    std::printf("] dtype=%d transpose=%d ret=%d", static_cast<int>(w_acl_dtype), transpose_weight ? 1 : 0, static_cast<int>(ret));
    if (ret == 0) std::printf(" ws=%lu", ws);
    std::printf("\n");
    return ret;
}

int main() {
    const int M = 1;
    const int K = 1024;
    const int N = 3584;
    const int group = 128;

    try_weight_quant_v3({K, N}, ACL_INT8 == ACL_INT8 ? DType::Int8 : DType::Int8, ACL_INT8, "unpacked [K,N] int8", M, K, N, group);
    try_weight_quant_v3({N, K}, DType::Int8, ACL_INT8, "unpacked [N,K] int8", M, K, N, group);
    try_weight_quant_v3({K, N / 8}, DType::Int32, ACL_INT32, "AWQ [K,N/8] int32", M, K, N, group);
    try_weight_quant_v3({K, N}, DType::Int8, ACL_INT4, "raw [K,N] int4", M, K, N, group);
    try_weight_quant_v3({N, K}, DType::Int8, ACL_INT4, "raw [N,K] int4", M, K, N, group);

    try_quant_dequant({K, N}, DType::Int8, ACL_INT8, false, "unpacked [K,N] int8", M, K, N, group);
    try_quant_dequant({N, K}, DType::Int8, ACL_INT8, true, "unpacked [N,K] int8", M, K, N, group);
    try_quant_dequant({K, N / 8}, DType::Int32, ACL_INT32, false, "AWQ [K,N/8] int32", M, K, N, group);
    try_quant_dequant({K, N}, DType::Int8, ACL_INT4, false, "raw [K,N] int4", M, K, N, group);
    try_quant_dequant({N, K}, DType::Int8, ACL_INT4, true, "raw [N,K] int4", M, K, N, group);
    return 0;
}
