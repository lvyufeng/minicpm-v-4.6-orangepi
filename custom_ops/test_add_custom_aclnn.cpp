#include <acl/acl.h>
#include <acl/acl_base.h>
#include <aclnn/acl_meta.h>
#include "aclnn_add_custom.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK_ACL(expr) do { auto _ret = (expr); if (_ret != ACL_SUCCESS) { \
    std::fprintf(stderr, "ACL error %d at %s:%d: %s\n", (int)_ret, __FILE__, __LINE__, #expr); return 1; }} while (0)
#define CHECK_NN(expr) do { auto _ret = (expr); if (_ret != 0) { \
    std::fprintf(stderr, "ACLNN error %d at %s:%d: %s\n", (int)_ret, __FILE__, __LINE__, #expr); return 1; }} while (0)

static uint16_t fp32_to_fp16_bits(float f) {
    uint32_t x = *reinterpret_cast<uint32_t*>(&f);
    uint32_t sign = (x >> 16) & 0x8000;
    int exp = ((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = x & 0x7fffff;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7c00);
    return (uint16_t)(sign | (exp << 10) | (mant >> 13));
}

int main() {
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    const int64_t n = 1024;
    std::vector<uint16_t> hx(n), hy(n), hz(n, 0);
    for (int64_t i = 0; i < n; ++i) {
        hx[i] = fp32_to_fp16_bits((float)i * 0.01f);
        hy[i] = fp32_to_fp16_bits(1.0f);
    }

    void *dx = nullptr, *dy = nullptr, *dz = nullptr;
    CHECK_ACL(aclrtMalloc(&dx, n * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dy, n * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dz, n * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(dx, n * sizeof(uint16_t), hx.data(), n * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(dy, n * sizeof(uint16_t), hy.data(), n * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE));

    int64_t dims[1] = {n};
    int64_t strides[1] = {1};
    aclTensor *tx = aclCreateTensor(dims, 1, ACL_FLOAT16, strides, 0, ACL_FORMAT_ND, dims, 1, dx);
    aclTensor *ty = aclCreateTensor(dims, 1, ACL_FLOAT16, strides, 0, ACL_FORMAT_ND, dims, 1, dy);
    aclTensor *tz = aclCreateTensor(dims, 1, ACL_FLOAT16, strides, 0, ACL_FORMAT_ND, dims, 1, dz);
    if (!tx || !ty || !tz) { std::fprintf(stderr, "aclCreateTensor failed\n"); return 1; }

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    CHECK_NN(aclnnAddCustomGetWorkspaceSize(tx, ty, tz, &workspaceSize, &executor));
    void *workspace = nullptr;
    if (workspaceSize > 0) CHECK_ACL(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_NN(aclnnAddCustom(workspace, workspaceSize, executor, stream));
    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtMemcpy(hz.data(), n * sizeof(uint16_t), dz, n * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST));

    std::printf("workspace=%lu first_bits=%04x mid_bits=%04x last_bits=%04x\n", workspaceSize, hz[0], hz[n/2], hz[n-1]);

    aclDestroyTensor(tx); aclDestroyTensor(ty); aclDestroyTensor(tz);
    if (workspace) aclrtFree(workspace);
    aclrtFree(dx); aclrtFree(dy); aclrtFree(dz);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
