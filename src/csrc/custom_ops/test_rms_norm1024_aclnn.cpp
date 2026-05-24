#include <acl/acl.h>
#include <acl/acl_base.h>
#include <aclnn/acl_meta.h>
#include "aclnn_rms_norm1024_custom.h"

#include <algorithm>
#include <cmath>
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

static float fp16_bits_to_fp32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) out = sign;
        else {
            exp = 1;
            while ((mant & 0x400) == 0) { mant <<= 1; exp--; }
            mant &= 0x3ff;
            out = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7f800000 | (mant << 13);
    } else {
        out = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    return *reinterpret_cast<float*>(&out);
}

int main() {
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    const int64_t rows = 8;
    const int64_t hidden = 1024;
    const int64_t n = rows * hidden;
    std::vector<uint16_t> hx(n), hg(hidden), hz(n, 0);
    for (int64_t i = 0; i < n; ++i) {
        float v = std::sin((float)i * 0.013f) * 0.7f + std::cos((float)i * 0.017f) * 0.2f;
        hx[i] = fp32_to_fp16_bits(v);
    }
    for (int64_t i = 0; i < hidden; ++i) {
        float g = 1.0f + 0.01f * std::sin((float)i * 0.03f);
        hg[i] = fp32_to_fp16_bits(g);
    }

    void *dx = nullptr, *dg = nullptr, *dz = nullptr;
    CHECK_ACL(aclrtMalloc(&dx, n * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dg, hidden * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dz, n * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(dx, n * sizeof(uint16_t), hx.data(), n * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(dg, hidden * sizeof(uint16_t), hg.data(), hidden * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE));

    int64_t xDims[2] = {rows, hidden};
    int64_t xStrides[2] = {hidden, 1};
    int64_t gDims[1] = {hidden};
    int64_t gStrides[1] = {1};
    aclTensor *tx = aclCreateTensor(xDims, 2, ACL_FLOAT16, xStrides, 0, ACL_FORMAT_ND, xDims, 2, dx);
    aclTensor *tg = aclCreateTensor(gDims, 1, ACL_FLOAT16, gStrides, 0, ACL_FORMAT_ND, gDims, 1, dg);
    aclTensor *tz = aclCreateTensor(xDims, 2, ACL_FLOAT16, xStrides, 0, ACL_FORMAT_ND, xDims, 2, dz);
    if (!tx || !tg || !tz) { std::fprintf(stderr, "aclCreateTensor failed\n"); return 1; }

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    CHECK_NN(aclnnRmsNorm1024CustomGetWorkspaceSize(tx, tg, 1e-6, tz, &workspaceSize, &executor));
    void *workspace = nullptr;
    if (workspaceSize > 0) CHECK_ACL(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_NN(aclnnRmsNorm1024Custom(workspace, workspaceSize, executor, stream));
    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtMemcpy(hz.data(), n * sizeof(uint16_t), dz, n * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST));

    float maxAbs = 0.0f;
    float meanAbs = 0.0f;
    for (int64_t r = 0; r < rows; ++r) {
        float sumSq = 0.0f;
        for (int64_t i = 0; i < hidden; ++i) {
            float v = fp16_bits_to_fp32(hx[r * hidden + i]);
            sumSq += v * v;
        }
        float invRms = 1.0f / std::sqrt(sumSq / hidden + 1e-6f);
        for (int64_t i = 0; i < hidden; ++i) {
            float ref = fp16_bits_to_fp32(hx[r * hidden + i]) * invRms * fp16_bits_to_fp32(hg[i]);
            float got = fp16_bits_to_fp32(hz[r * hidden + i]);
            float diff = std::fabs(got - ref);
            maxAbs = std::max(maxAbs, diff);
            meanAbs += diff;
        }
    }
    meanAbs /= n;
    std::printf("workspace=%lu max_abs=%f mean_abs=%f first=%f mid=%f last=%f\n",
                workspaceSize, maxAbs, meanAbs,
                fp16_bits_to_fp32(hz[0]), fp16_bits_to_fp32(hz[n/2]), fp16_bits_to_fp32(hz[n-1]));

    aclDestroyTensor(tx); aclDestroyTensor(tg); aclDestroyTensor(tz);
    if (workspace) aclrtFree(workspace);
    aclrtFree(dx); aclrtFree(dg); aclrtFree(dz);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
