#include "minicpmv/acl_context.h"
#include "minicpmv/ops.h"
#include "minicpmv/tensor.h"

#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using namespace minicpmv;

int main() {
    AclContext ctx(0);
    constexpr int K = 1024;
    constexpr int N = 3584;

    std::mt19937 rng(0);
    std::uniform_int_distribution<int> xd(-8, 7);
    std::uniform_int_distribution<int> wd(-8, 7);

    std::vector<int8_t> hx(K);
    std::vector<int8_t> hw(static_cast<size_t>(K) * N);
    for (auto& v : hx) v = static_cast<int8_t>(xd(rng));
    for (auto& v : hw) v = static_cast<int8_t>(wd(rng));

    Tensor x({1, K}, DType::Int8);
    Tensor w({K, N}, DType::Int8);
    Tensor out({1, N}, DType::Int32);
    x.copy_from_host(hx.data(), hx.size());
    w.copy_from_host(hw.data(), hw.size());
    out.allocate();

    matmul_w8a8_i32(x, w, out, ctx.stream());

    std::vector<int32_t> ho(N);
    out.copy_to_host(ho.data(), ho.size() * sizeof(int32_t));

    int mismatches = 0;
    int first_bad = -1;
    for (int n = 0; n < N; ++n) {
        int32_t ref = 0;
        for (int k = 0; k < K; ++k) {
            ref += static_cast<int32_t>(hx[k]) * static_cast<int32_t>(hw[static_cast<size_t>(k) * N + n]);
        }
        if (ho[n] != ref) {
            if (first_bad < 0) first_bad = n;
            ++mismatches;
        }
    }

    if (mismatches != 0) {
        std::printf("W8A8 i32 mismatch count=%d first_bad=%d got=%d\n", mismatches, first_bad, ho[first_bad]);
        return 1;
    }
    std::printf("W8A8 i32 exact match K=%d N=%d\n", K, N);
    return 0;
}
