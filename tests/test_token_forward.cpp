#include "minicpmv/acl_context.h"
#include "minicpmv/ops.h"
#include "minicpmv/tensor.h"
#include "minicpmv/weights.h"

#include <acl/acl.h>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace minicpmv;

int main() {
    AclContext ctx(0);
    WeightsIndex index(default_safetensors_path());

    if (!index.contains("model.language_model.embed_tokens.weight")) {
        std::cerr << "missing embed_tokens.weight" << std::endl;
        return 1;
    }

    Tensor weight = index.load_to_device_as("model.language_model.embed_tokens.weight", DType::Float16);
    const int64_t vocab = weight.shape()[0];
    const int64_t hidden = weight.shape()[1];
    if (hidden != 1024) {
        std::cerr << "unexpected hidden dim: " << hidden << std::endl;
        return 2;
    }

    std::vector<int32_t> ids = {1, 2, 10, 100, 1000, 12345, 54321, 200000};
    for (auto id : ids) {
        if (id < 0 || id >= vocab) {
            std::cerr << "token id out of range: " << id << std::endl;
            return 3;
        }
    }

    const int64_t N = static_cast<int64_t>(ids.size());
    Tensor hidden_states({N, hidden}, DType::Float16);
    hidden_states.allocate();
    try {
        embedding_lookup(weight, ids, hidden_states, ctx.stream());
    } catch (const std::exception& e) {
        std::cerr << "embedding_lookup failed: " << e.what()
                  << "\nrecent: " << aclGetRecentErrMsg() << std::endl;
        return 4;
    }

    Tensor logits({N, vocab}, DType::Float16);
    logits.allocate();
    try {
        matmul_b_transposed(hidden_states, weight, logits, ctx.stream());
    } catch (const std::exception& e) {
        std::cerr << "matmul failed: " << e.what()
                  << "\nrecent: " << aclGetRecentErrMsg() << std::endl;
        return 5;
    }

    Tensor pred({N}, DType::Int64);
    pred.allocate();
    try {
        argmax_last_dim(logits, pred, ctx.stream());
    } catch (const std::exception& e) {
        std::cerr << "argmax failed: " << e.what()
                  << "\nrecent: " << aclGetRecentErrMsg() << std::endl;
        return 6;
    }

    std::vector<int64_t> host_pred(N);
    pred.copy_to_host(host_pred.data(), host_pred.size() * sizeof(int64_t));

    int errors = 0;
    for (int64_t i = 0; i < N; ++i) {
        if (host_pred[i] != ids[i]) {
            std::cerr << "mismatch row=" << i << " expect=" << ids[i] << " got=" << host_pred[i] << std::endl;
            ++errors;
        }
    }

    if (errors) {
        std::cerr << "token forward smoke failed, mismatches=" << errors << std::endl;
        return 7;
    }

    std::cout << "token forward smoke ok ids=";
    for (auto id : ids) std::cout << id << ',';
    std::cout << std::endl;
    return 0;
}
