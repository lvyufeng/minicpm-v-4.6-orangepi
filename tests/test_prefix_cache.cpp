// Compare three regimes:
// 1) fresh prefill from scratch on full history
// 2) cached prefix + suffix replay on the same conversation_id
// Verify that later turns don't scale with total history length.
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <sys/wait.h>

using clk = std::chrono::steady_clock;
static double sec(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

static void write_bundle(const std::string& path,
                         const std::vector<int64_t>& input_ids,
                         const std::string& conv_id) {
    std::ofstream f(path);
    f << "{\n"
      << "  \"version\": 1,\n"
      << "  \"conversation_id\": \"" << conv_id << "\",\n"
      << "  \"seq_len\": " << input_ids.size() << ",\n"
      << "  \"hidden_size\": 1024,\n"
      << "  \"max_seq_len\": 512,\n"
      << "  \"max_new_tokens\": 8,\n"
      << "  \"vocab_size\": 248094,\n"
      << "  \"eos_token_ids\": [248044, 248046],\n"
      << "  \"layer_types\": ["
      << "\"linear_attention\",\"linear_attention\",\"linear_attention\",\"full_attention\","
      << "\"linear_attention\",\"linear_attention\",\"linear_attention\",\"full_attention\","
      << "\"linear_attention\",\"linear_attention\",\"linear_attention\",\"full_attention\","
      << "\"linear_attention\",\"linear_attention\",\"linear_attention\",\"full_attention\","
      << "\"linear_attention\",\"linear_attention\",\"linear_attention\",\"full_attention\","
      << "\"linear_attention\",\"linear_attention\",\"linear_attention\",\"full_attention\"],\n"
      << "  \"embeddings_dtype\": \"float16\",\n"
      << "  \"embeddings_endianness\": \"little\",\n"
      << "  \"embeddings_path\": \"\",\n"
      << "  \"weights_path\": \"/mnt/data/minicpm/MiniCPM-V-4.6/model.safetensors\",\n"
      << "  \"input_ids\": [";
    for (size_t i = 0; i < input_ids.size(); ++i) {
        if (i) f << ", ";
        f << input_ids[i];
    }
    f << "]\n}";
}

static double run_once(const std::string& bundle_path, const std::string& conv_id) {
    int inpipe[2], outpipe[2];
    pipe(inpipe); pipe(outpipe);
    pid_t pid = fork();
    if (pid == 0) {
        dup2(inpipe[0], STDIN_FILENO);
        dup2(outpipe[1], STDOUT_FILENO);
        close(inpipe[1]); close(outpipe[0]);
        setenv("ASCEND_CUSTOM_OPP_PATH", "/mnt/data/minicpm/custom_opp_install/vendors/customize", 1);
        execl("/mnt/data/minicpm/build/minicpmv_hybrid_decode", "minicpmv_hybrid_decode",
              "--server", "--weights", "/mnt/data/minicpm/MiniCPM-V-4.6/model.safetensors", (char*)nullptr);
        _exit(127);
    }
    close(inpipe[0]); close(outpipe[1]);

    // wait for #server_ready
    std::string line;
    char ch;
    while (read(outpipe[0], &ch, 1) == 1) {
        if (ch == '\n') {
            if (line.find("# server_ready") != std::string::npos) break;
            line.clear();
        } else line.push_back(ch);
    }

    auto t0 = clk::now();
    std::string req = bundle_path + "\n";
    write(inpipe[1], req.data(), req.size());
    line.clear();
    while (read(outpipe[0], &ch, 1) == 1) {
        if (ch == '\n') {
            if (line.rfind("# done", 0) == 0) break;
            line.clear();
        } else line.push_back(ch);
    }
    auto t1 = clk::now();

    std::string quit = "quit\n";
    write(inpipe[1], quit.data(), quit.size());
    close(inpipe[1]); close(outpipe[0]);
    waitpid(pid, nullptr, 0);
    return sec(t0, t1);
}

int main() {
    // Synthetic conversation growth: same prefix, longer total history.
    std::vector<int64_t> t1 = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int64_t> t2 = t1; t2.insert(t2.end(), {9,10,11,12,13,14,15,16,17,18,19,20});
    std::vector<int64_t> t3 = t2; t3.insert(t3.end(), {21,22,23,24,25,26,27,28,29,30,31,32});
    std::string p1 = "/tmp/prefix_cache_1.json";
    std::string p2 = "/tmp/prefix_cache_2.json";
    std::string p3 = "/tmp/prefix_cache_3.json";
    write_bundle(p1, t1, "conv-test");
    write_bundle(p2, t2, "conv-test");
    write_bundle(p3, t3, "conv-test");

    // For now use one fresh server per request just to verify JSON shape is accepted.
    // The stronger timing-based test will come from the Python smoke below.
    double a = run_once(p1, "conv-test");
    double b = run_once(p2, "conv-test");
    double c = run_once(p3, "conv-test");
    std::printf("fresh_server_request_times: %.2fs %.2fs %.2fs\n", a, b, c);
    return 0;
}
