#include "include/gyro_native.h"
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

extern "C" {
__declspec(dllexport) extern const uint32_t D3D12SDKVersion = 722;
__declspec(dllexport) extern const char *D3D12SDKPath = ".\\D3D12\\";
}

static bool ok(gd_result result, const char *operation) {
    if (result >= 0) return true;
    std::fprintf(stderr, "%s failed: HRESULT 0x%08x\n", operation, uint32_t(result));
    return false;
}

static bool number(const char *text, uint64_t &value) {
    if (!text || !*text || *text == '-') return false;
    char *end = nullptr;
    errno = 0;
    value = std::strtoull(text, &end, 0);
    return errno == 0 && end && *end == 0;
}

static std::filesystem::path utf8_path(const char *text) {
    std::u8string path;
    for (; *text; ++text) path.push_back(static_cast<char8_t>(*text));
    return std::filesystem::path(path);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: gyro-native-smoke FILE [--offset N] [--bytes N] [--adapter N] [--shader FILE] [--debug]\n");
        return 2;
    }
    uint64_t offset = 0, bytes = 1024 * 1024;
    const char *shader_path = nullptr;
    gd_options options{};
    for (int i = 2; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--debug")) { options.enable_debug = 1; continue; }
        if (i + 1 == argc) return 2;
        const char *option = argv[i++];
        if (!std::strcmp(option, "--shader")) { shader_path = argv[i]; continue; }
        uint64_t value = 0;
        if (!number(argv[i], value)) return 2;
        if (!std::strcmp(option, "--offset")) offset = value;
        else if (!std::strcmp(option, "--bytes")) bytes = value;
        else if (!std::strcmp(option, "--adapter") && value <= UINT32_MAX) options.adapter_index = uint32_t(value);
        else return 2;
    }
    if (!bytes || bytes > 32 * 1024 * 1024) return 2;
    options.require_linalg = shader_path != nullptr;
    gd_context *context = nullptr;
    if (!ok(gd_create(&options, &context), "Create context")) return 1;
    gd_caps caps{};
    gd_get_caps(context, &caps);
    std::printf("Adapter: %s\nSM: 0x%x\nLinAlg tier: 0x%x\nWave GEMM flags: 0x%x\nVRAM budget: %llu bytes\n",
        caps.adapter_name, caps.shader_model, caps.linalg_tier, caps.wave16_fp16_fp32_flags,
        static_cast<unsigned long long>(caps.local_budget_bytes));
    int exit_code = 1;
    try {
        gd_file *file = nullptr;
        gd_buffer *buffer = nullptr;
        if (!ok(gd_open_file(context, argv[1], &file), "Open source file")) throw 1;
        uint64_t file_bytes = 0;
        gd_file_size(file, &file_bytes);
        if (offset >= file_bytes) { std::fprintf(stderr, "Offset exceeds file size.\n"); throw 1; }
        if (bytes > file_bytes - offset) bytes = file_bytes - offset;
        if (!ok(gd_create_buffer(context, bytes, &buffer), "Create destination buffer")) throw 1;
        std::vector<uint8_t> expected(size_t(bytes), 0);
        std::vector<uint8_t> actual(size_t(bytes), 0);
        std::ifstream input(utf8_path(argv[1]), std::ios::binary);
        if (offset > uint64_t(std::numeric_limits<std::streamoff>::max())) throw 1;
        input.seekg(static_cast<std::streamoff>(offset));
        input.read(reinterpret_cast<char*>(expected.data()), std::streamsize(bytes));
        if (uint64_t(input.gcount()) != bytes) throw 1;
        gd_range ranges[2]{};
        uint32_t first = bytes > 1 ? uint32_t(bytes / 2) : uint32_t(bytes);
        ranges[0] = {offset, 0, first, 0};
        ranges[1] = {offset + first, first, uint32_t(bytes) - first, 0};
        uint32_t range_count = bytes > 1 ? 2 : 1;
        for (uint32_t pass = 0; pass < 2; ++pass) {
            uint64_t ticket = 0;
            if (!ok(gd_read_ranges(context, file, buffer, ranges, range_count, &ticket), "Enqueue file ranges")) throw 1;
            if (!ok(gd_readback(context, buffer, 0, uint32_t(bytes), actual.data(), 30000), "Read GPU buffer")) throw 1;
            if (actual != expected) { std::fprintf(stderr, "GPU bytes differ from source bytes.\n"); throw 1; }
        }
        std::printf("DirectStorage equality: PASS (%llu bytes, two submissions)\n", static_cast<unsigned long long>(bytes));
        if (shader_path) {
            std::ifstream shader(utf8_path(shader_path), std::ios::binary | std::ios::ate);
            auto size = shader.tellg();
            if (size <= 0 || size > 16 * 1024 * 1024) throw 1;
            std::vector<uint8_t> dxil(static_cast<size_t>(size));
            shader.seekg(0);
            shader.read(reinterpret_cast<char*>(dxil.data()), size);
            if (!shader) throw 1;
            float error = 0;
            if (!ok(gd_validate_linalg(context, dxil.data(), dxil.size(), &error, 30000), "LinAlg validation")) throw 1;
            std::printf("LinAlg FP16/FP32 GEMM: PASS (maximum absolute error %.9g)\n", error);
        }
        exit_code = 0;
    } catch (...) {
        if (exit_code) std::fprintf(stderr, "Native smoke test failed.\n");
    }
    if (!ok(gd_destroy(context, 30000), "Destroy context")) return 1;
    return exit_code;
}
