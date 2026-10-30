// Gyro DX parity dump: writes selected graph tensors and final logits as raw float32 files.
// Usage: llama-gyro-dump -m MODEL --gyro-out DIR --gyro-tokens "1,2,3" [--gyro-step] [--gyro-filter a,b,c] [--gyro-gen N]
// Every file starts with four little-endian int64 dims (ne0..ne3), followed by ne0*ne1*ne2*ne3 float32 values.
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <sstream>
#include <map>

struct dump_state {
    std::string dir;
    std::vector<std::string> filters;
    int step = 0;
    std::vector<uint8_t> raw;
    std::vector<float> f32;
    std::map<std::string, int> seen; // repeated names in one step get a #N suffix
};

static bool wanted(const dump_state & s, const char * name) {
    if (s.filters.empty()) return true;
    for (const auto & f : s.filters) {
        if (f.size() && f.back() == '*') {
            if (strncmp(name, f.c_str(), f.size() - 1) == 0) return true;
        } else if (f == name) {
            return true;
        }
    }
    return false;
}

static float to_f32(ggml_type type, const uint8_t * p) {
    switch (type) {
        case GGML_TYPE_F32:  return *(const float *) p;
        case GGML_TYPE_F16:  return ggml_fp16_to_fp32(*(const ggml_fp16_t *) p);
        case GGML_TYPE_BF16: return ggml_bf16_to_fp32(*(const ggml_bf16_t *) p);
        case GGML_TYPE_I32:  return (float) *(const int32_t *) p;
        default: return NAN;
    }
}

static bool dump_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * s = (dump_state *) user_data;
    if (ask) return wanted(*s, t->name);
    if (!wanted(*s, t->name)) return true;
    if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_BF16 && t->type != GGML_TYPE_I32) return true;

    const size_t nbytes = ggml_nbytes(t);
    s->raw.resize(nbytes);
    ggml_backend_tensor_get(t, s->raw.data(), 0, nbytes);

    const int64_t n = ggml_nelements(t);
    s->f32.resize(n);
    const size_t es = ggml_type_size(t->type);
    int64_t k = 0;
    for (int64_t i3 = 0; i3 < t->ne[3]; ++i3)
    for (int64_t i2 = 0; i2 < t->ne[2]; ++i2)
    for (int64_t i1 = 0; i1 < t->ne[1]; ++i1)
    for (int64_t i0 = 0; i0 < t->ne[0]; ++i0) {
        const size_t off = i0*t->nb[0] + i1*t->nb[1] + i2*t->nb[2] + i3*t->nb[3];
        s->f32[k++] = (off + es <= nbytes) ? to_f32(t->type, s->raw.data() + off) : NAN;
    }

    std::string safe = t->name;
    const int occurrence = s->seen[safe]++;
    if (occurrence > 0) safe += "#" + std::to_string(occurrence);
    for (auto & c : safe) if (c == '/' || c == '\\' || c == ':' || c == ' ') c = '_';
    char path[1024];
    snprintf(path, sizeof(path), "%s/s%03d.%s.bin", s->dir.c_str(), s->step, safe.c_str());
    FILE * f = fopen(path, "wb");
    if (!f) return true;
    int64_t ne[4] = { t->ne[0], t->ne[1], t->ne[2], t->ne[3] };
    fwrite(ne, sizeof(ne), 1, f);
    fwrite(s->f32.data(), sizeof(float), n, f);
    fclose(f);
    return true;
}

static std::vector<llama_token> parse_tokens(const std::string & text) {
    std::vector<llama_token> out;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) if (!item.empty()) out.push_back((llama_token) std::stol(item));
    return out;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    // strip our own flags before common parsing
    dump_state state;
    std::string tokens_text;
    bool step_mode = false;
    int n_gen = 0;
    std::vector<char *> rest;
    for (int i = 0; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--gyro-out" && i + 1 < argc) { state.dir = argv[++i]; continue; }
        if (a == "--gyro-tokens" && i + 1 < argc) { tokens_text = argv[++i]; continue; }
        if (a == "--gyro-step") { step_mode = true; continue; }
        if (a == "--gyro-gen" && i + 1 < argc) { n_gen = atoi(argv[++i]); continue; }
        if (a == "--gyro-filter" && i + 1 < argc) {
            std::stringstream ss(argv[++i]); std::string item;
            while (std::getline(ss, item, ',')) if (!item.empty()) state.filters.push_back(item);
            continue;
        }
        rest.push_back(argv[i]);
    }

    common_params params;
    common_init();
    if (!common_params_parse((int) rest.size(), rest.data(), params, LLAMA_EXAMPLE_COMMON)) return 1;
    if (state.dir.empty()) { fprintf(stderr, "--gyro-out is required\n"); return 1; }

    llama_backend_init();
    params.cb_eval = dump_cb;
    params.cb_eval_user_data = &state;
    params.warmup = false;

    auto init = common_init_from_params(params);
    auto * model = init->model();
    auto * ctx = init->context();
    if (!model || !ctx) return 1;
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::vector<llama_token> tokens = tokens_text.empty()
        ? common_tokenize(ctx, params.prompt, llama_vocab_get_add_bos(vocab), true)
        : parse_tokens(tokens_text);
    if (tokens.empty()) return 1;

    {
        std::string p = state.dir + "/tokens.txt";
        FILE * f = fopen(p.c_str(), "w");
        for (auto t : tokens) fprintf(f, "%d\n", t);
        fclose(f);
    }

    const int n_vocab = llama_vocab_n_tokens(vocab);
    auto write_logits = [&](int step, const float * logits) {
        char path[1024];
        snprintf(path, sizeof(path), "%s/s%03d.logits.bin", state.dir.c_str(), step);
        FILE * f = fopen(path, "wb");
        int64_t ne[4] = { n_vocab, 1, 1, 1 };
        fwrite(ne, sizeof(ne), 1, f);
        fwrite(logits, sizeof(float), n_vocab, f);
        fclose(f);
    };

    if (step_mode) {
        for (size_t i = 0; i < tokens.size(); ++i) {
            state.step = (int) i;
            state.seen.clear();
            llama_token t = tokens[i];
            if (llama_decode(ctx, llama_batch_get_one(&t, 1))) return 1;
            write_logits((int) i, llama_get_logits_ith(ctx, -1));
        }
    } else {
        state.step = 0;
        if (llama_decode(ctx, llama_batch_get_one(tokens.data(), (int) tokens.size()))) return 1;
        write_logits(0, llama_get_logits_ith(ctx, -1));
    }

    // greedy continuation for whole-model token-trace parity
    if (n_gen > 0) {
        std::string p = state.dir + "/greedy.txt";
        FILE * f = fopen(p.c_str(), "w");
        state.filters = { "__none__" };
        for (int g = 0; g < n_gen; ++g) {
            const float * logits = llama_get_logits_ith(ctx, -1);
            int best = 0;
            for (int v = 1; v < n_vocab; ++v) if (logits[v] > logits[best]) best = v;
            fprintf(f, "%d\n", best);
            fflush(f);
            if (llama_vocab_is_eog(vocab, best)) break;
            llama_token t = best;
            if (llama_decode(ctx, llama_batch_get_one(&t, 1))) return 1;
        }
        fclose(f);
    }

    llama_perf_context_print(ctx);
    llama_backend_free();
    return 0;
}
