/*
 * qwen_asr_onnx.c - Qwen3-ASR ONNX Runtime inference pipeline
 *
 * Pipeline: audio → mel spectrogram → encoder ONNX → prompt embedding →
 *           decoder prefill ONNX → decode loop ONNX → token decode → text
 */

#include "qwen_asr_onnx.h"
#include "qwen_asr_audio.h"
#include "qwen_asr_tokenizer.h"
#include "qwen_asr.h"  /* for constants */
#include "ort/onnxruntime_c_api.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <stdarg.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* Timing results — written by transcribe, readable via qwen_onnx_get_last_timing() */
static double last_mel_ms, last_enc_ms, last_prefill_ms, last_decode_ms, last_total_ms;
static int last_n_tokens;

void qwen_onnx_get_last_timing(double *mel, double *enc, double *prefill,
                                double *decode, double *total, int *n_tokens) {
    if (mel) *mel = last_mel_ms;
    if (enc) *enc = last_enc_ms;
    if (prefill) *prefill = last_prefill_ms;
    if (decode) *decode = last_decode_ms;
    if (total) *total = last_total_ms;
    if (n_tokens) *n_tokens = last_n_tokens;
}

int qwen_onnx_verbose = 0;

/* File-based logger for device diagnostics (stderr not accessible in E2E tests) */
static FILE *log_file = NULL;

void qwen_onnx_set_log_file(const char *path) {
    if (log_file) fclose(log_file);
    log_file = fopen(path, "w");
    if (log_file) {
        fprintf(log_file, "[qwen_onnx] log file opened\n");
        fflush(log_file);
    }
}

static void log_msg(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);

    if (log_file) {
        va_list args2;
        va_start(args2, fmt);
        vfprintf(log_file, fmt, args2);
        va_end(args2);
        fflush(log_file);
    }
}

/* Last error buffer for diagnostics from Swift */
static char last_error_buf[1024] = {0};

const char *qwen_onnx_get_last_error(void) {
    return last_error_buf;
}

static void set_last_error(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(last_error_buf, sizeof(last_error_buf), fmt, args);
    va_end(args);
    fprintf(stderr, "qwen_onnx ERROR: %s\n", last_error_buf);
}

/* ======================================================================== */
/* Constants                                                                 */
/* ======================================================================== */

#define MAX_DEC_LAYERS 28
#define MAX_NEW_TOKENS 1024
#define CHUNK_SIZE     100   /* mel frames per encoder chunk */

/* Prompt prefix: <|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|> */
static const int PROMPT_PREFIX[] = {151644, 8948, 198, 151645, 198, 151644, 872, 198, 151669};
static const int N_PREFIX = 9;

/* Prompt suffix: <|audio_end|><|im_end|>\n<|im_start|>assistant\n */
static const int PROMPT_SUFFIX[] = {151670, 151645, 198, 151644, 77091, 198};
static const int N_SUFFIX = 6;

/* EOS tokens */
static const int EOS_TOKENS[] = {151643, 151645};
static const int N_EOS = 2;

/* ======================================================================== */
/* ONNX Context                                                              */
/* ======================================================================== */

struct qwen_onnx_ctx {
    const OrtApi    *api;
    OrtEnv          *env;
    OrtSession      *encoder;
    OrtSession      *prefill;     /* NULL — loaded on-demand in transcribe() */
    OrtSession      *decode;      /* NULL — loaded on-demand in transcribe() */
    OrtMemoryInfo   *mem_info;

    /* Token embeddings [vocab_size, hidden_dim] stored as fp16, memory-mapped.
     * mmap avoids the 297 MB malloc — the OS only pages in accessed portions.
     * During transcription, ~200 tokens × 1024 dim × 2 bytes = ~400 KB accessed. */
    uint16_t        *embed_tokens_fp16;  /* mmap'd pointer into .npy file */
    size_t           embed_mmap_size;    /* total mmap'd size for munmap */
    void            *embed_mmap_base;    /* base pointer for munmap (may differ from embed_tokens_fp16) */
    int              vocab_size;
    int              hidden_dim;

    /* Decoder layer count (0 until first transcribe determines it) */
    int              n_layers;

    /* Tokenizer */
    qwen_tokenizer_t *tokenizer;

    /* Stored for on-demand session loading */
    char            *model_dir;
    int              enc_threads;
    int              dec_threads;
    int              keep_sessions;
};

/* ======================================================================== */
/* Helpers                                                                   */
/* ======================================================================== */

static int is_eos(int token) {
    for (int i = 0; i < N_EOS; i++)
        if (token == EOS_TOKENS[i]) return 1;
    return 0;
}

static int argmax_f32(const float *data, int n) {
    int best = 0;
    float best_val = data[0];
    for (int i = 1; i < n; i++) {
        if (data[i] > best_val) { best_val = data[i]; best = i; }
    }
    return best;
}

/* Convert float16 (IEEE 754 half-precision) to float32 */
static float fp16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) { f = sign; }
        else {
            exp = 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            f = sign | ((uint32_t)(exp + 127 - 15) << 23) | ((uint32_t)mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000 | ((uint32_t)mant << 13);
    } else {
        f = sign | ((uint32_t)(exp + 127 - 15) << 23) | ((uint32_t)mant << 13);
    }
    float result;
    memcpy(&result, &f, 4);
    return result;
}

/* ======================================================================== */
/* NPY File Loader                                                           */
/* ======================================================================== */

/* Load a .npy file containing a 2D float32 or float16 array.
 * Always returns float32 data. Sets shape[0] and shape[1]. */
static float *load_npy(const char *path, int *rows, int *cols) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "qwen_onnx: cannot open %s\n", path); return NULL; }

    /* Read magic + version */
    uint8_t header[10];
    if (fread(header, 1, 10, f) != 10) { fclose(f); return NULL; }
    if (memcmp(header, "\x93NUMPY", 6) != 0) { fclose(f); return NULL; }

    int major = header[6];
    uint32_t hdr_len;
    if (major == 1) {
        hdr_len = (uint32_t)header[8] | ((uint32_t)header[9] << 8);
    } else {
        /* v2: 4-byte header length at offset 8 */
        uint8_t extra[2];
        if (fread(extra, 1, 2, f) != 2) { fclose(f); return NULL; }
        hdr_len = (uint32_t)header[8] | ((uint32_t)header[9] << 8) |
                  ((uint32_t)extra[0] << 16) | ((uint32_t)extra[1] << 24);
    }

    char *hdr_str = (char *)malloc(hdr_len + 1);
    if (fread(hdr_str, 1, hdr_len, f) != hdr_len) { free(hdr_str); fclose(f); return NULL; }
    hdr_str[hdr_len] = '\0';

    /* Parse dtype: '<f4' (float32) or '<f2' (float16) */
    int is_fp16 = (strstr(hdr_str, "'<f2'") != NULL || strstr(hdr_str, "\"<f2\"") != NULL);

    /* Parse shape: (rows, cols) */
    char *sp = strstr(hdr_str, "shape");
    if (!sp) { free(hdr_str); fclose(f); return NULL; }
    char *lp = strchr(sp, '(');
    if (!lp) { free(hdr_str); fclose(f); return NULL; }
    int r = 0, c = 0;
    sscanf(lp + 1, "%d, %d", &r, &c);
    free(hdr_str);

    if (r <= 0 || c <= 0) { fclose(f); return NULL; }

    /* Read raw data */
    size_t n_elements = (size_t)r * c;
    float *data = (float *)malloc(n_elements * sizeof(float));

    if (is_fp16) {
        uint16_t *buf = (uint16_t *)malloc(n_elements * sizeof(uint16_t));
        if (fread(buf, sizeof(uint16_t), n_elements, f) != n_elements) {
            free(buf); free(data); fclose(f); return NULL;
        }
        for (size_t i = 0; i < n_elements; i++)
            data[i] = fp16_to_f32(buf[i]);
        free(buf);
    } else {
        if (fread(data, sizeof(float), n_elements, f) != n_elements) {
            free(data); fclose(f); return NULL;
        }
    }

    fclose(f);
    *rows = r;
    *cols = c;
    return data;
}

/* Load a .npy file containing a 2D float16 array. Returns raw uint16_t data.
 * Falls back to loading float32 .npy and converting to fp16 for compatibility. */
static uint16_t *load_npy_fp16(const char *path, int *rows, int *cols) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "qwen_onnx: cannot open %s\n", path); return NULL; }

    uint8_t header[10];
    if (fread(header, 1, 10, f) != 10) { fclose(f); return NULL; }
    if (memcmp(header, "\x93NUMPY", 6) != 0) { fclose(f); return NULL; }

    int major = header[6];
    uint32_t hdr_len;
    if (major == 1) {
        hdr_len = (uint32_t)header[8] | ((uint32_t)header[9] << 8);
    } else {
        uint8_t extra[2];
        if (fread(extra, 1, 2, f) != 2) { fclose(f); return NULL; }
        hdr_len = (uint32_t)header[8] | ((uint32_t)header[9] << 8) |
                  ((uint32_t)extra[0] << 16) | ((uint32_t)extra[1] << 24);
    }

    char *hdr_str = (char *)malloc(hdr_len + 1);
    if (fread(hdr_str, 1, hdr_len, f) != hdr_len) { free(hdr_str); fclose(f); return NULL; }
    hdr_str[hdr_len] = '\0';

    int is_fp16 = (strstr(hdr_str, "'<f2'") != NULL || strstr(hdr_str, "\"<f2\"") != NULL);

    char *sp = strstr(hdr_str, "shape");
    if (!sp) { free(hdr_str); fclose(f); return NULL; }
    char *lp = strchr(sp, '(');
    if (!lp) { free(hdr_str); fclose(f); return NULL; }
    int r = 0, c = 0;
    sscanf(lp + 1, "%d, %d", &r, &c);
    free(hdr_str);

    if (r <= 0 || c <= 0) { fclose(f); return NULL; }

    size_t n_elements = (size_t)r * c;
    uint16_t *data = (uint16_t *)malloc(n_elements * sizeof(uint16_t));

    if (is_fp16) {
        if (fread(data, sizeof(uint16_t), n_elements, f) != n_elements) {
            free(data); fclose(f); return NULL;
        }
    } else {
        /* fp32 file — read and convert to fp16 (lossy but saves memory) */
        float *tmp = (float *)malloc(n_elements * sizeof(float));
        if (fread(tmp, sizeof(float), n_elements, f) != n_elements) {
            free(tmp); free(data); fclose(f); return NULL;
        }
        /* Simple fp32→fp16 conversion (just for fallback; we expect fp16 input) */
        for (size_t i = 0; i < n_elements; i++) {
            /* Use fp16_to_f32 in reverse is complex; store as fp32 bits truncated */
            /* For a proper fallback, just load as fp32 and have caller handle it */
            uint32_t bits;
            memcpy(&bits, &tmp[i], 4);
            uint32_t sign = (bits >> 16) & 0x8000;
            int32_t exp_val = ((bits >> 23) & 0xFF) - 127 + 15;
            uint32_t mant = (bits >> 13) & 0x3FF;
            if (exp_val <= 0) data[i] = (uint16_t)sign;
            else if (exp_val >= 31) data[i] = (uint16_t)(sign | 0x7C00);
            else data[i] = (uint16_t)(sign | ((uint32_t)exp_val << 10) | mant);
        }
        free(tmp);
    }

    fclose(f);
    *rows = r;
    *cols = c;
    return data;
}

/* Memory-map a .npy file containing a 2D fp16 array.
 * Returns a pointer to the raw fp16 data within the mmap'd region.
 * Caller must munmap(mmap_base, mmap_size) when done. */
static uint16_t *mmap_npy_fp16(const char *path, int *rows, int *cols,
                                 void **mmap_base, size_t *mmap_size) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { log_msg("qwen_onnx: cannot open %s for mmap\n", path); return NULL; }

    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return NULL; }
    size_t file_size = (size_t)st.st_size;

    void *mapped = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mapped == MAP_FAILED) {
        log_msg("qwen_onnx: mmap failed for %s (size=%zu)\n", path, file_size);
        return NULL;
    }

    /* Parse .npy header to find data offset */
    const uint8_t *p = (const uint8_t *)mapped;
    if (memcmp(p, "\x93NUMPY", 6) != 0) { munmap(mapped, file_size); return NULL; }

    int major = p[6];
    uint32_t hdr_len;
    size_t data_offset;
    if (major == 1) {
        hdr_len = (uint32_t)p[8] | ((uint32_t)p[9] << 8);
        data_offset = 10 + hdr_len;
    } else {
        hdr_len = (uint32_t)p[8] | ((uint32_t)p[9] << 8) |
                  ((uint32_t)p[10] << 16) | ((uint32_t)p[11] << 24);
        data_offset = 12 + hdr_len;
    }

    /* Parse header string for dtype and shape */
    char *hdr_str = (char *)malloc(hdr_len + 1);
    memcpy(hdr_str, p + (data_offset - hdr_len), hdr_len);
    hdr_str[hdr_len] = '\0';

    int is_fp16 = (strstr(hdr_str, "'<f2'") != NULL || strstr(hdr_str, "\"<f2\"") != NULL);
    if (!is_fp16) {
        log_msg("qwen_onnx: mmap_npy_fp16 requires fp16 data, got other dtype\n");
        free(hdr_str);
        munmap(mapped, file_size);
        return NULL;
    }

    char *sp = strstr(hdr_str, "shape");
    if (!sp) { free(hdr_str); munmap(mapped, file_size); return NULL; }
    char *lp = strchr(sp, '(');
    if (!lp) { free(hdr_str); munmap(mapped, file_size); return NULL; }
    int r = 0, c = 0;
    sscanf(lp + 1, "%d, %d", &r, &c);
    free(hdr_str);

    if (r <= 0 || c <= 0) { munmap(mapped, file_size); return NULL; }

    *rows = r;
    *cols = c;
    *mmap_base = mapped;
    *mmap_size = file_size;

    log_msg("qwen_onnx: mmap'd %s (%zu bytes, data at offset %zu, %dx%d fp16)\n",
            path, file_size, data_offset, r, c);

    return (uint16_t *)((uint8_t *)mapped + data_offset);
}

/* Embed a single token: convert fp16 embedding row to fp32 output buffer */
static void embed_token_fp16(const uint16_t *embed_fp16, int token_id,
                              int hidden_dim, float *out) {
    const uint16_t *row = embed_fp16 + (size_t)token_id * hidden_dim;
    for (int i = 0; i < hidden_dim; i++)
        out[i] = fp16_to_f32(row[i]);
}

/* ======================================================================== */
/* ORT Helper Macros                                                         */
/* ======================================================================== */

#define ORT_CHECK(expr) do { \
    OrtStatus *_s = (expr); \
    if (_s) { \
        const char *_m = ctx->api->GetErrorMessage(_s); \
        log_msg("qwen_onnx ORT error: %s\n", _m); \
        ctx->api->ReleaseStatus(_s); \
        goto cleanup; \
    } \
} while(0)

#define ORT_CHECK_LOAD(expr) do { \
    OrtStatus *_s = (expr); \
    if (_s) { \
        const char *_m = api->GetErrorMessage(_s); \
        set_last_error("ORT load error at %s:%d: %s", __FILE__, __LINE__, _m); \
        api->ReleaseStatus(_s); \
        qwen_onnx_free(ctx); \
        return NULL; \
    } \
} while(0)

/* ======================================================================== */
/* Load / Free                                                               */
/* ======================================================================== */

static char *path_join(const char *dir, const char *file) {
    size_t dlen = strlen(dir);
    size_t flen = strlen(file);
    char *p = (char *)malloc(dlen + flen + 2);
    memcpy(p, dir, dlen);
    if (dlen > 0 && dir[dlen-1] != '/') p[dlen++] = '/';
    memcpy(p + dlen, file, flen + 1);
    return p;
}

static const char *find_model(const char *dir, const char *base_name) {
    /* Try INT8 first, then full precision */
    static char buf[1024];
    /* Build int8 name: "encoder.onnx" → "encoder.int8.onnx" */
    const char *dot = strrchr(base_name, '.');
    if (dot) {
        size_t prefix_len = dot - base_name;
        snprintf(buf, sizeof(buf), "%s/%.*s.int8%s", dir, (int)prefix_len, base_name, dot);
        FILE *f = fopen(buf, "rb");
        if (f) { fclose(f); return buf; }
    }
    snprintf(buf, sizeof(buf), "%s/%s", dir, base_name);
    return buf;
}

static int create_session_with_fallback(
    const OrtApi *api,
    OrtEnv *env,
    const char *model_path,
    int intra_threads,
    const GraphOptimizationLevel *levels,
    int n_levels,
    OrtSession **out_session
) {
    for (int i = 0; i < n_levels; i++) {
        OrtSessionOptions *opts = NULL;
        OrtStatus *s = api->CreateSessionOptions(&opts);
        if (s) {
            const char *m = api->GetErrorMessage(s);
            log_msg("qwen_onnx ORT error: %s\n", m);
            api->ReleaseStatus(s);
            return 0;
        }

        s = api->SetSessionGraphOptimizationLevel(opts, levels[i]);
        if (!s) s = api->SetIntraOpNumThreads(opts, intra_threads);
        if (!s) s = api->SetInterOpNumThreads(opts, 1);
        if (!s) s = api->DisableMemPattern(opts);
        if (s) {
            const char *m = api->GetErrorMessage(s);
            log_msg("qwen_onnx ORT error: %s\n", m);
            api->ReleaseStatus(s);
            api->ReleaseSessionOptions(opts);
            return 0;
        }

        log_msg("qwen_onnx: CreateSession (opt=%d, threads=%d) %s ...\n",
                (int)levels[i], intra_threads, model_path);
        s = api->CreateSession(env, model_path, opts, out_session);
        api->ReleaseSessionOptions(opts);
        if (!s) {
            log_msg("qwen_onnx: loaded OK (opt=%d)\n", (int)levels[i]);
            return 1;
        }

        const char *m = api->GetErrorMessage(s);
        log_msg("qwen_onnx: CreateSession failed (opt=%d) for %s: %s\n",
                (int)levels[i], model_path, m);
        set_last_error("CreateSession failed (opt=%d) for %s: %s", (int)levels[i], model_path, m);
        api->ReleaseStatus(s);
    }

    return 0;
}

qwen_onnx_ctx_t *qwen_onnx_load(const char *model_dir) {
    last_error_buf[0] = '\0';
    log_msg("qwen_onnx_load: model_dir=%s\n", model_dir ? model_dir : "(null)");

    qwen_onnx_ctx_t *ctx = (qwen_onnx_ctx_t *)calloc(1, sizeof(qwen_onnx_ctx_t));
    if (!ctx) { set_last_error("calloc failed"); return NULL; }

    const OrtApi *api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    ctx->api = api;

    /* Create ORT environment */
    ORT_CHECK_LOAD(api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "qwen_onnx", &ctx->env));
    /* Use device allocator (not arena) to reduce peak memory on constrained devices */
    ORT_CHECK_LOAD(api->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &ctx->mem_info));

    /* Session options: full parallelism for encoder, lower thread fanout for autoregressive decoder. */
    int n_threads = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (n_threads < 1) n_threads = 4;
    if (n_threads > 8) n_threads = 8;
    /* Keep encoder parallelism higher; decoder remains conservative for memory stability. */
    int enc_threads = (n_threads >= 6) ? 6 : n_threads;
    int dec_threads = (n_threads >= 6) ? 3 : 2;
    log_msg("qwen_onnx: threads enc=%d dec=%d (cores=%d)\n", enc_threads, dec_threads, n_threads);

    /* macOS: keep ORT sessions loaded to avoid re-creating them inside transcribe().
     * iOS/Android: stay conservative to reduce peak RSS on mobile devices. */
    int keep_sessions = 0;
#if defined(__APPLE__) && defined(TARGET_OS_OSX) && TARGET_OS_OSX
    keep_sessions = 1;
#endif

    /* Store model_dir and thread counts for on-demand session loading in transcribe() */
    ctx->model_dir = strdup(model_dir);
    ctx->dec_threads = dec_threads;
    ctx->keep_sessions = keep_sessions;

    /* Load tokenizer first (small, ~2 MB) */
    log_msg("qwen_onnx: loading tokenizer...\n");
    char *vocab_path = path_join(model_dir, "vocab.json");
    ctx->tokenizer = qwen_tokenizer_load(vocab_path);
    free(vocab_path);
    if (!ctx->tokenizer) {
        set_last_error("failed to load tokenizer from vocab.json");
        qwen_onnx_free(ctx);
        return NULL;
    }
    log_msg("qwen_onnx: tokenizer loaded OK\n");

    /* Memory-map token embeddings as fp16 — zero physical memory at init.
     * The OS pages in only the ~400 KB actually accessed during transcription
     * (200 tokens × 1024 dim × 2 bytes) instead of loading all 297 MB. */
    log_msg("qwen_onnx: mmap'ing embeddings...\n");
    char *embed_path = path_join(model_dir, "embed_tokens.fp16.npy");
    {
        FILE *ef = fopen(embed_path, "rb");
        if (!ef) {
            free(embed_path);
            embed_path = path_join(model_dir, "embed_tokens.npy");
        } else {
            fclose(ef);
        }
    }

    int rows, cols;
    ctx->embed_tokens_fp16 = mmap_npy_fp16(embed_path, &rows, &cols,
                                            &ctx->embed_mmap_base, &ctx->embed_mmap_size);
    if (!ctx->embed_tokens_fp16) {
        set_last_error("failed to mmap embed_tokens from %s", embed_path);
        free(embed_path);
        qwen_onnx_free(ctx);
        return NULL;
    }
    free(embed_path);
    ctx->vocab_size = rows;
    ctx->hidden_dim = cols;
    log_msg("qwen_onnx: embeddings %d x %d (fp16 mmap'd, 0 MB physical)\n", rows, cols);

    /* ALL ONNX sessions (encoder, prefill, decode) are loaded on-demand in transcribe().
     * This keeps qwen_onnx_load() nearly zero-cost in memory:
     *   tokenizer: ~2 MB, embeddings: mmap'd (0 physical), no ORT sessions.
     * On 4 GB devices, loading even just the encoder (~200 MB ORT overhead) at
     * init + app overhead can trigger jetsam. */
    ctx->n_layers = 0;
    ctx->enc_threads = enc_threads;

    if (ctx->keep_sessions) {
        /* Eager-load sessions on desktop where memory pressure is much lower.
         * This shifts heavy CreateSession costs out of the inference hot path. */
        const GraphOptimizationLevel enc_levels[] = { ORT_DISABLE_ALL };
        const GraphOptimizationLevel dec_levels[] = { ORT_ENABLE_BASIC, ORT_DISABLE_ALL };
        const int n_enc_levels = (int)(sizeof(enc_levels) / sizeof(enc_levels[0]));
        const int n_dec_levels = (int)(sizeof(dec_levels) / sizeof(dec_levels[0]));
        log_msg("[QwenOnnx] eager: loading encoder/prefill/decode sessions (keep_sessions=1)\n");

        const char *enc_path = find_model(ctx->model_dir, "encoder.onnx");
        if (!create_session_with_fallback(api, ctx->env, enc_path, ctx->enc_threads,
                                          enc_levels, n_enc_levels, &ctx->encoder)) {
            set_last_error("failed to eager-load encoder session");
            qwen_onnx_free(ctx);
            return NULL;
        }

        const char *pf_path = find_model(ctx->model_dir, "decoder_prefill.onnx");
        if (!create_session_with_fallback(api, ctx->env, pf_path, ctx->dec_threads,
                                          dec_levels, n_dec_levels, &ctx->prefill)) {
            set_last_error("failed to eager-load decoder_prefill session");
            qwen_onnx_free(ctx);
            return NULL;
        }

        const char *dc_path = find_model(ctx->model_dir, "decoder_decode.onnx");
        if (!create_session_with_fallback(api, ctx->env, dc_path, ctx->dec_threads,
                                          dec_levels, n_dec_levels, &ctx->decode)) {
            set_last_error("failed to eager-load decoder_decode session");
            qwen_onnx_free(ctx);
            return NULL;
        }

        /* Determine decoder layer count once. */
        size_t n_outputs = 0;
        ORT_CHECK_LOAD(api->SessionGetOutputCount(ctx->prefill, &n_outputs));
        ctx->n_layers = (int)(n_outputs - 1) / 2;
        log_msg("[QwenOnnx] eager: decoder layers=%d\n", ctx->n_layers);
    }

    if (ctx->keep_sessions) {
        log_msg("qwen_onnx: load complete (tokenizer + embeddings mmap + eager ORT sessions)\n");
    } else {
        log_msg("qwen_onnx: load complete (tokenizer + embeddings mmap). "
                "All ONNX sessions loaded on-demand.\n");
    }
    return ctx;
}

void qwen_onnx_free(qwen_onnx_ctx_t *ctx) {
    if (!ctx) return;
    const OrtApi *api = ctx->api;
    if (api) {
        if (ctx->encoder)  api->ReleaseSession(ctx->encoder);
        if (ctx->prefill)  api->ReleaseSession(ctx->prefill);
        if (ctx->decode)   api->ReleaseSession(ctx->decode);
        if (ctx->mem_info) api->ReleaseMemoryInfo(ctx->mem_info);
        if (ctx->env)      api->ReleaseEnv(ctx->env);
    }
    if (ctx->embed_mmap_base) munmap(ctx->embed_mmap_base, ctx->embed_mmap_size);
    if (ctx->tokenizer)    qwen_tokenizer_free(ctx->tokenizer);
    if (ctx->model_dir)    free(ctx->model_dir);
    free(ctx);
}

/* ======================================================================== */
/* Transcription                                                             */
/* ======================================================================== */

char *qwen_onnx_transcribe(qwen_onnx_ctx_t *ctx, const float *samples, int n_samples) {
    if (!ctx || !samples || n_samples <= 0) return NULL;

    const OrtApi *api = ctx->api;
    const int hidden = ctx->hidden_dim;
    char *result = NULL;

    /* Tracking arrays for cleanup */
    OrtValue *enc_input = NULL, *enc_output = NULL;
    OrtValue *prefill_input = NULL;
    OrtValue **prefill_outputs = NULL;
    OrtValue *decode_token_input = NULL, *decode_pos_input = NULL;
    OrtValue **decode_outputs = NULL;
    OrtValue **kv_caches = NULL;  /* 2 * n_layers OrtValue pointers */
    float *decode_token_buf = NULL;
    int64_t decode_pos_val = 0;
    float *input_embeds = NULL;
    int *generated = NULL;

    /* These will be set after on-demand decoder loading */
    int n_layers = 0, n_kv = 0;
    int prefill_n_outputs = 0, decode_n_inputs = 0, decode_n_outputs = 0;

    generated = (int *)malloc(MAX_NEW_TOKENS * sizeof(int));
    if (!generated) goto cleanup;

    double t_start = get_time_ms();

    /* ---- Step 1: Mel spectrogram ---- */
    int n_frames;
    float *mel = qwen_mel_spectrogram(samples, n_samples, &n_frames);
    if (!mel) { log_msg("qwen_onnx: mel spectrogram failed\n"); goto cleanup; }
    double t_mel = get_time_ms();
    log_msg("[QwenOnnx] mel spectrogram: %.1f ms\n", t_mel - t_start);

    /* Pad frames to multiple of CHUNK_SIZE */
    int pad_frames = (CHUNK_SIZE - (n_frames % CHUNK_SIZE)) % CHUNK_SIZE;
    int padded_frames = n_frames + pad_frames;
    if (pad_frames > 0) {
        float *padded = (float *)calloc((size_t)QWEN_MEL_BINS * padded_frames, sizeof(float));
        memcpy(padded, mel, (size_t)QWEN_MEL_BINS * n_frames * sizeof(float));
        free(mel);
        mel = padded;
    }
    log_msg("Mel: %d x %d (padded from %d)\n", QWEN_MEL_BINS, padded_frames, n_frames);

    /* ---- Step 1b: On-demand encoder loading ---- */
    if (!ctx->encoder) {
        const GraphOptimizationLevel enc_levels[] = { ORT_DISABLE_ALL };
        const char *enc_path = find_model(ctx->model_dir, "encoder.onnx");
        log_msg("[QwenOnnx] on-demand: loading encoder...\n");
        double t_enc_load = get_time_ms();
        if (!create_session_with_fallback(api, ctx->env, enc_path, ctx->enc_threads,
                                          enc_levels, 1, &ctx->encoder)) {
            set_last_error("failed to load encoder on-demand");
            goto cleanup;
        }
        log_msg("[QwenOnnx] encoder loaded: %.1f ms\n", get_time_ms() - t_enc_load);
    }

    /* ---- Step 2: Run encoder ---- */
    {
        int64_t mel_shape[] = {1, QWEN_MEL_BINS, padded_frames};
        size_t mel_size = sizeof(float) * QWEN_MEL_BINS * padded_frames;
        ORT_CHECK(api->CreateTensorWithDataAsOrtValue(ctx->mem_info, mel, mel_size,
                  mel_shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &enc_input));

        const char *enc_in_names[]  = {"mel_input"};
        const char *enc_out_names[] = {"audio_embeddings"};
        OrtValue *enc_inputs[]  = {enc_input};
        OrtValue *enc_outputs[] = {NULL};
        ORT_CHECK(api->Run(ctx->encoder, NULL, enc_in_names, (const OrtValue *const *)enc_inputs, 1,
                  enc_out_names, 1, enc_outputs));
        enc_output = enc_outputs[0];
    }
    free(mel); mel = NULL;
    double t_encoder = get_time_ms();
    log_msg("[QwenOnnx] encoder: %.1f ms\n", t_encoder - t_mel);

    /* Get audio embedding shape */
    OrtTensorTypeAndShapeInfo *enc_info;
    ORT_CHECK(api->GetTensorTypeAndShape(enc_output, &enc_info));
    int64_t enc_shape[3];
    ORT_CHECK(api->GetDimensions(enc_info, enc_shape, 3));
    api->ReleaseTensorTypeAndShapeInfo(enc_info);

    int n_audio = (int)enc_shape[1];
    log_msg("Audio embeddings: %d tokens x %d dim\n", n_audio, (int)enc_shape[2]);

    float *audio_embeds;
    ORT_CHECK(api->GetTensorMutableData(enc_output, (void **)&audio_embeds));

    /* Release encoder session to free ~191 MB before loading decoder.
     * The enc_output OrtValue owns its own memory and survives session release. */
    if (ctx->encoder && !ctx->keep_sessions) {
        api->ReleaseSession(ctx->encoder);
        ctx->encoder = NULL;
        log_msg("[QwenOnnx] released encoder session (freeing ~191 MB)\n");
    }

    /* ---- Step 2b: On-demand prefill loading ---- */
    /* Load prefill session on-demand. On 4 GB devices (iPad Pro 3rd gen),
     * loading all sessions at init causes OOM.
     * Strategy: load encoder → run → release → load prefill → run → release → load decode → run → release.
     * Peak: one_decoder(570MB) + KV caches + app overhead ≈ ~1.0 GB. */
    {
        /* The autoregressive decoder is the hot loop: enable basic ORT graph fusions. */
        const GraphOptimizationLevel dec_levels[] = { ORT_ENABLE_BASIC, ORT_DISABLE_ALL };
        int n_dec_levels = (int)(sizeof(dec_levels) / sizeof(dec_levels[0]));

        if (!ctx->prefill) {
            const char *pf_path = find_model(ctx->model_dir, "decoder_prefill.onnx");
            log_msg("[QwenOnnx] on-demand: loading decoder_prefill...\n");
            double t_pf_load = get_time_ms();
            if (!create_session_with_fallback(api, ctx->env, pf_path, ctx->dec_threads,
                                              dec_levels, n_dec_levels, &ctx->prefill)) {
                set_last_error("failed to load decoder_prefill on-demand");
                goto cleanup;
            }
            log_msg("[QwenOnnx] decoder_prefill loaded: %.1f ms\n", get_time_ms() - t_pf_load);
        }

        /* Determine n_layers from prefill output count (first time only) */
        if (ctx->n_layers == 0) {
            size_t n_outputs;
            ORT_CHECK(api->SessionGetOutputCount(ctx->prefill, &n_outputs));
            ctx->n_layers = (int)(n_outputs - 1) / 2;
            log_msg("[QwenOnnx] decoder layers: %d\n", ctx->n_layers);
        }
    }

    /* Set up sizes that depend on n_layers */
    n_layers = ctx->n_layers;
    n_kv = 2 * n_layers;
    prefill_n_outputs = 1 + n_kv;
    decode_n_inputs = 2 + n_kv;
    decode_n_outputs = 1 + n_kv;

    prefill_outputs = (OrtValue **)calloc(prefill_n_outputs, sizeof(OrtValue *));
    decode_outputs  = (OrtValue **)calloc(decode_n_outputs, sizeof(OrtValue *));
    kv_caches       = (OrtValue **)calloc(n_kv, sizeof(OrtValue *));
    if (!prefill_outputs || !decode_outputs || !kv_caches) goto cleanup;

    /* ---- Step 3: Build input embeddings ---- */
    int prompt_len = N_PREFIX + n_audio + N_SUFFIX;
    input_embeds = (float *)malloc((size_t)prompt_len * hidden * sizeof(float));
    if (!input_embeds) goto cleanup;

    /* Embed prefix tokens (fp16→fp32 on-the-fly) */
    for (int i = 0; i < N_PREFIX; i++)
        embed_token_fp16(ctx->embed_tokens_fp16, PROMPT_PREFIX[i], hidden,
                         input_embeds + i * hidden);

    /* Insert audio embeddings */
    memcpy(input_embeds + N_PREFIX * hidden, audio_embeds, (size_t)n_audio * hidden * sizeof(float));

    /* Embed suffix tokens (fp16→fp32 on-the-fly) */
    for (int i = 0; i < N_SUFFIX; i++)
        embed_token_fp16(ctx->embed_tokens_fp16, PROMPT_SUFFIX[i], hidden,
                         input_embeds + (N_PREFIX + n_audio + i) * hidden);

    /* ---- Step 4: Run decoder prefill ---- */
    {
        int64_t emb_shape[] = {1, prompt_len, hidden};
        size_t emb_size = sizeof(float) * prompt_len * hidden;
        ORT_CHECK(api->CreateTensorWithDataAsOrtValue(ctx->mem_info, input_embeds, emb_size,
                  emb_shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &prefill_input));

        /* Build output names */
        char output_name_bufs[1 + MAX_DEC_LAYERS * 2][24];
        const char *pf_out_names[1 + MAX_DEC_LAYERS * 2];
        pf_out_names[0] = "logits";
        for (int i = 0; i < n_layers; i++) {
            snprintf(output_name_bufs[1 + i], 24, "k_cache_%d", i);
            pf_out_names[1 + i] = output_name_bufs[1 + i];
        }
        for (int i = 0; i < n_layers; i++) {
            snprintf(output_name_bufs[1 + n_layers + i], 24, "v_cache_%d", i);
            pf_out_names[1 + n_layers + i] = output_name_bufs[1 + n_layers + i];
        }

        const char *pf_in_names[] = {"input_embeds"};
        OrtValue *pf_inputs[] = {prefill_input};
        ORT_CHECK(api->Run(ctx->prefill, NULL, pf_in_names, (const OrtValue *const *)pf_inputs, 1,
                  pf_out_names, prefill_n_outputs, prefill_outputs));
    }
    double t_prefill = get_time_ms();
    log_msg("[QwenOnnx] prefill: %.1f ms\n", t_prefill - t_encoder);

    /* Extract first token from prefill logits */
    {
        float *logits;
        ORT_CHECK(api->GetTensorMutableData(prefill_outputs[0], (void **)&logits));
        int first_token = argmax_f32(logits, ctx->vocab_size);
        generated[0] = first_token;
        log_msg("First token: %d\n", first_token);

        /* Transfer KV caches from prefill output */
        for (int i = 0; i < n_kv; i++) {
            kv_caches[i] = prefill_outputs[1 + i];
            prefill_outputs[1 + i] = NULL;  /* prevent double-free */
        }
        /* Free prefill logits */
        api->ReleaseValue(prefill_outputs[0]);
        prefill_outputs[0] = NULL;
    }

    /* ---- Step 4b: Release prefill, load decode on-demand ---- */
    /* Release prefill session BEFORE loading decode to minimize peak memory.
     * KV caches from prefill are still held in kv_caches[]. */
    if (ctx->prefill && !ctx->keep_sessions) {
        api->ReleaseSession(ctx->prefill);
        ctx->prefill = NULL;
        log_msg("[QwenOnnx] released decoder_prefill (freeing ~570 MB)\n");
    }

    if (!ctx->decode) {
        /* The autoregressive decoder is the hot loop: enable basic ORT graph fusions. */
        const GraphOptimizationLevel dec_levels[] = { ORT_ENABLE_BASIC, ORT_DISABLE_ALL };
        int n_dec_levels = (int)(sizeof(dec_levels) / sizeof(dec_levels[0]));
        const char *dc_path = find_model(ctx->model_dir, "decoder_decode.onnx");
        log_msg("[QwenOnnx] on-demand: loading decoder_decode...\n");
        double t_dc_load = get_time_ms();
        if (!create_session_with_fallback(api, ctx->env, dc_path, ctx->dec_threads,
                                          dec_levels, n_dec_levels, &ctx->decode)) {
            set_last_error("failed to load decoder_decode on-demand");
            goto cleanup;
        }
        log_msg("[QwenOnnx] decoder_decode loaded: %.1f ms\n", get_time_ms() - t_dc_load);
    }

    /* ---- Step 5: Decode loop ---- */
    {
        int n_generated = 1;
        int token = generated[0];

        /* Pre-build input/output name strings */
        char in_name_bufs[2 + MAX_DEC_LAYERS * 2][24];
        const char *dc_in_names[2 + MAX_DEC_LAYERS * 2];
        dc_in_names[0] = "token_embed";
        dc_in_names[1] = "position";
        for (int i = 0; i < n_layers; i++) {
            snprintf(in_name_bufs[2 + i], 24, "k_cache_in_%d", i);
            dc_in_names[2 + i] = in_name_bufs[2 + i];
        }
        for (int i = 0; i < n_layers; i++) {
            snprintf(in_name_bufs[2 + n_layers + i], 24, "v_cache_in_%d", i);
            dc_in_names[2 + n_layers + i] = in_name_bufs[2 + n_layers + i];
        }

        char out_name_bufs[1 + MAX_DEC_LAYERS * 2][24];
        const char *dc_out_names[1 + MAX_DEC_LAYERS * 2];
        dc_out_names[0] = "logits";
        for (int i = 0; i < n_layers; i++) {
            snprintf(out_name_bufs[1 + i], 24, "k_cache_out_%d", i);
            dc_out_names[1 + i] = out_name_bufs[1 + i];
        }
        for (int i = 0; i < n_layers; i++) {
            snprintf(out_name_bufs[1 + n_layers + i], 24, "v_cache_out_%d", i);
            dc_out_names[1 + n_layers + i] = out_name_bufs[1 + n_layers + i];
        }

        /* Reuse decode input tensors across all steps to avoid per-token OrtValue creation overhead. */
        int64_t tok_shape[] = {1, 1, hidden};
        int64_t pos_shape[] = {1};
        decode_token_buf = (float *)malloc((size_t)hidden * sizeof(float));
        if (!decode_token_buf) goto cleanup;
        ORT_CHECK(api->CreateTensorWithDataAsOrtValue(ctx->mem_info, decode_token_buf,
                  (size_t)hidden * sizeof(float), tok_shape, 3,
                  ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &decode_token_input));
        ORT_CHECK(api->CreateTensorWithDataAsOrtValue(ctx->mem_info, &decode_pos_val,
                  sizeof(int64_t), pos_shape, 1,
                  ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &decode_pos_input));

        for (int step = 0; step < MAX_NEW_TOKENS - 1; step++) {
            if (is_eos(token)) break;

            embed_token_fp16(ctx->embed_tokens_fp16, token, hidden, decode_token_buf);
            decode_pos_val = (int64_t)(prompt_len + step);

            /* Build input array: [token_embed, position, k_0..k_n, v_0..v_n] */
            OrtValue *dc_inputs[2 + MAX_DEC_LAYERS * 2];
            dc_inputs[0] = decode_token_input;
            dc_inputs[1] = decode_pos_input;
            for (int i = 0; i < n_kv; i++)
                dc_inputs[2 + i] = kv_caches[i];

            /* Run decode */
            memset(decode_outputs, 0, decode_n_outputs * sizeof(OrtValue *));
            ORT_CHECK(api->Run(ctx->decode, NULL, dc_in_names,
                      (const OrtValue *const *)dc_inputs, decode_n_inputs,
                      dc_out_names, decode_n_outputs, decode_outputs));

            /* Extract next token */
            float *logits;
            ORT_CHECK(api->GetTensorMutableData(decode_outputs[0], (void **)&logits));
            token = argmax_f32(logits, ctx->vocab_size);
            generated[n_generated++] = token;

            /* Free old KV caches, keep new ones */
            for (int i = 0; i < n_kv; i++) {
                api->ReleaseValue(kv_caches[i]);
                kv_caches[i] = decode_outputs[1 + i];
                decode_outputs[1 + i] = NULL;
            }
            /* Free decode logits */
            api->ReleaseValue(decode_outputs[0]);
            decode_outputs[0] = NULL;
        }

        double t_decode = get_time_ms();
        log_msg("[QwenOnnx] decode loop: %.1f ms (%d tokens, %.1f ms/token)\n",
                t_decode - t_prefill, n_generated, (t_decode - t_prefill) / (n_generated > 0 ? n_generated : 1));
        log_msg("[QwenOnnx] TOTAL inference: %.1f ms (%.2f audio sec)\n",
                t_decode - t_start, n_samples / 16000.0);
        last_mel_ms = t_mel - t_start;
        last_enc_ms = t_encoder - t_mel;
        last_prefill_ms = t_prefill - t_encoder;
        last_decode_ms = t_decode - t_prefill;
        last_total_ms = t_decode - t_start;
        last_n_tokens = n_generated;
        log_msg("Generated %d tokens\n", n_generated);

        /* Strip trailing EOS tokens */
        while (n_generated > 0 && is_eos(generated[n_generated - 1]))
            n_generated--;

        /* Decode tokens to text */
        /* Concatenate decoded token strings */
        size_t text_cap = 4096;
        char *text = (char *)malloc(text_cap);
        text[0] = '\0';
        size_t text_len = 0;

        int past_asr_text = 0;
        for (int i = 0; i < n_generated; i++) {
            if (generated[i] == QWEN_TOKEN_ASR_TEXT) {
                past_asr_text = 1;
                continue;
            }
            /* Skip language/special tokens before <asr_text> */
            if (!past_asr_text) continue;

            const char *piece = qwen_tokenizer_decode(ctx->tokenizer, generated[i]);
            if (piece) {
                size_t plen = strlen(piece);
                if (text_len + plen + 1 > text_cap) {
                    text_cap *= 2;
                    text = (char *)realloc(text, text_cap);
                }
                memcpy(text + text_len, piece, plen);
                text_len += plen;
                text[text_len] = '\0';
            }
        }

        /* If we never found <asr_text>, decode all tokens */
        if (!past_asr_text) {
            text_len = 0;
            text[0] = '\0';
            for (int i = 0; i < n_generated; i++) {
                /* Skip known special tokens */
                if (generated[i] >= 151643) continue;
                const char *piece = qwen_tokenizer_decode(ctx->tokenizer, generated[i]);
                if (piece) {
                    size_t plen = strlen(piece);
                    if (text_len + plen + 1 > text_cap) {
                        text_cap *= 2;
                        text = (char *)realloc(text, text_cap);
                    }
                    memcpy(text + text_len, piece, plen);
                    text_len += plen;
                    text[text_len] = '\0';
                }
            }
        }

        /* Trim leading/trailing whitespace */
        char *start = text;
        while (*start == ' ' || *start == '\n' || *start == '\t') start++;
        char *end = text + text_len;
        while (end > start && (end[-1] == ' ' || end[-1] == '\n' || end[-1] == '\t')) end--;

        size_t rlen = end - start;
        result = (char *)malloc(rlen + 1);
        memcpy(result, start, rlen);
        result[rlen] = '\0';
        free(text);
    }

cleanup:
    if (enc_input)  api->ReleaseValue(enc_input);
    if (enc_output) api->ReleaseValue(enc_output);
    if (prefill_input) api->ReleaseValue(prefill_input);
    if (decode_token_input) api->ReleaseValue(decode_token_input);
    if (decode_pos_input) api->ReleaseValue(decode_pos_input);
    if (prefill_outputs) {
        for (int i = 0; i < prefill_n_outputs; i++)
            if (prefill_outputs[i]) api->ReleaseValue(prefill_outputs[i]);
        free(prefill_outputs);
    }
    if (decode_outputs) {
        for (int i = 0; i < decode_n_outputs; i++)
            if (decode_outputs[i]) api->ReleaseValue(decode_outputs[i]);
        free(decode_outputs);
    }
    if (kv_caches) {
        for (int i = 0; i < n_kv; i++)
            if (kv_caches[i]) api->ReleaseValue(kv_caches[i]);
        free(kv_caches);
    }
    free(decode_token_buf);
    free(input_embeds);
    free(generated);

    /* Release ONNX sessions to keep memory low on mobile devices.
     * Desktop builds keep sessions loaded to avoid CreateSession costs. */
    if (!ctx->keep_sessions) {
        if (ctx->encoder) {
            api->ReleaseSession(ctx->encoder);
            ctx->encoder = NULL;
            log_msg("[QwenOnnx] released encoder session\n");
        }
        if (ctx->prefill) {
            api->ReleaseSession(ctx->prefill);
            ctx->prefill = NULL;
            log_msg("[QwenOnnx] released decoder_prefill session\n");
        }
        if (ctx->decode) {
            api->ReleaseSession(ctx->decode);
            ctx->decode = NULL;
            log_msg("[QwenOnnx] released decoder_decode session\n");
        }
    }

    return result;
}
