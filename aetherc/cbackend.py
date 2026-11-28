"""Aether native backend: lower validated .ae intent graphs to standalone C.

The generated C is self-contained and uses OpenMP for threading. When an
OpenBLAS installation is available it is linked in for sgemm; otherwise the
generated program falls back to a blocked, OpenMP-parallel matrix multiply.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from .ast_nodes import Binary, Call, Literal, Name, Program, Stmt, Task, Unary
from .cuda_template import CUDA_TEMPLATE
from .opencl_template import OPENCL_HOST_TEMPLATE, OPENCL_KERNEL_SOURCE

AETHER_C_VERSION = "1"


class CBackendError(RuntimeError):
    """Raised when C emission or compilation fails."""


@dataclass
class Toolchain:
    compiler: str
    kind: str  # gcc | clang | msvc
    version: str
    openblas_root: Path | None = None

    @property
    def has_blas(self) -> bool:
        return bool(self.openblas_root and (self.openblas_root / "include" / "cblas.h").is_file())

    @property
    def include_dir(self) -> Path | None:
        return self.openblas_root / "include" if self.has_blas else None

    @property
    def lib_dir(self) -> Path | None:
        return self.openblas_root / "lib" if self.has_blas else None

    @property
    def bin_dir(self) -> Path | None:
        return self.openblas_root / "bin" if self.has_blas else None

    @property
    def blas_name(self) -> str:
        return "OpenBLAS" if self.has_blas else "blocked-sgemm"


def _compiler_version(path: str, kind: str) -> str:
    try:
        if kind == "msvc":
            proc = subprocess.run(
                [path], capture_output=True, text=True, timeout=20, errors="replace"
            )
            for line in (proc.stdout + proc.stderr).splitlines():
                if "Version" in line:
                    return line.strip()
            return "cl"
        proc = subprocess.run(
            [path, "--version"], capture_output=True, text=True, timeout=20, errors="replace"
        )
        first = (proc.stdout or proc.stderr).splitlines()
        return first[0].strip() if first else path
    except Exception:
        return path


def find_openblas() -> Path | None:
    candidates: list[Path] = []
    explicit = os.environ.get("AETHER_OPENBLAS")
    if explicit:
        candidates.append(Path(explicit))
    local = os.environ.get("LOCALAPPDATA")
    if local:
        candidates.append(Path(local) / "Aether" / "toolchain" / "openblas")
    candidates.append(Path.home() / ".aether" / "toolchain" / "openblas")
    for candidate in candidates:
        if (candidate / "include" / "cblas.h").is_file() and (candidate / "lib").is_dir():
            return candidate
    return None


def find_toolchain(preferred: str | None = None) -> Toolchain | None:
    """Locate a native C compiler, including the portable bundled toolchain."""
    kind_by_name = {"gcc": "gcc", "cc": "gcc", "clang": "clang", "cl": "msvc"}

    def make(path: str) -> Toolchain:
        name = Path(path).stem.lower()
        kind = kind_by_name.get(name.split(".")[0], "gcc")
        if name.startswith("clang"):
            kind = "clang"
        return Toolchain(
            compiler=path,
            kind=kind,
            version=_compiler_version(path, kind),
            openblas_root=find_openblas(),
        )

    if preferred:
        resolved = shutil.which(preferred) or (preferred if Path(preferred).is_file() else None)
        if not resolved:
            raise CBackendError(f"compiler not found: {preferred}")
        return make(str(resolved))

    explicit = os.environ.get("AETHER_CC")
    if explicit:
        resolved = shutil.which(explicit) or (explicit if Path(explicit).is_file() else None)
        if resolved:
            return make(str(resolved))

    # On Windows, prefer the bundled, verified toolchain over whatever happens
    # to be first on PATH (third-party toolchains vary in runtime support).
    local = os.environ.get("LOCALAPPDATA")
    if os.name == "nt" and local:
        for name in ("gcc.exe", "clang.exe"):
            candidate = Path(local) / "Aether" / "toolchain" / "w64devkit" / "bin" / name
            if candidate.is_file():
                return make(str(candidate))

    for name in ("gcc", "clang", "cl"):
        resolved = shutil.which(name)
        if resolved:
            return make(resolved)

    local = os.environ.get("LOCALAPPDATA")
    roots: list[Path] = []
    if local:
        roots += [
            Path(local) / "Aether" / "toolchain" / "w64devkit" / "bin",
            Path(local) / "Aether" / "toolchain" / "mingw64" / "bin",
        ]
    for root in roots:
        for name in ("gcc.exe", "clang.exe"):
            candidate = root / name
            if candidate.is_file():
                return make(str(candidate))
    return None


# --------------------------------------------------------------------------- C source


PREAMBLE = r'''/*
 * Aether 0.2 -- generated native execution plan. DO NOT EDIT.
 * source: __SOURCE__
 * plan:   f32 | OpenMP threads | OpenBLAS sgemm when available | fused bias+relu
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdarg.h>

#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef AETHER_OPENBLAS
#include <cblas.h>
extern void openblas_set_num_threads(int num_threads);
#endif

/* ------------------------------------------------------------------ sockets */
#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <process.h>
#ifdef _MSC_VER
#pragma comment(lib, "ws2_32.lib")
#endif
typedef SOCKET ae_socket_t;
#define AE_INVALID_SOCKET INVALID_SOCKET
#define AE_CLOSE_SOCKET(s) closesocket(s)
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <pthread.h>
#include <signal.h>
typedef int ae_socket_t;
#define AE_INVALID_SOCKET (-1)
#define AE_CLOSE_SOCKET(s) close(s)
#include <netdb.h>
#endif

static int ae_net_ready = 0;

static void ae_net_once(void) {
    if (ae_net_ready) return;
#ifdef _WIN32
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) == 0) ae_net_ready = 1;
#else
    ae_net_ready = 1;
#endif
}

static int ae_recv(ae_socket_t fd, char *buf, int cap) {
#ifdef _WIN32
    return recv(fd, buf, cap, 0);
#else
    return (int)read(fd, buf, (size_t)cap);
#endif
}

static void ae_send_all(ae_socket_t fd, const char *buf, size_t len) {
#ifdef _WIN32
    int offset = 0;
    while (offset < (int)len) {
        int n = send(fd, buf + offset, (int)(len - (size_t)offset), 0);
        if (n <= 0) return;
        offset += n;
    }
#else
    size_t offset = 0;
    while (offset < len) {
        ssize_t n = write(fd, buf + offset, len - offset);
        if (n <= 0) return;
        offset += (size_t)n;
    }
#endif
}

#define AE_VERSION "1.0.0"
#define AE_MAX_BODY (1024 * 1024)  /* 1 MiB request-body hard limit */
#define AE_SOURCE "__SOURCE__"

static int ae_events = 0;
static int ae_silent = 0;
static uint64_t ae_rng_state = 0xAE7EULL;

/* Adaptive BLAS threading: small GEMMs lose more to synchronization than they
 * gain from cores, so the backend sizes the BLAS pool per call. The compiler
 * emits plain GEMM calls; the runtime tunes threads from the shapes. */
static int ae_max_threads = 1;
static int ae_blas_current = -1;

static int ae_threads_for(double flops) {
    int want = (int)(flops / 6e6);
    if (want < 1) want = 1;
    if (want > ae_max_threads) want = ae_max_threads;
    return want;
}

#ifdef AETHER_OPENBLAS
static void ae_blas_tune(int m, int k, int n) {
    int want = ae_threads_for(2.0 * (double)m * k * n);
    if (want != ae_blas_current) {
        openblas_set_num_threads(want);
        ae_blas_current = want;
    }
}
#else
static void ae_blas_tune(int m, int k, int n) {
    (void)m; (void)k; (void)n;
}
#endif

static double ae_now_ms(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart * 1000.0 / (double)freq.QuadPart;
#else
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1000.0 + (double)t.tv_nsec / 1e6;
#endif
}

static void ae_event(const char *fmt, ...) {
    va_list ap;
    if (!ae_events) return;
    va_start(ap, fmt);
    fputs("AETHER ", stderr);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
}

static void ae_human(const char *fmt, ...) {
    va_list ap;
    if (ae_silent) return;
    va_start(ap, fmt);
    vfprintf(stdout, fmt, ap);
    fputc('\n', stdout);
    fflush(stdout);
    va_end(ap);
}

static void *ae_alloc(size_t bytes) {
    void *p = malloc(bytes ? bytes : 1);
    if (!p) {
        fprintf(stderr, "aether: out of memory\n");
        exit(3);
    }
    return p;
}

static float *ae_falloc(size_t count) {
    return (float *)ae_alloc(count * sizeof(float));
}

static double ae_finite(double value) {
    return isfinite(value) ? value : 0.0;
}

static uint64_t ae_next(void) {
    uint64_t z = (ae_rng_state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static void ae_seed(uint64_t seed) {
    ae_rng_state = seed ? seed : 0x9E3779B97F4A7C15ULL;
}

static float ae_uniform(void) {
    return (float)((ae_next() >> 40) * (1.0 / 16777216.0));
}

static float ae_normal(void) {
    double u1 = ae_uniform();
    double u2 = ae_uniform();
    if (u1 < 1e-7) u1 = 1e-7;
    return (float)(sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2));
}

/* Deterministic, embarrassingly parallel normals: element i derives both
 * Box-Muller uniforms from a splitmix64 hash of its index, so large tensors
 * and datasets fill with OpenMP at full core count. */
static uint64_t ae_mix(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

static double ae_unit(uint64_t index) {
    return (double)(ae_mix(index) >> 11) * (1.0 / 9007199254740992.0) + 1e-13;
}

static float ae_normal_indexed(uint64_t index, uint64_t salt) {
    double u1 = ae_unit(index * 2 + salt * 2);
    double u2 = ae_unit(index * 2 + salt * 2 + 1);
    return (float)(sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2));
}

static void ae_fill_normal(float *data, size_t count, uint64_t salt) {
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < (long long)count; i++) {
        data[i] = ae_normal_indexed((uint64_t)i, salt);
    }
}

typedef struct { int rows, cols; float *data; } Tensor;
typedef struct { int samples, features, classes; float *x; int *y; } Dataset;
typedef struct {
    int in_dim, hidden, out_dim, kind, classes, trained;
    float *w1, *b1, *w2, *b2;
} Model;
typedef struct { int rows, out_dim, chunks; double checksum; unsigned digest; double accuracy; double loss; double gflops; } StreamResult;

static char ae_base_dir[1024] = "";

static void ae_resolve_path(const char *path, char *out, size_t cap) {
    int absolute = path[0] == '/' || path[0] == '\\' || (path[0] && path[1] == ':');
    if (!absolute && ae_base_dir[0]) {
        snprintf(out, cap, "%s/%s", ae_base_dir, path);
    } else {
        snprintf(out, cap, "%s", path);
    }
}

static void ae_gemm(const float *a, const float *b, float *c, int m, int k, int n) {
    ae_blas_tune(m, k, n);
#ifdef AETHER_OPENBLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, 1.0f, a, k, b, n, 0.0f, c, n);
#else
    #pragma omp parallel for schedule(static) num_threads(ae_threads_for(2.0 * (double)m * k * n))
    for (int i = 0; i < m; i++) {
        float *ci = c + (size_t)i * n;
        for (int j = 0; j < n; j++) ci[j] = 0.0f;
        for (int p = 0; p < k; p++) {
            float aip = a[(size_t)i * k + p];
            const float *bp = b + (size_t)p * n;
            for (int j = 0; j < n; j++) ci[j] += aip * bp[j];
        }
    }
#endif
}

/* C = A^T * B, where A is stored as k x m and B as k x n. */
static void ae_gemm_atb(const float *a, const float *b, float *c, int k, int m, int n) {
    ae_blas_tune(m, k, n);
#ifdef AETHER_OPENBLAS
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                m, n, k, 1.0f, a, m, b, n, 0.0f, c, n);
#else
    #pragma omp parallel for schedule(static) num_threads(ae_threads_for(2.0 * (double)m * k * n))
    for (int i = 0; i < m; i++) {
        float *ci = c + (size_t)i * n;
        for (int j = 0; j < n; j++) ci[j] = 0.0f;
        for (int p = 0; p < k; p++) {
            float aip = a[(size_t)p * m + i];
            const float *bp = b + (size_t)p * n;
            for (int j = 0; j < n; j++) ci[j] += aip * bp[j];
        }
    }
#endif
}

/* C = A * B^T, where A is stored as m x k and B as n x k. */
static void ae_gemm_abt(const float *a, const float *b, float *c, int m, int k, int n) {
    ae_blas_tune(m, k, n);
#ifdef AETHER_OPENBLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                m, n, k, 1.0f, a, k, b, k, 0.0f, c, n);
#else
    #pragma omp parallel for schedule(static) num_threads(ae_threads_for(2.0 * (double)m * k * n))
    for (int i = 0; i < m; i++) {
        float *ci = c + (size_t)i * n;
        const float *ai = a + (size_t)i * k;
        for (int j = 0; j < n; j++) {
            const float *bj = b + (size_t)j * k;
            float sum = 0.0f;
            for (int p = 0; p < k; p++) sum += ai[p] * bj[p];
            ci[j] = sum;
        }
    }
#endif
}

static void ae_forward(const Model *m, const float *x, int n, float *hidden, float *logits) {
    ae_gemm(x, m->w1, hidden, n, m->in_dim, m->hidden);
    for (int i = 0; i < n; i++) {
        float *row = hidden + (size_t)i * m->hidden;
        for (int j = 0; j < m->hidden; j++) {
            float v = row[j] + m->b1[j];
            row[j] = v > 0.0f ? v : 0.0f;
        }
    }
    ae_gemm(hidden, m->w2, logits, n, m->hidden, m->out_dim);
    for (int i = 0; i < n; i++) {
        float *row = logits + (size_t)i * m->out_dim;
        for (int j = 0; j < m->out_dim; j++) row[j] += m->b2[j];
    }
}

static void ae_softmax(float *logits, int n, int k) {
    for (int i = 0; i < n; i++) {
        float *row = logits + (size_t)i * k;
        float peak = row[0];
        for (int j = 1; j < k; j++) if (row[j] > peak) peak = row[j];
        float sum = 0.0f;
        for (int j = 0; j < k; j++) {
            float e = expf(row[j] - peak);
            row[j] = e;
            sum += e;
        }
        float inverse = sum > 0.0f ? 1.0f / sum : 0.0f;
        for (int j = 0; j < k; j++) row[j] *= inverse;
    }
}

static Tensor ae_tensor_new(int rows, int cols) {    Tensor t;
    t.rows = rows;
    t.cols = cols;
    t.data = ae_falloc((size_t)rows * cols);
    ae_fill_normal(t.data, (size_t)rows * cols,
                   0xAE7EULL ^ ((uint64_t)rows << 16) ^ (uint64_t)cols);
    double mib = (double)((size_t)rows * cols * sizeof(float)) / (1024.0 * 1024.0);
    ae_event("{\"event\":\"tensor\",\"rows\":%d,\"cols\":%d,\"mib\":%.6f,\"dtype\":\"f32\",\"backend\":\"c\"}",
             rows, cols, mib);
    ae_human("  [tensor] %dx%d | f32 | %.2f MiB | normal(seed=0xAE7E)", rows, cols, mib);
    return t;
}

static Dataset ae_dataset_new(int samples, int features, int classes) {
    Dataset d;
    d.samples = samples;
    d.features = features;
    d.classes = classes;
    d.x = ae_falloc((size_t)samples * features);
    d.y = (int *)ae_alloc((size_t)samples * sizeof(int));

    uint64_t dataset_salt = 0xAE7EULL ^ ((uint64_t)samples << 12) ^ ((uint64_t)features << 4) ^ (uint64_t)classes;
    ae_fill_normal(d.x, (size_t)samples * features, dataset_salt);

    float *true_w = ae_falloc((size_t)features * classes);
    float scale = 1.0f / sqrtf((float)features);
    for (int i = 0; i < features * classes; i++) true_w[i] = ae_normal() * scale;

    float *logits = ae_falloc((size_t)samples * classes);
    ae_gemm(d.x, true_w, logits, samples, features, classes);
    for (size_t i = 0; i < (size_t)samples * classes; i++) {
        logits[i] += 0.25f * ae_normal_indexed(i, dataset_salt + 0x51EDULL);
    }
    for (int i = 0; i < samples; i++) {
        int best = 0;
        float value = logits[(size_t)i * classes];
        for (int j = 1; j < classes; j++) {
            float candidate = logits[(size_t)i * classes + j];
            if (candidate > value) { value = candidate; best = j; }
        }
        d.y[i] = best;
    }
    free(true_w);
    free(logits);

    double mib = ((double)((size_t)samples * features * sizeof(float)) +
                  (double)((size_t)samples * sizeof(int))) / (1024.0 * 1024.0);
    ae_event("{\"event\":\"dataset\",\"samples\":%d,\"features\":%d,\"classes\":%d,\"mib\":%.6f,\"backend\":\"c\"}",
             samples, features, classes, mib);
    ae_human("  [dataset] %d samples x %d features | %d classes | %.2f MiB | synthetic", samples, features, classes, mib);
    return d;
}

static Model ae_model_new(int in_dim, int hidden, int out_dim, int kind) {
    Model m;
    memset(&m, 0, sizeof(m));
    m.in_dim = in_dim;
    m.hidden = hidden;
    m.out_dim = out_dim;
    m.kind = kind;
    m.classes = kind ? out_dim : 0;
    m.trained = kind ? 0 : 1;

    m.w1 = ae_falloc((size_t)in_dim * hidden);
    m.b1 = ae_falloc((size_t)hidden);
    m.w2 = ae_falloc((size_t)hidden * out_dim);
    m.b2 = ae_falloc((size_t)out_dim);

    uint64_t model_salt = 0xAE7EULL ^ ((uint64_t)in_dim << 20) ^ ((uint64_t)hidden << 8) ^ (uint64_t)out_dim ^ (kind ? 0x1000ULL : 0);
    float s1 = sqrtf(2.0f / (float)in_dim);
    float s2 = sqrtf(2.0f / (float)hidden);
    ae_fill_normal(m.w1, (size_t)in_dim * hidden, model_salt);
    for (int i = 0; i < in_dim * hidden; i++) m.w1[i] *= s1;
    for (int i = 0; i < hidden; i++) m.b1[i] = 0.0f;
    ae_fill_normal(m.w2, (size_t)hidden * out_dim, model_salt + 0x2B1FULL);
    for (int i = 0; i < hidden * out_dim; i++) m.w2[i] *= s2;
    for (int i = 0; i < out_dim; i++) m.b2[i] = 0.0f;

    double mib = ((double)((size_t)in_dim * hidden + hidden + (size_t)hidden * out_dim + out_dim) * sizeof(float)) / (1024.0 * 1024.0);
    int params = in_dim * hidden + hidden + hidden * out_dim + out_dim;
    ae_event("{\"event\":\"model\",\"kind\":\"%s\",\"in_dim\":%d,\"hidden\":%d,\"out_dim\":%d,\"params\":%d,\"mib\":%.6f,\"backend\":\"c\"}",
             kind ? "classifier" : "mlp", in_dim, hidden, out_dim, params, mib);
    if (kind) {
        ae_human("  [model] classifier %d -> %d -> %d | %d params | %.2f MiB | optimize(speed:max)",
                 in_dim, hidden, out_dim, params, mib);
    } else {
        ae_human("  [net]    %d -> %d -> %d | %d params | %.2f MiB | optimize(speed:max)",
                 in_dim, hidden, out_dim, params, mib);
    }
    return m;
}

static void ae_evaluate(const Model *m, const Dataset *d, double *accuracy, double *loss) {
    int batch = 512;
    if (batch > d->samples) batch = d->samples;
    float *hidden = ae_falloc((size_t)batch * m->hidden);
    float *logits = ae_falloc((size_t)batch * m->out_dim);
    long long correct = 0;
    double total_loss = 0.0;
    for (int start = 0; start < d->samples; start += batch) {
        int n = d->samples - start < batch ? d->samples - start : batch;
        ae_forward(m, d->x + (size_t)start * m->in_dim, n, hidden, logits);
        ae_softmax(logits, n, m->out_dim);
        for (int i = 0; i < n; i++) {
            float *row = logits + (size_t)i * m->out_dim;
            int label = d->y[start + i];
            total_loss += -log((double)row[label] + 1e-9);
            int best = 0;
            for (int j = 1; j < m->out_dim; j++) if (row[j] > row[best]) best = j;
            if (best == label) correct++;
        }
    }
    free(hidden);
    free(logits);
    *accuracy = (double)correct / (double)d->samples;
    *loss = total_loss / (double)d->samples;
}

static void ae_copy_weights(const Model *m, float *w1, float *b1, float *w2, float *b2) {
    memcpy(w1, m->w1, sizeof(float) * (size_t)m->in_dim * m->hidden);
    memcpy(b1, m->b1, sizeof(float) * (size_t)m->hidden);
    memcpy(w2, m->w2, sizeof(float) * (size_t)m->hidden * m->out_dim);
    memcpy(b2, m->b2, sizeof(float) * (size_t)m->out_dim);
}

static void ae_restore_weights(Model *m, const float *w1, const float *b1, const float *w2, const float *b2) {
    memcpy(m->w1, w1, sizeof(float) * (size_t)m->in_dim * m->hidden);
    memcpy(m->b1, b1, sizeof(float) * (size_t)m->hidden);
    memcpy(m->w2, w2, sizeof(float) * (size_t)m->hidden * m->out_dim);
    memcpy(m->b2, b2, sizeof(float) * (size_t)m->out_dim);
}

static double ae_train(Model *m, Dataset *d, int epochs, int batch, float learning_rate, int *heals_out, int *steps_out, double *accuracy_out) {
    if (batch > d->samples) batch = d->samples;
    int in = m->in_dim, hidden_dim = m->hidden, out = m->out_dim;

    float *xb = ae_falloc((size_t)batch * in);
    float *hb = ae_falloc((size_t)batch * hidden_dim);
    float *logits = ae_falloc((size_t)batch * out);
    float *dh = ae_falloc((size_t)batch * hidden_dim);

    size_t nw1 = (size_t)in * hidden_dim, nb1 = (size_t)hidden_dim;
    size_t nw2 = (size_t)hidden_dim * out, nb2 = (size_t)out;
    float *gw1 = ae_falloc(nw1), *gb1 = ae_falloc(nb1), *gw2 = ae_falloc(nw2), *gb2 = ae_falloc(nb2);
    float *vw1 = ae_falloc(nw1), *vb1 = ae_falloc(nb1), *vw2 = ae_falloc(nw2), *vb2 = ae_falloc(nb2);
    float *bw1 = ae_falloc(nw1), *bb1 = ae_falloc(nb1), *bw2 = ae_falloc(nw2), *bb2 = ae_falloc(nb2);
    memset(vw1, 0, sizeof(float) * nw1); memset(vb1, 0, sizeof(float) * nb1);
    memset(vw2, 0, sizeof(float) * nw2); memset(vb2, 0, sizeof(float) * nb2);
    ae_copy_weights(m, bw1, bb1, bw2, bb2);

    int *order = (int *)ae_alloc((size_t)d->samples * sizeof(int));
    int *label = (int *)ae_alloc((size_t)batch * sizeof(int));

    float lr = learning_rate;
    int heals = 0, steps = 0;
    double best_loss = 1e300, best_accuracy = 0.0;

    ae_event("{\"event\":\"train_start\",\"epochs\":%d,\"samples\":%d,\"features\":%d,\"classes\":%d,\"batch\":%d,\"lr\":%.6g,\"optimizer\":\"sgd+momentum\",\"backend\":\"c\"}",
             epochs, d->samples, d->features, d->classes, batch, lr);
    ae_human("  [train] %d epochs | %d samples | batch %d | sgd+momentum lr=%.3g | self_heal",
             epochs, d->samples, batch, lr);

    double train_started = ae_now_ms();
    for (int epoch = 0; epoch < epochs; epoch++) {
        double epoch_started = ae_now_ms();
        for (int i = 0; i < d->samples; i++) order[i] = i;
        for (int i = d->samples - 1; i > 0; i--) {
            int j = (int)(ae_next() % (uint64_t)(i + 1));
            int tmp = order[i]; order[i] = order[j]; order[j] = tmp;
        }

        double loss_sum = 0.0;
        long long seen = 0;
        int heals_epoch = 0;

        for (int start = 0; start < d->samples; start += batch) {
            int n = d->samples - start < batch ? d->samples - start : batch;
            for (int i = 0; i < n; i++) {
                int row = order[start + i];
                memcpy(xb + (size_t)i * in, d->x + (size_t)row * in, sizeof(float) * (size_t)in);
                label[i] = d->y[row];
            }

            ae_forward(m, xb, n, hb, logits);
            ae_softmax(logits, n, out);

            double loss = 0.0;
            for (int i = 0; i < n; i++) loss += -log((double)logits[(size_t)i * out + label[i]] + 1e-9);
            loss /= (double)n;

            for (int i = 0; i < n; i++) {
                float *row = logits + (size_t)i * out;
                for (int j = 0; j < out; j++) {
                    row[j] = (j == label[i]) ? row[j] - 1.0f : row[j];
                    row[j] /= (float)n;
                }
            }

            ae_gemm_atb(hb, logits, gw2, n, hidden_dim, out);
            memset(gb2, 0, sizeof(float) * nb2);
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < out; j++) gb2[j] += logits[(size_t)i * out + j];
            }

            ae_gemm_abt(logits, m->w2, dh, n, out, hidden_dim);
            for (size_t i = 0; i < (size_t)n * hidden_dim; i++) if (hb[i] <= 0.0f) dh[i] = 0.0f;

            ae_gemm_atb(xb, dh, gw1, n, in, hidden_dim);
            memset(gb1, 0, sizeof(float) * nb1);
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < hidden_dim; j++) gb1[j] += dh[(size_t)i * hidden_dim + j];
            }

            double norm_sq = 0.0;
            for (size_t i = 0; i < nw1; i++) norm_sq += (double)gw1[i] * gw1[i];
            for (size_t i = 0; i < nb1; i++) norm_sq += (double)gb1[i] * gb1[i];
            for (size_t i = 0; i < nw2; i++) norm_sq += (double)gw2[i] * gw2[i];
            for (size_t i = 0; i < nb2; i++) norm_sq += (double)gb2[i] * gb2[i];

            if (!isfinite(loss) || !isfinite(norm_sq)) {
                lr = lr * 0.5f > 1e-4f ? lr * 0.5f : 1e-4f;
                heals++;
                heals_epoch++;
                ae_event("{\"event\":\"heal\",\"stage\":\"step\",\"epoch\":%d,\"reason\":\"non-finite loss or gradient\",\"lr\":%.6g,\"backend\":\"c\"}",
                         epoch + 1, lr);
                continue;
            }

            float scale = norm_sq > 25.0 ? (float)(5.0 / sqrt(norm_sq)) : 1.0f;
            for (size_t i = 0; i < nw1; i++) { vw1[i] = 0.9f * vw1[i] - lr * scale * gw1[i]; m->w1[i] += vw1[i]; }
            for (size_t i = 0; i < nb1; i++) { vb1[i] = 0.9f * vb1[i] - lr * scale * gb1[i]; m->b1[i] += vb1[i]; }
            for (size_t i = 0; i < nw2; i++) { vw2[i] = 0.9f * vw2[i] - lr * scale * gw2[i]; m->w2[i] += vw2[i]; }
            for (size_t i = 0; i < nb2; i++) { vb2[i] = 0.9f * vb2[i] - lr * scale * gb2[i]; m->b2[i] += vb2[i]; }

            loss_sum += loss * (double)n;
            seen += n;
            steps++;
        }

        double accuracy = 0.0, validation_loss = 0.0;
        ae_evaluate(m, d, &accuracy, &validation_loss);
        double mean_loss = seen ? loss_sum / (double)seen : 0.0;
        int diverged = !isfinite(validation_loss) || (epoch > 0 && validation_loss > best_loss * 4.0 + 1e-6);
        if (diverged) {
            ae_restore_weights(m, bw1, bb1, bw2, bb2);
            lr = lr * 0.5f > 1e-4f ? lr * 0.5f : 1e-4f;
            heals++;
            heals_epoch++;
            ae_evaluate(m, d, &accuracy, &validation_loss);
            mean_loss = validation_loss;
            ae_event("{\"event\":\"heal\",\"stage\":\"epoch\",\"epoch\":%d,\"reason\":\"loss diverged; restored best weights\",\"lr\":%.6g,\"backend\":\"c\"}",
                     epoch + 1, lr);
            ae_human("  [heal]   epoch %d | restored best weights | lr -> %.4g", epoch + 1, lr);
        }
        if (validation_loss < best_loss) {
            best_loss = validation_loss;
            best_accuracy = accuracy;
            ae_copy_weights(m, bw1, bb1, bw2, bb2);
        }

        double epoch_ms = ae_now_ms() - epoch_started;
        ae_event("{\"event\":\"epoch\",\"total\":%d,\"epoch\":%d,\"loss\":%.6g,\"val_loss\":%.6g,\"accuracy\":%.6g,\"lr\":%.6g,\"ms\":%.6f,\"heals\":%d,\"backend\":\"c\"}",
                 epochs, epoch + 1, ae_finite(mean_loss), ae_finite(validation_loss), ae_finite(accuracy), ae_finite(lr), epoch_ms, heals_epoch);
        ae_human("  [epoch %d/%d] loss %.4f | acc %.4f | lr %.4g | %.1f ms",
                 epoch + 1, epochs, ae_finite(mean_loss), ae_finite(accuracy), ae_finite(lr), epoch_ms);
    }

    double elapsed = ae_now_ms() - train_started;
    m->trained = 1;
    *heals_out = heals;
    *steps_out = steps;
    *accuracy_out = best_accuracy;
    double throughput = elapsed > 0.0 ? (double)d->samples * epochs / (elapsed / 1000.0) : 0.0;

    ae_event("{\"event\":\"train_done\",\"epochs\":%d,\"samples\":%d,\"ms\":%.6f,\"accuracy\":%.6f,\"loss\":%.6g,\"heals\":%d,\"steps\":%d,\"lr\":%.6g,\"samples_per_s\":%.6f,\"backend\":\"c\"}",
             epochs, d->samples, elapsed, ae_finite(best_accuracy), ae_finite(best_loss), heals, steps, ae_finite(lr), throughput);
    ae_human("  [trained] acc %.4f | loss %.4f | %d epochs | %d steps | %d heal(s) | %.1f ms | %.0f samples/s",
             ae_finite(best_accuracy), ae_finite(best_loss), epochs, steps, heals, elapsed, throughput);

    free(xb); free(hb); free(logits); free(dh);
    free(gw1); free(gb1); free(gw2); free(gb2);
    free(vw1); free(vb1); free(vw2); free(vb2);
    free(bw1); free(bb1); free(bw2); free(bb2);
    free(order); free(label);
    return elapsed;
}

static void ae_json_escape(const char *src, char *dst, size_t cap) {
    size_t o = 0;
    for (const char *p = src; *p && o + 2 < cap; p++) {
        if (*p == '\\' || *p == '"') dst[o++] = '\\';
        dst[o++] = *p;
    }
    dst[o] = 0;
}

static void ae_export(const Model *m, const char *path) {
    uint32_t header[8];
    header[0] = 0x31424541u;  /* "AEB1" little-endian */
    header[1] = 1u;
    header[2] = (uint32_t)m->kind;
    header[3] = (uint32_t)m->in_dim;
    header[4] = (uint32_t)m->hidden;
    header[5] = (uint32_t)m->out_dim;
    header[6] = (uint32_t)m->classes;
    header[7] = 0u;

    /* Relative export paths resolve against the .ae source directory. */
    int absolute = path[0] == '/' || path[0] == '\\' || (path[0] && path[1] == ':');
    char resolved[1200];
    const char *target = path;
    if (!absolute && ae_base_dir[0]) {
        snprintf(resolved, sizeof(resolved), "%s/%s", ae_base_dir, path);
        target = resolved;
    }

    FILE *handle = fopen(target, "wb");
    if (!handle) {
        fprintf(stderr, "aether: cannot write %s\n", target);
        exit(3);
    }
    fwrite(header, sizeof(uint32_t), 8, handle);
    fwrite(m->w1, sizeof(float), (size_t)m->in_dim * m->hidden, handle);
    fwrite(m->b1, sizeof(float), (size_t)m->hidden, handle);
    fwrite(m->w2, sizeof(float), (size_t)m->hidden * m->out_dim, handle);
    fwrite(m->b2, sizeof(float), (size_t)m->out_dim, handle);
    long bytes = ftell(handle);
    fclose(handle);

    char escaped[2400];
    ae_json_escape(target, escaped, sizeof(escaped));
    ae_event("{\"event\":\"export\",\"path\":\"%s\",\"bytes\":%ld,\"params\":%d,\"format\":\"AEB1\",\"kind\":\"%s\",\"backend\":\"c\"}",
             escaped, bytes, m->in_dim * m->hidden + m->hidden + m->hidden * m->out_dim + m->out_dim,
             m->kind ? "classifier" : "mlp");
    ae_human("  [export] %s | %.1f KiB | AEB1 | %d params",
             target, (double)bytes / 1024.0, m->in_dim * m->hidden + m->hidden + m->hidden * m->out_dim + m->out_dim);
}

static StreamResult ae_stream(const Model *m, const Tensor *t, int chunks) {
    int rows = t->rows;
    int total = chunks < 1 ? 1 : (chunks > rows ? rows : chunks);
    double gflops_per_row = 2.0 * ((double)m->in_dim * m->hidden + (double)m->hidden * m->out_dim) / 1e9;
    double gflops = (double)rows * gflops_per_row;

    ae_event("{\"event\":\"stream_start\",\"chunks\":%d,\"rows\":%d,\"out_dim\":%d,\"chunk_rows\":%d,\"policy\":\"max\",\"backend\":\"c\"}",
             total, rows, m->out_dim, rows / total);
    ae_human("  [stream] %d chunks | %d rows/chunk | out %d | speed:max | OpenMP", total, rows / total, m->out_dim);

    unsigned digest = 2166136261u;
    double checksum = 0.0;
    double started = ae_now_ms();

    for (int index = 0; index < total; index++) {
        int start = (int)((long long)index * rows / total);
        int end = (int)((long long)(index + 1) * rows / total);
        int n = end - start;
        double chunk_started = ae_now_ms();

        float *hidden = ae_falloc((size_t)n * m->hidden);
        float *logits = ae_falloc((size_t)n * m->out_dim);
        ae_forward(m, t->data + (size_t)start * m->in_dim, n, hidden, logits);

        double sum = 0.0, square_sum = 0.0, peak = 0.0;
        for (size_t i = 0; i < (size_t)n * m->out_dim; i++) {
            double v = logits[i];
            sum += v;
            square_sum += v * v;
            if (fabs(v) > peak) peak = fabs(v);
            unsigned char *raw = (unsigned char *)&logits[i];
            for (int b = 0; b < 4; b++) { digest ^= raw[b]; digest *= 16777619u; }
        }
        long count = (long)n * m->out_dim;
        double mean = count ? sum / count : 0.0;
        double variance = count ? square_sum / count - mean * mean : 0.0;
        if (variance < 0.0) variance = 0.0;
        checksum += sum;
        double chunk_ms = ae_now_ms() - chunk_started;

        ae_event("{\"event\":\"chunk\",\"index\":%d,\"total\":%d,\"row_start\":%d,\"row_end\":%d,\"rows\":%d,\"mean\":%.6g,\"std\":%.6g,\"peak\":%.6g,\"ms\":%.6f,\"backend\":\"c\"}",
                 index + 1, total, start, end - 1, n, mean, sqrt(variance), peak, chunk_ms);
        ae_human("  [chunk %d/%d] rows %04d-%04d | mean %+.4f | std %.4f | peak %.3f | %.2f ms",
                 index + 1, total, start, end - 1, mean, sqrt(variance), peak, chunk_ms);

        free(hidden);
        free(logits);
    }

    double elapsed = ae_now_ms() - started;
    StreamResult result;
    result.rows = rows;
    result.out_dim = m->out_dim;
    result.chunks = total;
    result.checksum = checksum;
    result.digest = digest;
    result.accuracy = 0.0;
    result.loss = 0.0;
    result.gflops = gflops;

    ae_event("{\"event\":\"summary\",\"chunks\":%d,\"rows\":%d,\"out_dim\":%d,\"gflops\":%.6f,\"ms\":%.6f,\"gflops_per_s\":%.6f,\"checksum\":%.6g,\"digest\":\"0x%08x\",\"backend\":\"c\"}",
             total, rows, m->out_dim, gflops, elapsed, elapsed > 0.0 ? gflops / (elapsed / 1000.0) : 0.0, checksum, digest);
    ae_human("  [done]   %.2f GFLOP | %.1f ms | %.1f GFLOP/s | out[%dx%d] | checksum %+.2f | digest 0x%08x",
             gflops, elapsed, elapsed > 0.0 ? gflops / (elapsed / 1000.0) : 0.0, rows, m->out_dim, checksum, digest);
    return result;
}

static StreamResult ae_stream_classifier(const Model *m, const Dataset *d, int chunks) {
    int total = chunks < 1 ? 1 : (chunks > d->samples ? d->samples : chunks);
    double gflops_per_row = 2.0 * ((double)m->in_dim * m->hidden + (double)m->hidden * m->out_dim) / 1e9;
    double gflops = (double)d->samples * gflops_per_row;

    ae_event("{\"event\":\"stream_start\",\"chunks\":%d,\"rows\":%d,\"out_dim\":%d,\"chunk_rows\":%d,\"policy\":\"max\",\"kind\":\"classifier\",\"backend\":\"c\"}",
             total, d->samples, m->out_dim, d->samples / total);
    ae_human("  [stream] %d chunks | %d rows/chunk | out %d | classifier", total, d->samples / total, m->out_dim);

    unsigned digest = 2166136261u;
    long long correct = 0;
    double loss_sum = 0.0, confidence_sum = 0.0;
    double started = ae_now_ms();

    for (int index = 0; index < total; index++) {
        int start = (int)((long long)index * d->samples / total);
        int end = (int)((long long)(index + 1) * d->samples / total);
        int n = end - start;
        double chunk_started = ae_now_ms();

        float *hidden = ae_falloc((size_t)n * m->hidden);
        float *logits = ae_falloc((size_t)n * m->out_dim);
        ae_forward(m, d->x + (size_t)start * m->in_dim, n, hidden, logits);
        ae_softmax(logits, n, m->out_dim);

        long long chunk_correct = 0;
        double chunk_loss = 0.0, chunk_confidence = 0.0;
        for (int i = 0; i < n; i++) {
            float *row = logits + (size_t)i * m->out_dim;
            int label = d->y[start + i];
            int best = 0;
            for (int j = 1; j < m->out_dim; j++) if (row[j] > row[best]) best = j;
            if (best == label) chunk_correct++;
            chunk_loss += -log((double)row[label] + 1e-9);
            chunk_confidence += row[best];
            for (int j = 0; j < m->out_dim; j++) {
                unsigned char *raw = (unsigned char *)&row[j];
                for (int b = 0; b < 4; b++) { digest ^= raw[b]; digest *= 16777619u; }
            }
        }
        correct += chunk_correct;
        loss_sum += chunk_loss;
        confidence_sum += chunk_confidence;
        double accuracy = (double)chunk_correct / (double)n;
        double mean_loss = chunk_loss / (double)n;
        double confidence = chunk_confidence / (double)n;
        double chunk_ms = ae_now_ms() - chunk_started;

        ae_event("{\"event\":\"chunk\",\"index\":%d,\"total\":%d,\"row_start\":%d,\"row_end\":%d,\"rows\":%d,\"accuracy\":%.6g,\"confidence\":%.6g,\"loss\":%.6g,\"ms\":%.6f,\"backend\":\"c\"}",
                 index + 1, total, start, end - 1, n, accuracy, confidence, mean_loss, chunk_ms);
        ae_human("  [chunk %d/%d] samples %04d-%04d | acc %.4f | conf %.4f | loss %.4f | %.2f ms",
                 index + 1, total, start, end - 1, accuracy, confidence, mean_loss, chunk_ms);

        free(hidden);
        free(logits);
    }

    double elapsed = ae_now_ms() - started;
    double accuracy = (double)correct / (double)d->samples;
    double mean_loss = loss_sum / (double)d->samples;
    StreamResult result;
    result.rows = d->samples;
    result.out_dim = m->out_dim;
    result.chunks = total;
    result.checksum = confidence_sum;
    result.digest = digest;
    result.accuracy = accuracy;
    result.loss = mean_loss;
    result.gflops = gflops;

    ae_event("{\"event\":\"summary\",\"chunks\":%d,\"rows\":%d,\"out_dim\":%d,\"gflops\":%.6f,\"ms\":%.6f,\"gflops_per_s\":%.6f,\"checksum\":%.6g,\"digest\":\"0x%08x\",\"accuracy\":%.6g,\"loss\":%.6g,\"kind\":\"classifier\",\"backend\":\"c\"}",
             total, d->samples, m->out_dim, gflops, elapsed, elapsed > 0.0 ? gflops / (elapsed / 1000.0) : 0.0, confidence_sum, digest, accuracy, mean_loss);
    ae_human("  [done]   %.4f accuracy | loss %.4f | %.1f ms | %.1f GFLOP/s | out[%dx%d]",
             accuracy, mean_loss, elapsed, elapsed > 0.0 ? gflops / (elapsed / 1000.0) : 0.0, d->samples, m->out_dim);
    return result;
}

static char *ae_json_escape_heap(const char *src) {
    size_t cap = strlen(src) * 6 + 8;  /* worst case: every char -> \uXXXX */
    char *out = (char *)ae_alloc(cap);
    size_t o = 0;
    for (const char *p = src; *p && o + 7 < cap; p++) {
        unsigned char c = (unsigned char)*p;
        if (c == '"' || c == '\\') { out[o++] = '\\'; out[o++] = (char)c; }
        else if (c == '\n') { out[o++] = '\\'; out[o++] = 'n'; }
        else if (c == '\r') { out[o++] = '\\'; out[o++] = 'r'; }
        else if (c == '\t') { out[o++] = '\\'; out[o++] = 't'; }
        else if (c < 0x20) { o += (size_t)snprintf(out + o, cap - o, "\\u%04x", c); }
        else out[o++] = (char)c;
    }
    out[o] = 0;
    return out;
}
static void ae_emit_text(const char *text) {
    char *escaped = ae_json_escape_heap(text);
    ae_event("{\"event\":\"print\",\"text\":\"%s\",\"backend\":\"c\"}", escaped);
    free(escaped);
    ae_human("  [emit]   %s", text);
}

static void ae_log_text(const char *text) {
    char *escaped = ae_json_escape_heap(text);
    ae_event("{\"event\":\"log\",\"text\":\"%s\",\"backend\":\"c\"}", escaped);
    free(escaped);
    ae_human("  [log]    %s", text);
}

static void ae_emit_tensor(const Tensor *t) {
    double sum = 0.0, square_sum = 0.0, peak = 0.0;
    size_t count = (size_t)t->rows * t->cols;
    for (size_t i = 0; i < count; i++) {
        double v = t->data[i];
        sum += v; square_sum += v * v;
        if (fabs(v) > peak) peak = fabs(v);
    }
    double mean = count ? sum / count : 0.0;
    double variance = count ? square_sum / count - mean * mean : 0.0;
    if (variance < 0.0) variance = 0.0;
    ae_event("{\"event\":\"emit_tensor\",\"rows\":%d,\"cols\":%d,\"mean\":%.6g,\"std\":%.6g,\"peak\":%.6g,\"backend\":\"c\"}",
             t->rows, t->cols, mean, sqrt(variance), peak);
    ae_human("  [emit]   tensor[%dx%d] mean=%+.4f std=%.4f peak=%.3f", t->rows, t->cols, mean, sqrt(variance), peak);
}

static void ae_emit_stream(StreamResult value) {
    ae_event("{\"event\":\"emit_stream\",\"rows\":%d,\"out_dim\":%d,\"checksum\":%.6g,\"digest\":\"0x%08x\",\"accuracy\":%.6g,\"backend\":\"c\"}",
             value.rows, value.out_dim, value.checksum, value.digest, value.accuracy);
    ae_human("  [emit]   stream[%dx%d] checksum=%+.2f digest=0x%08x", value.rows, value.out_dim, value.checksum, value.digest);
}

/* ---------------------------------------------------------------- data.core */
static Dataset ae_load_csv(const char *path, const char *target_col) {
    char resolved[1200];
    ae_resolve_path(path, resolved, sizeof(resolved));
    FILE *handle = fopen(resolved, "rb");
    if (!handle) {
        fprintf(stderr, "aether: cannot open %s\n", resolved);
        exit(3);
    }

    char line[65536];
    if (!fgets(line, sizeof(line), handle)) {
        fprintf(stderr, "aether: %s is empty\n", resolved);
        exit(3);
    }
    /* header: split on commas, locate the target column */
    char columns[512][64];
    int column_count = 0;
    char *cursor = line;
    while (*cursor) {
        char *comma = strchr(cursor, ',');
        size_t span = comma ? (size_t)(comma - cursor) : strlen(cursor);
        if (span > 0 && cursor[span - 1] == '\n') span--;
        if (span > 0 && cursor[span - 1] == '\r') span--;
        if (span >= 64) span = 63;
        memcpy(columns[column_count], cursor, span);
        columns[column_count][span] = 0;
        column_count++;
        if (!comma) break;
        cursor = comma + 1;
    }
    int target_index = -1;
    for (int i = 0; i < column_count; i++) {
        if (strcmp(columns[i], target_col) == 0) target_index = i;
    }
    if (target_index < 0) {
        fprintf(stderr, "aether: target column '%s' not in %s header\n", target_col, resolved);
        exit(3);
    }
    int features = column_count - 1;
    if (features < 1 || features > 4096) {
        fprintf(stderr, "aether: %s has an unsupported feature count (%d)\n", resolved, features);
        exit(3);
    }

    size_t capacity = 1024;
    long long rows = 0;
    float *x = ae_falloc(capacity * (size_t)features);
    int *labels = (int *)ae_alloc(capacity * sizeof(int));
    char label_names[256][64];
    int label_count = 0;
    int feature_order[4096];
    int f = 0;
    for (int i = 0; i < column_count; i++) {
        if (i != target_index) feature_order[f++] = i;
    }

    while (fgets(line, sizeof(line), handle)) {
        size_t len = strlen(line);
        while (len && (line[len - 1] == '\n' || line[len - 1] == '\r')) line[--len] = 0;
        int blank = 1;
        for (size_t i = 0; i < len; i++) {
            if (line[i] != ' ' && line[i] != ',') { blank = 0; break; }
        }
        if (blank) continue;

        if ((size_t)rows >= capacity) {
            capacity *= 2;
            x = (float *)realloc(x, capacity * (size_t)features * sizeof(float));
            labels = (int *)realloc(labels, capacity * sizeof(int));
            if (!x || !labels) { fprintf(stderr, "aether: out of memory\n"); exit(3); }
        }

        char *cells[512];
        int cell_count = 0;
        cursor = line;
        while (*cursor && cell_count < 512) {
            cells[cell_count++] = cursor;
            char *comma = strchr(cursor, ',');
            if (!comma) break;
            *comma = 0;
            cursor = comma + 1;
        }
        if (cell_count != column_count) {
            fprintf(stderr, "aether: %s line %lld has %d cells, expected %d\n",
                    resolved, rows + 2, cell_count, column_count);
            exit(3);
        }

        float *row = x + (size_t)rows * features;
        for (int i = 0; i < features; i++) {
            row[i] = strtof(cells[feature_order[i]], NULL);
        }

        char *label_text = cells[target_index];
        char *end = NULL;
        float numeric = strtof(label_text, &end);
        int label;
        if (end != label_text && *end == 0) {
            label = (int)numeric;
        } else {
            label = -1;
            for (int i = 0; i < label_count; i++) {
                if (strcmp(label_names[i], label_text) == 0) { label = i; break; }
            }
            if (label < 0) {
                if (label_count >= 256) {
                    fprintf(stderr, "aether: too many distinct labels in %s\n", resolved);
                    exit(3);
                }
                snprintf(label_names[label_count], 64, "%s", label_text);
                label = label_count++;
            }
        }
        labels[rows] = label;
        rows++;
    }
    fclose(handle);

    int classes = 2;
    for (long long i = 0; i < rows; i++) {
        if (labels[i] + 1 > classes) classes = labels[i] + 1;
    }
    if (label_count > 0) classes = label_count;
    if (rows < 1) {
        fprintf(stderr, "aether: %s has no data rows\n", resolved);
        exit(3);
    }

    Dataset d;
    d.samples = (int)rows;
    d.features = features;
    d.classes = classes;
    d.x = x;
    d.y = labels;

    double mib = ((double)((size_t)rows * features * sizeof(float)) +
                  (double)((size_t)rows * sizeof(int))) / (1024.0 * 1024.0);
    char escaped[2400];
    ae_json_escape(resolved, escaped, sizeof(escaped));
    ae_event("{\"event\":\"dataset\",\"samples\":%d,\"features\":%d,\"classes\":%d,\"mib\":%.6f,\"source\":\"%s\",\"backend\":\"c\"}",
             d.samples, features, classes, mib, escaped);
    ae_human("  [dataset] %d rows x %d features | %d classes | %.2f MiB | from %s",
             d.samples, features, classes, mib, resolved);
    return d;
}

static void ae_save_json_text(const char *path, const char *json) {
    char resolved[1200];
    ae_resolve_path(path, resolved, sizeof(resolved));
    FILE *handle = fopen(resolved, "wb");
    if (!handle) {
        fprintf(stderr, "aether: cannot write %s\n", resolved);
        exit(3);
    }
    fputs(json, handle);
    long bytes = ftell(handle);
    fclose(handle);
    char escaped[2400];
    ae_json_escape(resolved, escaped, sizeof(escaped));
    ae_event("{\"event\":\"save_json\",\"path\":\"%s\",\"bytes\":%ld,\"backend\":\"c\"}", escaped, bytes);
    ae_human("  [json]   %s | %ld bytes", resolved, bytes);
}

static void ae_save_json_scalar(const char *key, double value, const char *path) {
    char json[256];
    snprintf(json, sizeof(json), "{\n  \"%s\": %.17g\n}\n", key, ae_finite(value));
    ae_save_json_text(path, json);
}

static void ae_save_json_stream(const StreamResult *r, const char *path) {
    char json[512];
    snprintf(json, sizeof(json),
             "{\n  \"rows\": %d,\n  \"out_dim\": %d,\n  \"chunks\": %d,\n"
             "  \"accuracy\": %.6f,\n  \"loss\": %.6f,\n  \"gflops\": %.6f,\n"
             "  \"checksum\": %.6g,\n  \"digest\": \"0x%08x\"\n}\n",
             r->rows, r->out_dim, r->chunks, ae_finite(r->accuracy), ae_finite(r->loss),
             r->gflops, r->checksum, r->digest);
    ae_save_json_text(path, json);
}

static void ae_save_json_model(const Model *m, const char *path) {
    char json[320];
    snprintf(json, sizeof(json),
             "{\n  \"kind\": \"%s\",\n  \"in_dim\": %d,\n  \"hidden\": %d,\n"
             "  \"out_dim\": %d,\n  \"classes\": %d,\n  \"params\": %d,\n"
             "  \"trained\": %s\n}\n",
             m->kind ? "classifier" : "mlp", m->in_dim, m->hidden, m->out_dim,
             m->classes, m->in_dim * m->hidden + m->hidden + m->hidden * m->out_dim + m->out_dim,
             m->trained ? "true" : "false");
    ae_save_json_text(path, json);
}

static void ae_while_guard(long long max_iters) {
    ae_event("{\"event\":\"while_guard\",\"max_iters\":%lld,\"message\":\"while loop hit max_iters(%lld); exiting\",\"backend\":\"c\"}",
             max_iters, max_iters);
    ae_human("  [guard]  while loop hit max_iters(%lld); exiting", max_iters);
}

/* Trains and returns the best validation accuracy (the `accuracy` binding). */
static double ae_train_accuracy(Model *m, Dataset *d, int epochs, int batch, float lr, int *heals, int *steps) {
    double accuracy = 0.0;
    ae_train(m, d, epochs, batch, lr, heals, steps, &accuracy);
    return accuracy;
}

/* -------------------------------------------------------- agent.core memory */
typedef struct { char key[64]; double value; } AeMem;
static AeMem ae_mem[256];
static int ae_mem_count = 0;
static int ae_mem_loaded = 0;

static void ae_memory_load(const char *path) {
    ae_mem_count = 0;
    ae_mem_loaded = 1;
    char resolved[1200];
    ae_resolve_path(path, resolved, sizeof(resolved));
    FILE *handle = fopen(resolved, "rb");
    if (!handle) return;
    static char buffer[65536];
    size_t got = fread(buffer, 1, sizeof(buffer) - 1, handle);
    fclose(handle);
    buffer[got] = 0;

    char *cursor = buffer;
    while (*cursor && ae_mem_count < 256) {
        char *quote = strchr(cursor, '"');
        if (!quote) break;
        char *key_start = quote + 1;
        char *key_end = strchr(key_start, '"');
        if (!key_end) break;
        size_t key_len = (size_t)(key_end - key_start);
        if (key_len >= 64) key_len = 63;
        char *colon = strchr(key_end + 1, ':');
        if (!colon) break;
        char *value_end = NULL;
        double value = strtod(colon + 1, &value_end);
        if (value_end != colon + 1) {
            AeMem *slot = &ae_mem[ae_mem_count++];
            memcpy(slot->key, key_start, key_len);
            slot->key[key_len] = 0;
            slot->value = value;
        }
        cursor = key_end + 1;
    }
}

static void ae_memory_save(const char *path) {
    char resolved[1200];
    ae_resolve_path(path, resolved, sizeof(resolved));
    FILE *handle = fopen(resolved, "wb");
    if (!handle) {
        fprintf(stderr, "aether: cannot write memory %s\n", resolved);
        exit(3);
    }
    fputs("{\n", handle);
    for (int i = 0; i < ae_mem_count; i++) {
        char escaped[160];
        ae_json_escape(ae_mem[i].key, escaped, sizeof(escaped));
        fprintf(handle, "  \"%s\": %.17g%s\n", escaped, ae_mem[i].value,
                i + 1 < ae_mem_count ? "," : "");
    }
    fputs("}\n", handle);
    fclose(handle);
}

static double ae_memory_recall(const char *key, double default_value, const char *path) {
    if (!ae_mem_loaded) ae_memory_load(path);
    double value = default_value;
    int found = 0;
    for (int i = 0; i < ae_mem_count; i++) {
        if (strcmp(ae_mem[i].key, key) == 0) { value = ae_mem[i].value; found = 1; break; }
    }
    char escaped[160];
    ae_json_escape(key, escaped, sizeof(escaped));
    ae_event("{\"event\":\"recall\",\"key\":\"%s\",\"found\":%s,\"value\":%.17g,\"default\":%.17g,\"backend\":\"c\"}",
             escaped, found ? "true" : "false", value, default_value);
    ae_human("  [memory] recalled %s = %.4g (%s)", key, value, found ? "hit" : "default");
    return value;
}

static long long ae_memory_remember(const char *key, double value, const char *path) {
    if (!ae_mem_loaded) ae_memory_load(path);
    double previous = 0.0;
    int existed = 0;
    int slot = -1;
    for (int i = 0; i < ae_mem_count; i++) {
        if (strcmp(ae_mem[i].key, key) == 0) { slot = i; existed = 1; previous = ae_mem[i].value; break; }
    }
    if (slot < 0 && ae_mem_count < 256) {
        slot = ae_mem_count++;
    }
    if (slot >= 0) {
        snprintf(ae_mem[slot].key, sizeof(ae_mem[slot].key), "%s", key);
        ae_mem[slot].value = value;
    }
    ae_memory_save(path);
    char escaped[160];
    ae_json_escape(key, escaped, sizeof(escaped));
    ae_event("{\"event\":\"remember\",\"key\":\"%s\",\"value\":%.17g,\"previous\":%.17g,\"existed\":%s,\"backend\":\"c\"}",
             escaped, value, previous, existed ? "true" : "false");
    ae_human("  [memory] remembered %s = %.4g%s", key, value,
             existed ? " (updated)" : "");
    return 1;
}

/* ------------------------------------------------------------ reductions */
static double ae_tensor_sum(const Tensor *t) {
    double acc = 0.0;
    const size_t count = (size_t)t->rows * t->cols;
    #pragma omp parallel for reduction(+:acc) schedule(static)
    for (long long i = 0; i < (long long)count; i++) acc += t->data[i];
    ae_event("{\"event\":\"reduce\",\"op\":\"sum\",\"value\":%.17g,\"elements\":%lld,\"backend\":\"c\"}",
             acc, (long long)count);
    ae_human("  [sum ]   %.6g | %lld elements", acc, (long long)count);
    return acc;
}

static double ae_tensor_mean(const Tensor *t) {
    double acc = 0.0;
    const size_t count = (size_t)t->rows * t->cols;
    #pragma omp parallel for reduction(+:acc) schedule(static)
    for (long long i = 0; i < (long long)count; i++) acc += t->data[i];
    double value = count ? acc / (double)count : 0.0;
    ae_event("{\"event\":\"reduce\",\"op\":\"mean\",\"value\":%.17g,\"elements\":%lld,\"backend\":\"c\"}",
             value, (long long)count);
    ae_human("  [mean]   %.6g | %lld elements", value, (long long)count);
    return value;
}

/* --------------------------------------------------------- reductions + math */
static double ae_tensor_min(const Tensor *t) {
    const size_t count = (size_t)t->rows * t->cols;
    double value = count ? (double)t->data[0] : 0.0;
    for (size_t i = 1; i < count; i++) {
        if ((double)t->data[i] < value) value = (double)t->data[i];
    }
    ae_event("{\"event\":\"reduce\",\"op\":\"min\",\"value\":%.17g,\"elements\":%lld,\"backend\":\"c\"}",
             value, (long long)count);
    ae_human("  [min ]   %.6g | %lld elements", value, (long long)count);
    return value;
}

static double ae_tensor_max(const Tensor *t) {
    const size_t count = (size_t)t->rows * t->cols;
    double value = count ? (double)t->data[0] : 0.0;
    for (size_t i = 1; i < count; i++) {
        if ((double)t->data[i] > value) value = (double)t->data[i];
    }
    ae_event("{\"event\":\"reduce\",\"op\":\"max\",\"value\":%.17g,\"elements\":%lld,\"backend\":\"c\"}",
             value, (long long)count);
    ae_human("  [max ]   %.6g | %lld elements", value, (long long)count);
    return value;
}

static long long ae_tensor_argmax(const Tensor *t) {
    const size_t count = (size_t)t->rows * t->cols;
    long long best = 0;
    for (size_t i = 1; i < count; i++) {
        if ((double)t->data[i] > (double)t->data[best]) best = (long long)i;
    }
    ae_event("{\"event\":\"reduce\",\"op\":\"argmax\",\"value\":%lld,\"elements\":%lld,\"backend\":\"c\"}",
             best, (long long)count);
    ae_human("  [argmax] %lld | %lld elements", best, (long long)count);
    return best;
}

/* value-taking wrappers: reductions over tensor expressions */
static double ae_tensor_mean_expr(Tensor t) { return ae_tensor_mean(&t); }
static double ae_tensor_sum_expr(Tensor t) { return ae_tensor_sum(&t); }
static double ae_tensor_min_expr(Tensor t) { return ae_tensor_min(&t); }
static double ae_tensor_max_expr(Tensor t) { return ae_tensor_max(&t); }
static long long ae_tensor_argmax_expr(Tensor t) { return ae_tensor_argmax(&t); }

static Tensor ae_tensor_scaled(const Tensor *t, double factor) {
    Tensor out;
    out.rows = t->rows;
    out.cols = t->cols;
    size_t count = (size_t)t->rows * t->cols;
    out.data = ae_falloc(count);
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < (long long)count; i++) out.data[i] = t->data[i] * (float)factor;
    ae_event("{\"event\":\"scale\",\"factor\":%.17g,\"rows\":%d,\"cols\":%d,\"backend\":\"c\"}",
             factor, out.rows, out.cols);
    ae_human("  [scale]  x%.4g | out[%dx%d]", factor, out.rows, out.cols);
    return out;
}

/* ------------------------------------------------------------- string utils */

static char *ae_str_dup(const char *text) {
    size_t len = strlen(text);
    char *copy = (char *)ae_alloc(len + 1);
    memcpy(copy, text, len + 1);
    return copy;
}

static char *ae_str_concat(const char *a, const char *b) {
    size_t la = strlen(a), lb = strlen(b);
    char *out = (char *)ae_alloc(la + lb + 1);
    memcpy(out, a, la);
    memcpy(out + la, b, lb + 1);
    return out;
}

static char *ae_str_of(double value) {
    char buffer[64];
    snprintf(buffer, sizeof(buffer), "%.6g", value);
    return ae_str_dup(buffer);
}

/* ------------------------------------------------------------------ llm.core */

static char *ae_json_string_field(const char *body, const char *key) {
    char pattern[128];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *cursor = body;
    while ((cursor = strstr(cursor, pattern)) != NULL) {
        cursor += strlen(pattern);
        while (*cursor == ' ' || *cursor == ':') cursor++;
        if (*cursor != '"') continue;
        cursor++;
        size_t cap = strlen(cursor) + 1;
        char *out = (char *)ae_alloc(cap);
        size_t o = 0;
        while (*cursor && o + 2 < cap) {
            if (*cursor == '\\' && cursor[1]) {
                char next = cursor[1];
                out[o++] = next == 'n' ? '\n' : next == 't' ? '\t' : next == 'r' ? '\r' : next;
                cursor += 2;
                continue;
            }
            if (*cursor == '"') break;
            out[o++] = *cursor++;
        }
        out[o] = 0;
        return out;
    }
    return NULL;
}

/* HTTPS via runtime-loaded transports: WinHTTP (Windows, Schannel TLS) and
 * libcurl (POSIX). Zero compile-time dependencies; honest errors otherwise. */
#ifdef _WIN32
typedef HMODULE ae_dl_t;
typedef FARPROC ae_sym_t;
static ae_dl_t ae_sys_dlopen(const char *path) { return LoadLibraryA(path); }
static ae_sym_t ae_sys_dlsym(ae_dl_t handle, const char *name) { return GetProcAddress(handle, name); }
#else
typedef void *ae_dl_t;
typedef void *ae_sym_t;
static ae_dl_t ae_sys_dlopen(const char *path) { return dlopen(path, RTLD_NOW | RTLD_LOCAL); }
static ae_sym_t ae_sys_dlsym(ae_dl_t handle, const char *name) { return dlsym(handle, name); }
#endif

#ifdef _WIN32
typedef void *ae_winhttp_t;
typedef ae_winhttp_t (*pfn_aeWinhttpOpen)(const char *, const char *, const char *, const char *, unsigned int);
typedef ae_winhttp_t (*pfn_aeWinhttpConnect)(ae_winhttp_t, const char *, unsigned short, unsigned int);
typedef ae_winhttp_t (*pfn_aeWinhttpOpenRequest)(ae_winhttp_t, const char *, const char *, const char *, const char *, const char *, unsigned int);
typedef int (*pfn_aeWinhttpSendRequest)(ae_winhttp_t, const char *, unsigned int, void *, unsigned int, unsigned int, void *);
typedef int (*pfn_aeWinhttpReceiveResponse)(ae_winhttp_t, void *);
typedef int (*pfn_aeWinhttpReadData)(ae_winhttp_t, void *, unsigned int, unsigned int *);
typedef int (*pfn_aeWinhttpCloseHandle)(ae_winhttp_t);
static pfn_aeWinhttpOpen aeWinhttpOpen;
static pfn_aeWinhttpConnect aeWinhttpConnect;
static pfn_aeWinhttpOpenRequest aeWinhttpOpenRequest;
static pfn_aeWinhttpSendRequest aeWinhttpSendRequest;
static pfn_aeWinhttpReceiveResponse aeWinhttpReceiveResponse;
static pfn_aeWinhttpReadData aeWinhttpReadData;
static pfn_aeWinhttpCloseHandle aeWinhttpCloseHandle;

static int ae_winhttp_load(void) {
    static ae_dl_t lib = NULL;
    if (lib) return aeWinhttpOpen ? 0 : -1;
    lib = ae_sys_dlopen("winhttp.dll");
    if (!lib) return -1;
#define AE_WINHTTP_SYM(var, name) \
    var = (pfn_##var)ae_sys_dlsym(lib, name); \
    if (!var) return -2
    AE_WINHTTP_SYM(aeWinhttpOpen, "WinHTTPOpen");
    AE_WINHTTP_SYM(aeWinhttpConnect, "WinHTTPConnect");
    AE_WINHTTP_SYM(aeWinhttpOpenRequest, "WinHTTPOpenRequest");
    AE_WINHTTP_SYM(aeWinhttpSendRequest, "WinHTTPSendRequest");
    AE_WINHTTP_SYM(aeWinhttpReceiveResponse, "WinHTTPReceiveResponse");
    AE_WINHTTP_SYM(aeWinhttpReadData, "WinHTTPReadData");
    AE_WINHTTP_SYM(aeWinhttpCloseHandle, "WinHTTPCloseHandle");
#undef AE_WINHTTP_SYM
    return 0;
}

/* POSTs body to an https:// URL via WinHTTP (Schannel TLS). Returns a heap
 * response body or NULL. */
static char *ae_https_post_winhttp(const char *url, const char *body, const char *bearer) {
    if (ae_winhttp_load() != 0) return NULL;
    /* url: https://host[:port]/path */
    const char *rest = url + 8;
    const char *slash = strchr(rest, '/');
    char host[256] = {0};
    unsigned short port = 443;
    const char *path = "/";
    if (slash) {
        size_t span = (size_t)(slash - rest);
        if (span >= sizeof(host)) span = sizeof(host) - 1;
        memcpy(host, rest, span);
        path = slash;
    } else {
        snprintf(host, sizeof(host), "%s", rest);
    }
    char *colon = strchr(host, ':');
    if (colon) { *colon = 0; port = (unsigned short)atoi(colon + 1); }

    ae_winhttp_t session = aeWinhttpOpen("Aether/1.0", NULL, NULL, NULL, 0);
    if (!session) return NULL;
    ae_winhttp_t connection = aeWinhttpConnect(session, host, port, 0);
    if (!connection) { aeWinhttpCloseHandle(session); return NULL; }
    ae_winhttp_t request = aeWinhttpOpenRequest(connection, "POST", path, NULL, NULL, NULL,
                                                0x00800000u /* WINHTTP_FLAG_SECURE */);
    if (!request) { aeWinhttpCloseHandle(connection); aeWinhttpCloseHandle(session); return NULL; }

    char headers[512];
    snprintf(headers, sizeof(headers), "Content-Type: application/json\r\nAuthorization: Bearer %s\r\n", bearer);
    size_t body_len = strlen(body);
    int sent = aeWinhttpSendRequest(request, headers, (unsigned int)strlen(headers),
                                    (void *)body, (unsigned int)body_len,
                                    (unsigned int)body_len, NULL);
    char *result = NULL;
    if (sent && aeWinhttpReceiveResponse(request, NULL)) {
        size_t cap = 262144, got = 0;
        result = (char *)ae_alloc(cap);
        unsigned int read = 0;
        for (;;) {
            if (got + 1 >= cap) break;
            if (!aeWinhttpReadData(request, result + got, (unsigned int)(cap - 1 - got), &read) || read == 0) break;
            got += read;
            result[got] = 0;
        }
    }
    aeWinhttpCloseHandle(request);
    aeWinhttpCloseHandle(connection);
    aeWinhttpCloseHandle(session);
    return result;
}
#endif

#ifndef _WIN32
typedef void *ae_curl_t;
typedef ae_curl_t (*pfn_curl_easy_init)(void);
typedef int (*pfn_curl_easy_setopt_impl)(ae_curl_t, int, ...);
typedef int (*pfn_curl_easy_perform_impl)(ae_curl_t);
typedef void (*pfn_curl_easy_cleanup)(ae_curl_t);
typedef void *(*pfn_curl_slist_append)(void *, const char *);
typedef void (*pfn_curl_slist_free_all)(void *);
static ae_curl_t curlEasyInit;
static pfn_curl_easy_setopt_impl curlEasySetopt;
static pfn_curl_easy_perform_impl curlEasyPerform;
static pfn_curl_easy_cleanup curlEasyCleanup;
static pfn_curl_slist_append curlSlistAppend;
static pfn_curl_slist_free_all curlSlistFreeAll;

static size_t ae_curl_write_cb(char *ptr, size_t size, size_t nmemb, void *userdata) {
    size_t total = size * nmemb;
    char **out = (char **)userdata;
    static char *buffer = NULL;
    static size_t used = 0, cap = 0;
    (void)out;
    if (!buffer) { cap = 262144; buffer = (char *)ae_alloc(cap); used = 0; buffer[0] = 0; }
    if (used + total + 1 >= cap) { cap = (used + total) * 2 + 8; buffer = (char *)realloc(buffer, cap); }
    memcpy(buffer + used, ptr, total);
    used += total;
    buffer[used] = 0;
    return total;
}

static void ae_curl_load(void) {
    static int loaded = 0;
    if (loaded) return;
    loaded = 1;
    static ae_dl_t lib = NULL;
    const char *candidates[] = {
        "libcurl.so.4", "libcurl.so", "/usr/lib/x86_64-linux-gnu/libcurl.so.4", NULL
    };
    for (int i = 0; candidates[i] && !lib; i++) lib = ae_dlopen(candidates[i]);
    if (!lib) return;
    curlEasyInit = (pfn_curl_easy_init)ae_sys_dlsym(lib, "curl_easy_init");
    curlEasySetopt = (pfn_curl_easy_setopt_impl)ae_sys_dlsym(lib, "curl_easy_setopt");
    curlEasyPerform = (pfn_curl_easy_perform_impl)ae_sys_dlsym(lib, "curl_easy_perform");
    curlEasyCleanup = (pfn_curl_easy_cleanup)ae_sys_dlsym(lib, "curl_easy_cleanup");
    curlSlistAppend = (pfn_curl_slist_append)ae_sys_dlsym(lib, "curl_slist_append");
    curlSlistFreeAll = (pfn_curl_slist_free_all)ae_sys_dlsym(lib, "curl_slist_free_all");
}

/* POSTs body to an https:// URL via libcurl. Returns heap body or NULL. */
static char *ae_https_post_curl(const char *url, const char *body, const char *bearer) {
    ae_curl_load();
    if (!curlEasyInit || !curlEasySetopt || !curlEasyPerform) return NULL;
    ae_curl_t handle = curlEasyInit();
    if (!handle) return NULL;
    static char *response = NULL;
    response = NULL;
    int opt_url = 10002, opt_post = 10015, opt_write = 20011, opt_follow = 52, opt_headers = 10023;
    curlEasySetopt(handle, opt_url, url);
    curlEasySetopt(handle, opt_post, body);
    curlEasySetopt(handle, opt_write, (void *)&ae_curl_write_cb);
    curlEasySetopt(handle, opt_follow, 1L);
    void *header_list = curlSlistAppend(NULL, "Content-Type: application/json");
    char auth_header[512];
    snprintf(auth_header, sizeof(auth_header), "Authorization: Bearer %s", bearer);
    header_list = curlSlistAppend(header_list, auth_header);
    curlEasySetopt(handle, opt_headers, header_list);
    int rc = curlEasyPerform();
    curlSlistFreeAll(header_list);
    curlEasyCleanup(handle);
    return rc == 0 ? ae_str_dup(response ? response : "") : NULL;
}
#endif

static char *ae_llm_ask(const char *prompt, const char *model) {
    const char *mock_env = getenv("AETHER_LLM_MOCK");
    const char *url_env = getenv("AETHER_LLM_URL");
    const char *key_env = getenv("AETHER_LLM_KEY");
    int mock = (mock_env && strcmp(mock_env, "1") == 0) || !(url_env && key_env);
    double started = ae_now_ms();
    char *reply = NULL;
    const char *mode = "mock";

    if (mock) {
        size_t cap = strlen(model) + strlen(prompt) + 32;
        reply = (char *)ae_alloc(cap);
        snprintf(reply, cap, "[%s mock] %s", model, prompt);
        mode = (mock_env && strcmp(mock_env, "1") == 0) ? "mock" : "auto-mock";
    } else if (strncmp(url_env, "https://", 8) == 0) {
        char *payload_json = ae_json_escape_heap(prompt);
        char body[8192];
        snprintf(body, sizeof(body),
                 "{\"model\":\"%s\",\"messages\":[{\"role\":\"user\",\"content\":\"%s\"}]}",
                 model, payload_json);
        free(payload_json);
        const char *bearer = key_env ? key_env : "";
#ifdef _WIN32
        char *response = ae_https_post_winhttp(url_env, body, bearer);
        if (response) {
            char *http_body = strstr(response, "\r\n\r\n");
            reply = http_body ? ae_json_string_field(http_body + 4, "content") : NULL;
            if (!reply) {
                reply = ae_str_dup("[llm error] no completion in HTTPS response");
                mode = "error";
            }
            free(response);
        } else {
            reply = ae_str_dup("[llm error] HTTPS request failed (WinHTTP unavailable or network error)");
            mode = "error";
        }
#else
        ae_curl_load();
        char *response = ae_https_post_curl(url_env, body, bearer);
        reply = response ? ae_json_string_field(response, "content") : NULL;
        if (!reply) {
            reply = ae_str_dup("[llm error] HTTPS request failed (libcurl not found or network error)");
            mode = "error";
        }
        free(response);
#endif
    } else if (strncmp(url_env, "http://", 7) != 0) {
        reply = ae_str_dup("[llm error] unsupported URL scheme; use https:// or http://");
        mode = "error";
    } else {
        const char *rest = url_env + 7;
        const char *slash = strchr(rest, '/');
        char host[256] = {0};
        int port = 80;
        const char *path = "/";
        if (slash) {
            size_t span = (size_t)(slash - rest);
            if (span >= sizeof(host)) span = sizeof(host) - 1;
            memcpy(host, rest, span);
            path = slash;
        } else {
            snprintf(host, sizeof(host), "%s", rest);
        }
        char *colon = strchr(host, ':');
        if (colon) { *colon = 0; port = atoi(colon + 1); }

        char *escaped_prompt = ae_json_escape_heap(prompt);
        char body[8192];
        snprintf(body, sizeof(body),
                 "{\"model\":\"%s\",\"messages\":[{\"role\":\"user\",\"content\":\"%s\"}]}",
                 model, escaped_prompt);
        free(escaped_prompt);
        char request[12288];
        snprintf(request, sizeof(request),
                 "POST %s HTTP/1.1\r\nHost: %s\r\nContent-Type: application/json\r\n"
                 "Content-Length: %d\r\nConnection: close\r\n\r\n%s",
                 path, host, (int)strlen(body), body);

        ae_net_once();
        struct addrinfo hints, *res = NULL;
        char port_str[8];
        snprintf(port_str, sizeof(port_str), "%d", port);
        memset(&hints, 0, sizeof(hints));
        hints.ai_family = AF_UNSPEC;
        hints.ai_socktype = SOCK_STREAM;
        if (getaddrinfo(host, port_str, &hints, &res) != 0 || !res) {
            reply = ae_str_dup("[llm error] cannot resolve host");
            mode = "error";
        } else {
            ae_socket_t fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
            if (fd == AE_INVALID_SOCKET || connect(fd, res->ai_addr, (int)res->ai_addrlen) != 0) {
                reply = ae_str_dup("[llm error] cannot connect (is the server running?)");
                mode = "error";
                if (fd != AE_INVALID_SOCKET) AE_CLOSE_SOCKET(fd);
            } else {
                ae_send_all(fd, request, strlen(request));
                size_t cap = 262144, got = 0;
                char *buffer = (char *)ae_alloc(cap);
                for (;;) {
                    if (got + 1 >= cap) break;
                    int n = ae_recv(fd, buffer + got, (int)(cap - 1 - got));
                    if (n <= 0) break;
                    got += (size_t)n;
                    buffer[got] = 0;
                }
                AE_CLOSE_SOCKET(fd);
                char *payload = strstr(buffer, "\r\n\r\n");
                reply = payload ? ae_json_string_field(payload + 4, "content") : NULL;
                if (!reply) {
                    reply = ae_str_dup("[llm error] no completion in response");
                    mode = "error";
                }
                free(buffer);
            }
            freeaddrinfo(res);
        }
    }

    double ms = ae_now_ms() - started;
    ae_event("{\"event\":\"ask\",\"model\":\"%s\",\"prompt_chars\":%d,\"reply_chars\":%d,\"mode\":\"%s\",\"ms\":%.6f,\"backend\":\"c\"}",
             model, (int)strlen(prompt), reply ? (int)strlen(reply) : 0, mode, ms);
    ae_human("  [llm]    %s | %d chars in | %d chars out | %s | %.1f ms",
             model, (int)strlen(prompt), reply ? (int)strlen(reply) : 0, mode, ms);
    if (!reply) reply = ae_str_dup("");
    return reply;
}

/* ---------------------------------------------------------------- sys.core */
static char *ae_read_file(const char *path) {
    char resolved[1200];
    ae_resolve_path(path, resolved, sizeof(resolved));
    FILE *handle = fopen(resolved, "rb");
    if (!handle) {
        char *message = (char *)ae_alloc(strlen(resolved) + 64);
        snprintf(message, strlen(resolved) + 64, "[read error] cannot open %s", resolved);
        return message;
    }
    fseek(handle, 0, SEEK_END);
    long size = ftell(handle);
    fseek(handle, 0, SEEK_SET);
    if (size < 0) size = 0;
    char *content = (char *)ae_alloc((size_t)size + 1);
    size_t got = fread(content, 1, (size_t)size, handle);
    content[got] = 0;
    fclose(handle);
    char *escaped = ae_json_escape_heap(resolved);
    ae_event("{\"event\":\"read_file\",\"path\":\"%s\",\"chars\":%lld,\"backend\":\"c\"}",
             escaped, (long long)got);
    ae_human("  [read]   %s | %lld chars", resolved, (long long)got);
    free(escaped);
    return content;
}

static long long ae_write_file(const char *path, const char *content) {
    char resolved[1200];
    ae_resolve_path(path, resolved, sizeof(resolved));
    FILE *handle = fopen(resolved, "wb");
    if (!handle) {
        fprintf(stderr, "aether: cannot write %s\n", resolved);
        exit(3);
    }
    size_t len = strlen(content);
    fwrite(content, 1, len, handle);
    fclose(handle);
    char *escaped = ae_json_escape_heap(resolved);
    ae_event("{\"event\":\"write_file\",\"path\":\"%s\",\"bytes\":%lld,\"backend\":\"c\"}",
             escaped, (long long)len);
    ae_human("  [write]  %s | %lld bytes", resolved, (long long)len);
    free(escaped);
    return (long long)len;
}

/* Numeric lookup in a flat JSON document: with a key, reads "key": <number>;
 * without, parses the document as a bare number. */
static double ae_load_json_num(const char *path, const char *key) {
    char resolved[1200];
    ae_resolve_path(path, resolved, sizeof(resolved));
    FILE *handle = fopen(resolved, "rb");
    if (!handle) {
        fprintf(stderr, "aether: cannot open %s\n", resolved);
        exit(3);
    }
    static char buffer[65536];
    size_t got = fread(buffer, 1, sizeof(buffer) - 1, handle);
    fclose(handle);
    buffer[got] = 0;

    double value = 0.0;
    if (key && key[0]) {
        char pattern[256];
        snprintf(pattern, sizeof(pattern), "\"%s\"", key);
        const char *cursor = strstr(buffer, pattern);
        if (cursor) {
            cursor += strlen(pattern);
            while (*cursor == ' ' || *cursor == ':') cursor++;
            char *end = NULL;
            value = strtod(cursor, &end);
            if (end == cursor) {
                fprintf(stderr, "aether: %s has no numeric field '%s'\n", resolved, key);
                exit(3);
            }
        } else {
            fprintf(stderr, "aether: %s has no field '%s'\n", resolved, key);
            exit(3);
        }
    } else {
        char *end = NULL;
        value = strtod(buffer, &end);
        if (end == buffer) {
            fprintf(stderr, "aether: %s is not a bare JSON number\n", resolved);
            exit(3);
        }
    }

    char *escaped_path = ae_json_escape_heap(resolved);
    char *escaped_key = key ? ae_json_escape_heap(key) : ae_str_dup("");
    ae_event("{\"event\":\"load_json\",\"path\":\"%s\",\"key\":\"%s\",\"value\":%.17g,\"backend\":\"c\"}",
             escaped_path, escaped_key, value);
    ae_human("  [json]   %s%s%s = %.6g", resolved, key ? "['" : "", key ? key : "",
             key ? "']" : "", value);
    return value;
}

/* -------------------------------------------------------------- planning */
static char *ae_plan(const char *goal, int max_steps) {
    const char *mock_env = getenv("AETHER_LLM_MOCK");
    const char *url_env = getenv("AETHER_LLM_URL");
    const char *key_env = getenv("AETHER_LLM_KEY");
    int mock = (mock_env && strcmp(mock_env, "1") == 0) || !(url_env && key_env);
    const char *mode = (mock_env && strcmp(mock_env, "1") == 0) ? "mock" : "auto-mock";
    char *reply = NULL;

    if (mock) {
        static const char *templates[] = {
            "Parse the goal and constraints",
            "Gather the data and context",
            "Execute the core analysis",
            "Validate the results",
            "Summarize the findings",
        };
        const int template_count = 5;
        if (max_steps < 1) max_steps = 1;
        if (max_steps > template_count) max_steps = template_count;
        size_t cap = strlen(goal) + (size_t)max_steps * 96 + 16;
        reply = (char *)ae_alloc(cap);
        size_t offset = 0;
        for (int i = 0; i < max_steps; i++) {
            offset += (size_t)snprintf(reply + offset, cap - offset, "%d. %s%s",
                                       i + 1, templates[i], i + 1 < max_steps ? "\n" : "");
        }
        mode = (mock_env && strcmp(mock_env, "1") == 0) ? "mock" : "auto-mock";
    } else {
        char prompt[4096];
        snprintf(prompt, sizeof(prompt),
                 "Break this goal into at most %d numbered steps. Reply with the "
                 "numbered list only.\nGoal: %s", max_steps, goal);
        reply = ae_llm_ask(prompt, "gpt-4o-mini");
        mode = "llm";
    }

    ae_event("{\"event\":\"plan\",\"goal_chars\":%d,\"max_steps\":%d,\"mode\":\"%s\",\"backend\":\"c\"}",
             (int)strlen(goal), max_steps, mode);
    ae_human("  [plan]   %d step(s) | %s", max_steps, mode);
    return reply;
}

/* ------------------------------------------------------------- AEB1 loading */
static Model ae_model_load(const char *path) {
    char resolved[1200];
    ae_resolve_path(path, resolved, sizeof(resolved));
    FILE *handle = fopen(resolved, "rb");
    if (!handle) {
        fprintf(stderr, "aether: cannot open model %s\n", resolved);
        exit(3);
    }
    uint32_t header[8];
    if (fread(header, sizeof(uint32_t), 8, handle) != 8 ||
        header[0] != 0x31424541u || header[1] != 1u) {
        fprintf(stderr, "aether: %s is not an AEB1 v1 model file\n", resolved);
        exit(3);
    }

    Model m;
    memset(&m, 0, sizeof(m));
    m.kind = (int)header[2];
    m.in_dim = (int)header[3];
    m.hidden = (int)header[4];
    m.out_dim = (int)header[5];
    m.classes = m.kind ? (int)header[6] : 0;
    m.trained = 1;
    size_t nw1 = (size_t)m.in_dim * m.hidden, nb1 = (size_t)m.hidden;
    size_t nw2 = (size_t)m.hidden * m.out_dim, nb2 = (size_t)m.out_dim;
    m.w1 = ae_falloc(nw1);
    m.b1 = ae_falloc(nb1);
    m.w2 = ae_falloc(nw2);
    m.b2 = ae_falloc(nb2);
    if (fread(m.w1, sizeof(float), nw1, handle) != nw1 ||
        fread(m.b1, sizeof(float), nb1, handle) != nb1 ||
        fread(m.w2, sizeof(float), nw2, handle) != nw2 ||
        fread(m.b2, sizeof(float), nb2, handle) != nb2) {
        fprintf(stderr, "aether: %s is truncated\n", resolved);
        exit(3);
    }
    fclose(handle);

    char escaped[2400];
    ae_json_escape(resolved, escaped, sizeof(escaped));
    int params = m.in_dim * m.hidden + m.hidden + m.hidden * m.out_dim + m.out_dim;
    ae_event("{\"event\":\"load\",\"path\":\"%s\",\"in_dim\":%d,\"hidden\":%d,\"out_dim\":%d,\"params\":%d,\"kind\":\"%s\",\"backend\":\"c\"}",
             escaped, m.in_dim, m.hidden, m.out_dim, params, m.kind ? "classifier" : "mlp");
    ae_human("  [model]  loaded %s | %d -> %d -> %d | %d params | AEB1",
             resolved, m.in_dim, m.hidden, m.out_dim, params);
    return m;
}

/* ------------------------------------------------------ native HTTP server */
static volatile int ae_shutdown = 0;

#ifdef _WIN32
static BOOL WINAPI ae_ctrl_handler(DWORD type) {
    (void)type;
    ae_shutdown = 1;
    return TRUE;
}
#else
static void ae_on_sigint(int sig) {
    (void)sig;
    ae_shutdown = 1;
}
#endif

static void ae_http_respond(ae_socket_t fd, int status, const char *json) {
    const char *reason = status == 200 ? "OK"
                       : status == 400 ? "Bad Request"
                       : status == 404 ? "Not Found" : "Error";
    char header[256];
    int n = snprintf(header, sizeof(header),
                     "HTTP/1.1 %d %s\r\n"
                     "Content-Type: application/json\r\n"
                     "Content-Length: %d\r\n"
                     "Connection: close\r\n\r\n",
                     status, reason, (int)strlen(json));
    ae_send_all(fd, header, (size_t)n);
    ae_send_all(fd, json, strlen(json));
}

/* Extracts a float array from {"features": [...]} or a bare [...] body. */
static int ae_json_extract_features(const char *body, float *out, int max) {
    const char *cursor = strstr(body, "\"features\"");
    if (cursor) {
        cursor = strchr(cursor, '[');
    } else {
        cursor = strchr(body, '[');
    }
    if (!cursor) return -1;
    cursor++;
    int count = 0;
    while (*cursor && *cursor != ']') {
        char *end = NULL;
        float value = strtof(cursor, &end);
        if (end == cursor) {
            cursor++;
            continue;
        }
        if (count < max) out[count] = value;
        count++;
        cursor = end;
    }
    return count;
}

static int strncasecmpaliases(const char *a, const char *b, size_t n) {
#ifdef _WIN32
    return _strnicmp(a, b, n);
#else
    return strncasecmp(a, b, n);
#endif
}

static int ae_header_int(const char *headers, const char *name) {
    size_t name_len = strlen(name);
    const char *line = headers;
    while (line && *line) {
        const char *eol = strstr(line, "\r\n");
        size_t span = eol ? (size_t)(eol - line) : strlen(line);
        if (span > name_len && line[name_len] == ':' &&
            strncasecmpaliases(line, name, name_len) == 0) {
            return atoi(line + name_len + 1);
        }
        line = eol ? eol + 2 : NULL;
    }
    return 0;
}

typedef struct {
    ae_socket_t fd;
    Model *model;
    char route[128];
} AeConn;

static void ae_handle_connection(ae_socket_t fd, Model *model, const char *route);

#ifdef _WIN32
static unsigned __stdcall ae_conn_trampoline(void *arg) {
    AeConn *conn = (AeConn *)arg;
    ae_handle_connection(conn->fd, conn->model, conn->route);
    free(conn);
    return 0;
}
#else
static void *ae_conn_trampoline(void *arg) {
    AeConn *conn = (AeConn *)arg;
    ae_handle_connection(conn->fd, conn->model, conn->route);
    free(conn);
    return NULL;
}
#endif

static int ae_header_value(const char *headers, const char *name, char *out, size_t cap) {
    size_t name_len = strlen(name);
    const char *line = headers;
    while (line && *line) {
        const char *eol = strstr(line, "\r\n");
        size_t span = eol ? (size_t)(eol - line) : strlen(line);
        if (span > name_len && line[name_len] == ':' &&
            strncasecmpaliases(line, name, name_len) == 0) {
            const char *value = line + name_len + 1;
            while (*value == ' ') value++;
            size_t vlen = (eol ? (size_t)(eol - value) : strlen(value));
            if (vlen >= cap) vlen = cap - 1;
            memcpy(out, value, vlen);
            out[vlen] = 0;
            return 1;
        }
        line = eol ? eol + 2 : NULL;
    }
    return 0;
}

static int ae_authorized(const char *headers) {
    const char *expected = getenv("AETHER_API_KEY");
    if (!expected || !expected[0]) return 1;  /* auth disabled by default */
    char presented[512] = "";
    if (!ae_header_value(headers, "Authorization", presented, sizeof(presented))) return 0;
    const char *prefix = "Bearer ";
    if (strncmp(presented, prefix, strlen(prefix)) != 0) return 0;
    return strcmp(presented + strlen(prefix), expected) == 0 ? 1 : 0;
}

static void ae_handle_connection(ae_socket_t fd, Model *model, const char *route) {
    const size_t cap = AE_MAX_BODY + 65536;  /* hard 1 MiB body + header headroom */
    char *buf = (char *)malloc(cap);
    if (!buf) {
        AE_CLOSE_SOCKET(fd);
        return;
    }
    size_t got = 0;
    for (;;) {
        if (got + 1 >= cap) break;
        int n = ae_recv(fd, buf + got, (int)(cap - 1 - got));
        if (n <= 0) break;
        got += (size_t)n;
        buf[got] = 0;
        char *sep = strstr(buf, "\r\n\r\n");
        if (sep) {
            int content_length = ae_header_int(buf, "Content-Length");
            size_t head = (size_t)(sep + 4 - buf);
            if (content_length > AE_MAX_BODY) break;  /* rejected below */
            if (content_length <= 0 || got >= head + (size_t)content_length) break;
        }
    }
    if (!got) {
        free(buf);
        AE_CLOSE_SOCKET(fd);
        return;
    }
    buf[got] = 0;

    char method[16] = {0}, path[1024] = {0};
    sscanf(buf, "%15s %1023s", method, path);
    char *query = strchr(path, '?');
    if (query) *query = 0;

    /* /health stays open for liveness probes; auth protects data routes. */
    int is_health = strcmp(method, "GET") == 0 &&
                    (strcmp(path, "/health") == 0 || strcmp(path, "/") == 0);
    if (!is_health && !ae_authorized(buf)) {
        ae_http_respond(fd, 401, "{\"error\":\"unauthorized: set the Authorization: Bearer header\"}");
        ae_event("{\"event\":\"request\",\"method\":\"%s\",\"path\":\"%s\",\"status\":401,\"backend\":\"c\"}",
                 method, path);
        free(buf);
        AE_CLOSE_SOCKET(fd);
        return;
    }
    {
        char *body_start = strstr(buf, "\r\n\r\n");
        int content_length = ae_header_int(buf, "Content-Length");
        if (content_length > AE_MAX_BODY) {
            ae_http_respond(fd, 413, "{\"error\":\"request body exceeds the 1 MiB limit\"}");
            ae_event("{\"event\":\"request\",\"method\":\"%s\",\"path\":\"%s\",\"status\":413,\"backend\":\"c\"}",
                     method, path);
            free(buf);
            AE_CLOSE_SOCKET(fd);
            return;
        }
        (void)body_start;
    }
    double request_started = ae_now_ms();

    if (strcmp(method, "GET") == 0 &&
        (strcmp(path, "/health") == 0 || strcmp(path, "/") == 0)) {
        char body[320];
        snprintf(body, sizeof(body),
                 "{\"status\":\"ok\",\"model\":\"served\",\"in_dim\":%d,\"out_dim\":%d,\"route\":\"%s\"}",
                 model->in_dim, model->out_dim, route);
        ae_http_respond(fd, 200, body);
        ae_event("{\"event\":\"request\",\"method\":\"GET\",\"path\":\"%s\",\"status\":200,\"ms\":%.6f,\"backend\":\"c\"}",
                 path, ae_now_ms() - request_started);
    } else if (strcmp(method, "POST") == 0 && strcmp(path, route) == 0) {
        char *body = strstr(buf, "\r\n\r\n");
        float *features = ae_falloc((size_t)model->in_dim);
        int count = body ? ae_json_extract_features(body + 4, features, model->in_dim) : -1;
        if (count != model->in_dim) {
            char body_out[160];
            snprintf(body_out, sizeof(body_out),
                     "{\"error\":\"features must be an array of %d numbers\",\"received\":%d}",
                     model->in_dim, count);
            ae_http_respond(fd, 400, body_out);
            ae_event("{\"event\":\"request\",\"method\":\"POST\",\"path\":\"%s\",\"status\":400,\"ms\":%.6f,\"backend\":\"c\"}",
                     path, ae_now_ms() - request_started);
        } else {
            float *hidden = ae_falloc((size_t)model->hidden);
            float *logits = ae_falloc((size_t)model->out_dim);
            ae_forward(model, features, 1, hidden, logits);
            ae_softmax(logits, 1, model->out_dim);
            int best = 0;
            for (int j = 1; j < model->out_dim; j++) {
                if (logits[j] > logits[best]) best = j;
            }
            size_t out_cap = (size_t)model->out_dim * 20 + 512;
            char *response = (char *)malloc(out_cap);
            if (response) {
                size_t offset = (size_t)snprintf(response, out_cap,
                    "{\"prediction\":%d,\"confidence\":%.6f,\"probabilities\":[",
                    best, (double)logits[best]);
                for (int j = 0; j < model->out_dim; j++) {
                    offset += (size_t)snprintf(response + offset, out_cap - offset,
                                               "%s%.6f", j ? "," : "", (double)logits[j]);
                }
                snprintf(response + offset, out_cap - offset,
                         "],\"in_dim\":%d,\"out_dim\":%d,\"elapsed_ms\":%.6f}",
                         model->in_dim, model->out_dim, ae_now_ms() - request_started);
                ae_http_respond(fd, 200, response);
                ae_event("{\"event\":\"request\",\"method\":\"POST\",\"path\":\"%s\",\"status\":200,\"prediction\":%d,\"confidence\":%.6f,\"ms\":%.6f,\"backend\":\"c\"}",
                         path, best, (double)logits[best], ae_now_ms() - request_started);
                ae_human("  [request] POST %s | %d features | pred %d (%.3f) | %.2f ms",
                         path, model->in_dim, best, (double)logits[best], ae_now_ms() - request_started);
                free(response);
            }
            free(hidden);
            free(logits);
        }
        free(features);
    } else {
        ae_http_respond(fd, 404, "{\"error\":\"not found\"}");
        ae_event("{\"event\":\"request\",\"method\":\"%s\",\"path\":\"%s\",\"status\":404,\"ms\":%.6f,\"backend\":\"c\"}",
                 method, path, ae_now_ms() - request_started);
    }

    free(buf);
    AE_CLOSE_SOCKET(fd);
}

static void ae_http_serve(Model *model, int port, const char *route) {
    const char *port_env = getenv("AETHER_PORT");
    if (port_env && atoi(port_env) > 0) port = atoi(port_env);
    int serve_seconds = 0;
    const char *seconds = getenv("AETHER_SERVE_SECONDS");
    if (seconds && atoi(seconds) > 0) serve_seconds = atoi(seconds);

#ifdef _WIN32
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        fprintf(stderr, "aether: WSAStartup failed\n");
        exit(3);
    }
    SetConsoleCtrlHandler(ae_ctrl_handler, TRUE);
#else
    signal(SIGPIPE, SIG_IGN);
    signal(SIGINT, ae_on_sigint);
#endif

    ae_socket_t listener = socket(AF_INET, SOCK_STREAM, 0);
    if (listener == AE_INVALID_SOCKET) {
        fprintf(stderr, "aether: socket() failed\n");
        exit(3);
    }
    int one = 1;
    setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, (const char *)&one, sizeof(one));
    struct sockaddr_in address;
    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_ANY);
    address.sin_port = htons((unsigned short)port);
    if (bind(listener, (struct sockaddr *)&address, sizeof(address)) != 0) {
        fprintf(stderr, "aether: cannot bind port %d\n", port);
        exit(3);
    }
    if (listen(listener, 16) != 0) {
        fprintf(stderr, "aether: listen() failed\n");
        exit(3);
    }

    ae_event("{\"event\":\"serve_start\",\"port\":%d,\"route\":\"%s\",\"in_dim\":%d,\"out_dim\":%d,\"backend\":\"c\"}",
             port, route, model->in_dim, model->out_dim);
    ae_human("  [serve]  http://127.0.0.1:%d%s | POST {\"features\":[...]} | GET /health",
             port, route);

    time_t deadline = serve_seconds > 0 ? time(NULL) + serve_seconds : 0;
    long requests = 0;
    while (!ae_shutdown) {
        struct timeval tv;
        tv.tv_sec = 0;
        tv.tv_usec = 200 * 1000;
        fd_set readers;
        FD_ZERO(&readers);
        FD_SET(listener, &readers);
        int ready = select((int)listener + 1, &readers, NULL, NULL, &tv);
        if (ae_shutdown) break;
        if (deadline && time(NULL) >= deadline) break;
        if (ready <= 0 || !FD_ISSET(listener, &readers)) continue;
        ae_socket_t client = accept(listener, NULL, NULL);
        if (client == AE_INVALID_SOCKET) continue;
        AeConn *conn = (AeConn *)malloc(sizeof(AeConn));
        if (!conn) {
            AE_CLOSE_SOCKET(client);
            continue;
        }
        conn->fd = client;
        conn->model = model;
        snprintf(conn->route, sizeof(conn->route), "%s", route);
#ifdef _WIN32
        uintptr_t thread = _beginthreadex(NULL, 0, ae_conn_trampoline, conn, 0, NULL);
        if (thread) {
            CloseHandle((HANDLE)thread);
        } else {
            free(conn);
        }
#else
        pthread_t thread;
        if (pthread_create(&thread, NULL, ae_conn_trampoline, conn) == 0) {
            pthread_detach(thread);
        } else {
            free(conn);
        }
#endif
        requests++;
    }

    AE_CLOSE_SOCKET(listener);
#ifdef _WIN32
    WSACleanup();
#endif
    ae_event("{\"event\":\"serve_stop\",\"port\":%d,\"requests\":%ld,\"backend\":\"c\"}", port, requests);
    ae_human("  [serve]  stopped | %ld request(s) served", requests);
}
'''


def _c_string(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _json_string(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


def _bind(call: Call, names: list[str]) -> dict[str, object]:
    bound: dict[str, object] = {}
    position = 0
    for arg in call.args:
        if arg.name is None:
            if position < len(names):
                bound[names[position]] = arg.value
                position += 1
        else:
            bound[arg.name] = arg.value
    return bound


def _int(value: object, default: int | None = None) -> int:
    if isinstance(value, Literal):
        return int(value.value)
    if default is not None:
        return default
    raise CBackendError("native backend requires integer literals for dimensions")


def _policy_speed(task: Task, call: Call | None = None) -> str:
    if call is not None:
        for mod in call.modifiers:
            if mod.name == "optimize":
                for arg in mod.args:
                    if arg.name == "speed":
                        if isinstance(arg.value, Literal) and isinstance(arg.value.value, str):
                            return arg.value.value
                        if isinstance(arg.value, Name):
                            return arg.value.id
    if task.modifier is not None and task.modifier.name == "optimize":
        for arg in task.modifier.args:
            if arg.name == "speed":
                if isinstance(arg.value, Literal) and isinstance(arg.value.value, str):
                    return arg.value.value
                if isinstance(arg.value, Name):
                    return arg.value.id
    return "balanced"


def _iter_statements(body: list[Stmt]):
    for stmt in body:
        yield stmt
        if stmt.body:
            yield from _iter_statements(stmt.body)
        if stmt.else_body:
            yield from _iter_statements(stmt.else_body)


def _types(task: Task) -> dict[str, tuple[str, str]]:
    """Map variable -> (kind, detail) where detail is 'model:classifier' etc."""
    known: dict[str, tuple[str, str]] = {}
    for stmt in _iter_statements(task.body):
        if stmt.kind != "assign" or not isinstance(stmt.expr, Call):
            continue
        call = stmt.expr
        if call.name == "tensor":
            known[stmt.name or ""] = ("tensor", "")
        elif call.name == "load_data":
            known[stmt.name or ""] = ("dataset", "")
        elif call.name == "load_csv":
            known[stmt.name or ""] = ("dataset", "")
        elif call.name == "neural_net":
            known[stmt.name or ""] = ("model", "mlp")
        elif call.name == "classifier":
            known[stmt.name or ""] = ("model", "classifier")
        elif call.name == "load_model":
            known[stmt.name or ""] = ("model", "served")
        elif call.name == "run":
            known[stmt.name or ""] = ("stream", "")
    return known


def _scalar_c_type(expr: object, strings: set[str]) -> str:
    if isinstance(expr, Literal) and isinstance(expr.value, str):
        return "const char *"
    if isinstance(expr, Literal) and isinstance(expr.value, bool):
        return "int"
    if isinstance(expr, Literal) and isinstance(expr.value, int):
        return "long long"
    if isinstance(expr, Binary):
        if expr.op in ("==", "!=", ">", "<", ">=", "<="):
            return "int"
        if expr.op == "%":
            return "long long"
        if isinstance(expr.left, Literal) and isinstance(expr.left.value, str):
            return "const char *"
        return "double"
    if isinstance(expr, Name):
        return "double"
    return "double"


def _run_expr(call: Call, known: dict[str, tuple[str, str]], chunks: int) -> str:
    bound = _bind(call, ["data"])
    data = bound.get("data")
    data_name = data.id if isinstance(data, Name) else "NULL"
    target = call.target or "NULL"
    kind, detail = known.get(target, ("model", "mlp"))
    data_kind, _ = known.get(data_name, ("tensor", ""))
    if detail == "classifier" and data_kind == "dataset":
        return f"ae_stream_classifier(&{target}, &{data_name}, {chunks})"
    return f"ae_stream(&{target}, &{data_name}, {chunks})"


def _is_str_expr(node: object, known: dict[str, tuple[str, str]], strings: set[str]) -> bool:
    if isinstance(node, Literal):
        return isinstance(node.value, str)
    if isinstance(node, Name):
        return node.id in strings
    if isinstance(node, Call):
        if node.name in ("ask_llm", "str", "plan", "read_file"):
            return True
        return node.name in _FN_RETS and _FN_RETS[node.name] == "str"
    if isinstance(node, Binary) and node.op == "+":
        return _is_str_expr(node.left, known, strings) or _is_str_expr(node.right, known, strings)
    return False


def _is_tensor_expr(node: object, known: dict[str, tuple[str, str]]) -> bool:
    if isinstance(node, Name):
        return known.get(node.id, ("", ""))[0] == "tensor"
    if isinstance(node, Binary) and node.op == "*":
        return _is_tensor_expr(node.left, known) or _is_tensor_expr(node.right, known)
    return False


def _expr_c(
    node: object,
    known: dict[str, tuple[str, str]],
    chunks: int,
    strings: set[str] | None = None,
) -> str:
    strings = strings if strings is not None else set()
    if isinstance(node, Binary):
        left = _expr_c(node.left, known, chunks, strings)
        right = _expr_c(node.right, known, chunks, strings)
        if node.op == "+":
            if _is_str_expr(node, known, strings):
                return f"ae_str_concat({left}, {right})"
            return f"(({left}) + ({right}))"
        if node.op == "*":
            if _is_tensor_expr(node.left, known):
                return f"ae_tensor_scaled(&({left}), (double)({right}))"
            if _is_tensor_expr(node.right, known):
                return f"ae_tensor_scaled(&({right}), (double)({left}))"
        if node.op == "%":
            return f"((long long)({left}) % (long long)({right}))"
        if node.op == "/":
            # Aether '/' is true division on both backends.
            return f"((double)({left}) / (double)({right}))"
        return f"(({left}) {node.op} ({right}))"
    if isinstance(node, Unary):
        return f"(-{_expr_c(node.operand, known, chunks, strings)})"
    if isinstance(node, Literal):
        if isinstance(node.value, str):
            return f'"{_c_string(node.value)}"'
        if isinstance(node.value, float):
            return repr(node.value)
        return str(node.value)
    if isinstance(node, Name):
        return node.id
    if not isinstance(node, Call):
        return "0"
    if node.name == "tensor":
        bound = _bind(node, ["rows", "cols"])
        return f"ae_tensor_new({_int(bound.get('rows'))}, {_int(bound.get('cols'))})"
    if node.name == "load_data":
        bound = _bind(node, ["samples", "features", "classes"])
        return (
            f"ae_dataset_new({_int(bound.get('samples'))}, {_int(bound.get('features'))}, "
            f"{_int(bound.get('classes'), 10)})"
        )
    if node.name == "neural_net":
        bound = _bind(node, ["in", "out", "hidden"])
        input_name = bound.get("in")
        width = _known_width(input_name, known)
        return (
            f"ae_model_new({width}, {_int(bound.get('hidden'), 512)}, "
            f"{_int(bound.get('out'))}, 0)"
        )
    if node.name == "classifier":
        bound = _bind(node, ["in", "classes", "hidden"])
        return (
            f"ae_model_new({_int(bound.get('in'))}, {_int(bound.get('hidden'), 128)}, "
            f"{_int(bound.get('classes'))}, 1)"
        )
    if node.name == "load_csv":
        bound = _bind(node, ["path", "target", "classes"])
        path_node = bound.get("path")
        target_node = bound.get("target")
        path = (
            path_node.value
            if isinstance(path_node, Literal) and isinstance(path_node.value, str)
            else "data.csv"
        )
        target = (
            target_node.value
            if isinstance(target_node, Literal) and isinstance(target_node.value, str)
            else "target"
        )
        return (
            f'ae_load_csv("{_c_string(path.replace(chr(92), "/"))}", '
            f'"{_c_string(target)}")'
        )
    if node.name == "load_model":
        bound = _bind(node, ["path"])
        node_path = bound.get("path")
        text = (
            node_path.value
            if isinstance(node_path, Literal) and isinstance(node_path.value, str)
            else "model.ae.bin"
        )
        return f'ae_model_load("{_c_string(text)}")'
    if node.name == "save_json":
        bound = _bind(node, ["value", "path"])
        value = bound.get("value")
        path_node = bound.get("path")
        path = (
            path_node.value
            if isinstance(path_node, Literal) and isinstance(path_node.value, str)
            else "report.json"
        )
        path = _c_string(path.replace("\\", "/"))
        if isinstance(value, Name):
            kind, _detail = known.get(value.id, ("value", ""))
            if kind == "stream":
                return f'ae_save_json_stream(&{value.id}, "{path}")'
            if kind == "model":
                return f'ae_save_json_model(&{value.id}, "{path}")'
            return f'ae_save_json_scalar("{_c_string(value.id)}", (double)({value.id}), "{path}")'
        if isinstance(value, Literal):
            literal = (
                f'"{_c_string(str(value.value))}"'
                if isinstance(value.value, str)
                else str(value.value)
            )
            return f'ae_save_json_scalar("value", (double)({literal}), "{path}")'
        return 'ae_save_json_scalar("value", 0.0, "")'
    if node.name == "save_json":
        bound = _bind(node, ["value", "path"])
        value = bound.get("value")
        path_node = bound.get("path")
        path = (
            path_node.value
            if isinstance(path_node, Literal) and isinstance(path_node.value, str)
            else "report.json"
        )
        path = _c_string(path.replace("\\", "/"))
        if isinstance(value, Name):
            kind, _detail = known.get(value.id, ("value", ""))
            if kind == "stream":
                return f'ae_save_json_stream(&{value.id}, "{path}")'
            if kind == "model":
                return f'ae_save_json_model(&{value.id}, "{path}")'
            return f'ae_save_json_scalar("{_c_string(value.id)}", (double)({value.id}), "{path}")'
        if isinstance(value, Literal):
            literal = (
                f'"{_c_string(str(value.value))}"'
                if isinstance(value.value, str)
                else str(value.value)
            )
            return f'ae_save_json_scalar("value", (double)({literal}), "{path}")'
        return 'ae_save_json_scalar("value", 0.0, "")'
    if node.name == "remember":
        bound = _bind(node, ["key", "value"])
        key = (
            bound.get("key").value
            if isinstance(bound.get("key"), Literal) and isinstance(bound.get("key").value, str)
            else "key"
        )
        return (
            f'ae_memory_remember("{_c_string(key)}", '
            f"(double)({_expr_c(bound.get('value'), known, chunks)}), \"{_MEMORY_FILE_C}\")"
        )
    if node.name == "recall":
        bound = _bind(node, ["key", "default"])
        key = (
            bound.get("key").value
            if isinstance(bound.get("key"), Literal) and isinstance(bound.get("key").value, str)
            else "key"
        )
        default = (
            _expr_c(bound.get("default"), known, chunks) if "default" in bound else "0.0"
        )
        return (
            f'ae_memory_recall("{_c_string(key)}", (double)({default}), "{_MEMORY_FILE_C}")'
        )
    if node.name in ("mean", "sum", "min", "max"):
        bound = _bind(node, ["value"])
        target = bound.get("value")
        if isinstance(target, Name):
            return f"ae_tensor_{node.name}(&{target.id})"
        return f"ae_tensor_{node.name}_expr({_expr_c(target, known, chunks, strings)})"
    if node.name == "argmax":
        bound = _bind(node, ["value"])
        target = bound.get("value")
        if isinstance(target, Name):
            return f"ae_tensor_argmax(&{target.id})"
        return f"ae_tensor_argmax_expr({_expr_c(target, known, chunks, strings)})"
    if node.name == "str":
        bound = _bind(node, ["value"])
        return f"ae_str_of((double)({_expr_c(bound.get('value'), known, chunks, strings)}))"
    if node.name == "ask_llm":
        bound = _bind(node, ["prompt"])
        model = "gpt-4o-mini"
        for mod in node.modifiers:
            if mod.name == "model" and mod.args:
                value = mod.args[0].value
                if isinstance(value, Literal) and isinstance(value.value, str):
                    model = value.value
                    break
        prompt = _expr_c(bound.get("prompt"), known, chunks, strings)
        return f'ae_llm_ask({prompt}, "{_c_string(model)}")'
    if node.name == "plan":
        bound = _bind(node, ["goal"])
        max_steps = 3
        for mod in node.modifiers:
            if mod.name == "max_steps" and mod.args:
                value = mod.args[0].value
                if isinstance(value, Literal) and isinstance(value.value, int):
                    max_steps = value.value
                    break
        goal = _expr_c(bound.get("goal"), known, chunks, strings)
        return f"ae_plan({goal}, {max_steps})"
    if node.name == "read_file":
        bound = _bind(node, ["path"])
        path_node = bound.get("path")
        text = (
            path_node.value
            if isinstance(path_node, Literal) and isinstance(path_node.value, str)
            else "input.txt"
        )
        return f'ae_read_file("{_c_string(text)}")'
    if node.name == "write_file":
        bound = _bind(node, ["path", "content"])
        path_node = bound.get("path")
        text = (
            path_node.value
            if isinstance(path_node, Literal) and isinstance(path_node.value, str)
            else "out.txt"
        )
        content_node = bound.get("content")
        if isinstance(content_node, Literal) and isinstance(content_node.value, str):
            content = f'"{_c_string(content_node.value)}"'
        elif isinstance(content_node, Name) and content_node.id in strings:
            content = content_node.id
        else:
            content = f"ae_str_of((double)({_expr_c(content_node, known, chunks, strings)}))"
        return f'ae_write_file("{_c_string(text)}", {content})'
    if node.name == "load_json":
        bound = _bind(node, ["path", "key"])
        path_node = bound.get("path")
        key_node = bound.get("key")
        text = (
            path_node.value
            if isinstance(path_node, Literal) and isinstance(path_node.value, str)
            else "data.json"
        )
        key = (
            f'"{_c_string(key_node.value)}"'
            if isinstance(key_node, Literal) and isinstance(key_node.value, str)
            else "NULL"
        )
        return f'ae_load_json_num("{_c_string(text)}", {key})'
    if node.name in _FNS:
        args = [_expr_c(arg.value, known, chunks, strings) for arg in node.args]
        return f"ae_fn_{node.name}({', '.join(args)})"
    if node.name == "run":
        return _run_expr(node, known, chunks)
    raise CBackendError(f"intent '{node.name}' is not supported by the C backend yet")


_WIDTHS: dict[str, int] = {}
_FNS: set[str] = set()
_FN_RETS: dict[str, str] = {}
_GUARD_ID: list[int] = [0]
_MEMORY_FILE_C: str = "agent.memory.json"


def _known_width(node: object, known: dict[str, tuple[str, str]]) -> int:
    if isinstance(node, Name):
        return _WIDTHS.get(node.id, 0)
    return 0


def _collect_scalars(
    body: list[Stmt],
    decls: dict[str, str],
    strings: set[str],
    known: dict[str, tuple[str, str]] | None = None,
) -> None:
    """Pre-declare every `let` scalar at function scope (C needs declarations)."""
    known = known or {}
    for stmt in _iter_statements(body):
        if stmt.kind == "train" and "accuracy" not in decls:
            decls["accuracy"] = "double"
        if stmt.kind != "let" or not stmt.name:
            continue
        if stmt.name in decls:
            continue
        if _is_str_expr(stmt.expr, known, strings):
            decls[stmt.name] = "const char *"
            strings.add(stmt.name)
            continue
        if _is_tensor_expr(stmt.expr, known):
            decls[stmt.name] = "Tensor"
            known[stmt.name] = ("tensor", "")
            continue
        ctype = _scalar_c_type(stmt.expr, strings)
        decls[stmt.name] = ctype
        if ctype.startswith("const char"):
            strings.add(stmt.name)


def _task_c(task: Task, chunks: int) -> list[str]:
    _WIDTHS.clear()
    _GUARD_ID[0] = 0
    known = _types(task)
    for stmt in _iter_statements(task.body):
        if stmt.kind == "assign" and isinstance(stmt.expr, Call):
            call = stmt.expr
            if call.name == "tensor":
                bound = _bind(call, ["rows", "cols"])
                _WIDTHS[stmt.name or ""] = _int(bound.get("cols"))
            elif call.name == "load_data":
                bound = _bind(call, ["samples", "features", "classes"])
                _WIDTHS[stmt.name or ""] = _int(bound.get("features"))
            elif call.name in ("neural_net", "classifier"):
                bound = _bind(call, ["in", "out", "classes", "hidden"])
                width = (
                    _int(bound.get("in"))
                    if call.name == "classifier"
                    else _known_width(bound.get("in"), known)
                )
                _WIDTHS[stmt.name or ""] = width

    decls: dict[str, str] = {}
    strings: set[str] = set()
    _collect_scalars(task.body, decls, strings, known)

    name = task.name
    lines = [
        f"static void task_{name}(void) {{",
        "    double task_started = ae_now_ms();",
    ]
    for scalar, ctype in decls.items():
        initializer = (
            '""' if ctype.startswith("const char")
            else "{0}" if ctype == "Tensor"
            else ("0.0" if ctype == "double" else "0")
        )
        lines.append(f"    {ctype} {scalar} = {initializer};")
    for stmt in task.body:
        lines.extend(_stmt_c(stmt, known, chunks, depth=0, strings=strings))
    if len(lines) == 2 + len(decls):
        lines.append("    (void)0;")
    lines.extend(
        [
            "    double task_ms = ae_now_ms() - task_started;",
            '    ae_event("{\\"event\\":\\"task_done\\",\\"task\\":\\"'
            + name
            + '\\",\\"ms\\":%.6f,\\"backend\\":\\"c\\"}", task_ms);',
            f'    ae_human("  [task]   {name} | %.1f ms", task_ms);',
            "}",
        ]
    )
    return lines


def _stmt_c(
    stmt: Stmt,
    known: dict[str, tuple[str, str]],
    chunks: int,
    depth: int = 0,
    strings: set[str] | None = None,
) -> list[str]:
    strings = strings or set()
    pad = "    " * (depth + 1)

    def render_inner(body: list[Stmt]) -> list[str]:
        lines: list[str] = []
        for inner in body:
            lines.extend(
                _stmt_c(inner, known, chunks, depth=depth + 1, strings=strings)
            )
        return lines or [pad + "    ;"]

    if stmt.kind == "assign":
        value = _expr_c(stmt.expr, known, chunks, strings)
        kind, _detail = known.get(stmt.name or "", ("value", ""))
        c_type = {
            "tensor": "Tensor",
            "dataset": "Dataset",
            "model": "Model",
            "stream": "StreamResult",
        }.get(kind, "double")
        return [f"{pad}{c_type} {stmt.name} = {value};"]

    if stmt.kind == "let":
        return [f"{pad}{stmt.name} = {_expr_c(stmt.expr, known, chunks, strings)};"]

    if stmt.kind == "repeat":
        count = _expr_c(stmt.count, known, chunks, strings)
        lines = [f"{pad}for (long long ae_i{depth} = 0; ae_i{depth} < (long long)({count}); ae_i{depth}++) {{"]
        lines.extend(render_inner(stmt.body))
        lines.append(f"{pad}}}")
        return lines

    if stmt.kind == "while":
        condition = _expr_c(stmt.expr, known, chunks, strings)
        max_iters = int(stmt.max_iters or 100_000)
        _GUARD_ID[0] += 1
        guard = f"ae_g{_GUARD_ID[0]}"
        lines = [f"{pad}long long {guard} = 0;"]
        lines.append(f"{pad}while ({condition}) {{")
        lines.append(f"{pad}    {guard}++;")
        lines.append(f"{pad}    if ({guard} > {max_iters}) {{ ae_while_guard({max_iters}); break; }}")
        lines.extend(render_inner(stmt.body))
        lines.append(f"{pad}}}")
        return lines

    if stmt.kind == "train":
        return [
            f"{pad}{{ int heals = 0, steps = 0;",
            f"{pad}  accuracy = ae_train_accuracy(&{stmt.target}, &{stmt.data}, {stmt.epochs}, 256, 0.1f, &heals, &steps); }}",
        ]

    if stmt.kind == "if":
        condition = _expr_c(stmt.expr, known, chunks, strings)
        lines = [f"{pad}if ({condition}) {{"]
        lines.extend(render_inner(stmt.body))
        lines.append(f"{pad}}} else {{")
        lines.extend(render_inner(stmt.else_body))
        lines.append(f"{pad}}}")
        return lines

    if stmt.kind == "log":
        expr = stmt.expr
        if isinstance(expr, Literal) and isinstance(expr.value, str):
            return [f'{pad}ae_log_text("{_c_string(expr.value)}");']
        if isinstance(expr, Name) and expr.id in strings:
            return [f'{pad}ae_log_text((const char *)({expr.id} ? {expr.id} : ""));']
        return [
            f"{pad}{{ char ae_buf{depth}[256]; snprintf(ae_buf{depth}, sizeof(ae_buf{depth}), \"%.6g\", (double)({_expr_c(expr, known, chunks, strings)})); ae_log_text(ae_buf{depth}); }}"
        ]

    if stmt.kind == "emit":
        expr = stmt.expr
        if isinstance(expr, Name) and expr.id in known:
            kind, _ = known[expr.id]
            emitter = {
                "tensor": "ae_emit_tensor",
                "stream": "ae_emit_stream",
            }.get(kind)
            if emitter:
                return [f"{pad}{emitter}(&{expr.id});" if kind == "tensor" else f"{pad}{emitter}({expr.id});"]
            return [f'{pad}ae_emit_text("{_c_string(expr.id)}");']
        if isinstance(expr, Literal):
            return [f'{pad}ae_emit_text("{_c_string(str(expr.value))}");']
        if isinstance(expr, Call) and expr.name == "run":
            return [f"{pad}{_run_expr(expr, known, chunks)};"]
        return [f"{pad}(void)0;"]

    if stmt.kind == "return":
        return [f"{pad}return {_expr_c(stmt.expr, known, chunks, strings)};"]

    if stmt.kind == "export":
        path = _c_string((stmt.path or "").replace("\\", "/"))
        return [f'{pad}ae_export(&{stmt.target}, "{path}");']

    if stmt.kind == "serve":
        route = "/predict"
        for mod in stmt.modifiers:
            if mod.name == "route" and mod.args:
                value = mod.args[0].value
                if isinstance(value, Literal) and isinstance(value.value, str):
                    route = value.value
                    break
        return [f'{pad}ae_http_serve(&{stmt.target}, {stmt.port or 8080}, "{_c_string(route)}");']

    if stmt.kind == "expr" and isinstance(stmt.expr, Call):
        return [f"{pad}{_expr_c(stmt.expr, known, chunks, strings)};"]

    return [f"{pad}(void)0;"]


_FN_RET_C = {"float": "double", "int": "long long", "bool": "int", "str": "const char *"}
_FN_FALLBACK = {"float": "0.0", "int": "0", "bool": "0", "str": '""'}


def _fn_c(function: Function) -> list[str]:
    ret = _FN_RET_C[function.ret_type]
    params = ", ".join(f"double {p}" for p in function.params)
    if not params:
        params = "void"
    decls: dict[str, str] = {}
    strings: set[str] = set()
    _collect_scalars(function.body, decls, strings)

    lines = [f"static {ret} ae_fn_{function.name}({params}) {{"]
    for scalar, ctype in decls.items():
        initializer = (
            '""' if ctype.startswith("const char")
            else "{0}" if ctype == "Tensor"
            else ("0.0" if ctype == "double" else "0")
        )
        lines.append(f"    {ctype} {scalar} = {initializer};")
    for stmt in function.body:
        lines.extend(_stmt_c(stmt, {}, chunks=8, depth=0, strings=strings))
    lines.append(f"    return {_FN_FALLBACK[function.ret_type]};  /* fall-through guard */")
    lines.append("}")
    return lines


def generate_c(
    program: Program,
    source_name: str,
    tasks: list[str] | None = None,
    chunks: int = 8,
) -> str:
    global _MEMORY_FILE_C
    _WIDTHS.clear()
    _FNS.clear()
    _FNS.update(fn.name for fn in program.functions)
    _FN_RETS.clear()
    _FN_RETS.update({fn.name: fn.ret_type for fn in program.functions})
    _MEMORY_FILE_C = _c_string(Path(source_name).stem + ".memory.json")
    source = _json_string(source_name.replace("\\", "/"))
    header = PREAMBLE.replace("__SOURCE__", source)
    out = [header]
    for fn in program.functions:
        out.append(f"static {_FN_RET_C[fn.ret_type]} ae_fn_{fn.name}(")
        params = ", ".join("double" for _ in fn.params) or "void"
        out.append(f"    {params});")
    out.append("")
    names = [task.name for task in program.tasks if not tasks or task.name in tasks]
    for fn in program.functions:
        out.extend(_fn_c(fn))
        out.append("")
    for task in program.tasks:
        if tasks and task.name not in tasks:
            continue
        out.extend(_task_c(task, chunks))
        out.append("")

    out.append("int main(int argc, char **argv) {")
    out.append("    (void)argc; (void)argv;")
    out.append('    ae_events = getenv("AETHER_EVENTS") != NULL;')
    out.append('    ae_silent = getenv("AETHER_SILENT") != NULL;')
    out.append('    const char *task = getenv("AETHER_TASK");')
    out.append('    if (task && task[0] == \'\\0\') task = NULL;')
    out.append('    {')
    out.append('        const char *base = getenv("AETHER_BASE_DIR");')
    out.append('        if (base && base[0]) {')
    out.append('            size_t n = strlen(base);')
    out.append('            if (n >= sizeof(ae_base_dir)) n = sizeof(ae_base_dir) - 1;')
    out.append('            memcpy(ae_base_dir, base, n);')
    out.append('            ae_base_dir[n] = 0;')
    out.append('        }')
    out.append('    }')
    out.append('    int threads = 1;')
    out.append("    #ifdef _OPENMP")
    out.append("    threads = omp_get_max_threads();")
    out.append("    #endif")
    out.append('    {')
    out.append('        const char *blas_env = getenv("OPENBLAS_NUM_THREADS");')
    out.append('        int configured = (blas_env && atoi(blas_env) > 0) ? atoi(blas_env) : threads;')
    out.append('        ae_max_threads = configured > 0 ? configured : 1;')
    out.append("    }")
    out.append("    #ifdef AETHER_OPENBLAS")
    out.append("    openblas_set_num_threads(ae_max_threads);")
    out.append("    ae_blas_current = ae_max_threads;")
    out.append("    #endif")
    out.append('    const char *tier = getenv("AETHER_DEVICE_TIER");')
    out.append('    if (!tier || !tier[0]) tier = "cpu";')
    out.append('    ae_event("{\\"event\\":\\"device\\",\\"device\\":\\"cpu\\",\\"machine\\":\\"native\\",\\"threads\\":%d,\\"compiler\\":\\"__CC__\\",\\"blas\\":\\"__BLAS__\\",\\"dtype\\":\\"f32\\",\\"tier\\":\\"%s\\",\\"backend\\":\\"c\\"}", threads, tier);')
    out.append('    ae_human("  [device] auto_hardware -> c/native | %d threads | __BLAS__ | %s tier | f32", threads, tier);')
    first = True
    for name in names:
        condition = f'(!task || strcmp(task, "{name}") == 0)' if first else f'(task && strcmp(task, "{name}") == 0)'
        out.append(f'    {"if" if first else "else if"} {condition} task_{name}();')
        first = False
    if names:
        out.append("    else {")
        out.append('        fprintf(stderr, "aether: unknown task \'%s\'\\n", task ? task : "");')
        out.append("        return 1;")
        out.append("    }")
    else:
        out.append("    (void)threads;")
    out.append("    return 0;")
    out.append("}")
    out.append("")
    return "\n".join(out)


def generate_for_toolchain(
    program: Program,
    source_name: str,
    toolchain: Toolchain,
    task_names: list[str] | None = None,
    chunks: int = 8,
) -> str:
    source = generate_c(program, source_name, task_names, chunks)
    compiler = _json_string(toolchain.version.replace("\\", "/")).replace('"', "'")
    blas = _json_string(toolchain.blas_name)
    return source.replace("__CC__", compiler).replace("__BLAS__", blas)


@dataclass
class BuildResult:
    source_file: Path
    binary: Path
    compile_ms: float
    command: str
    toolchain: Toolchain


def _copy_runtime_dll(toolchain: Toolchain, out_dir: Path) -> None:
    if not toolchain.has_blas or toolchain.bin_dir is None:
        return
    for pattern in ("*.dll", "*.so", "*.dylib"):
        for library in toolchain.bin_dir.glob(pattern):
            try:
                shutil.copy2(library, out_dir / library.name)
            except OSError:
                pass


def compile_c(
    source: str,
    out_dir: Path,
    stem: str,
    toolchain: Toolchain,
    threads: int = 16,
) -> BuildResult:
    import hashlib

    out_dir.mkdir(parents=True, exist_ok=True)
    source_file = out_dir / f"{stem}.c"
    source_file.write_text(source, encoding="utf-8", newline="\n")
    binary = out_dir / (f"{stem}.exe" if os.name == "nt" else stem)
    stamp = out_dir / f"{stem}.aebuild"

    # Compile cache: skip the compiler when the emitted source is unchanged.
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:24]
    if binary.is_file() and stamp.is_file() and stamp.read_text(encoding="utf-8").strip() == digest:
        return BuildResult(
            source_file=source_file,
            binary=binary,
            compile_ms=0.0,
            command="(cached: source unchanged)",
            toolchain=toolchain,
        )

    if toolchain.kind == "msvc":
        command = [
            toolchain.compiler,
            "/nologo",
            "/O2",
            "/openmp",
            "/std:c11",
            f"/Fe:{binary}",
            str(source_file),
        ]
        if toolchain.has_blas and toolchain.include_dir and toolchain.lib_dir:
            command += [
                "/DAETHER_OPENBLAS",
                f"/I{toolchain.include_dir}",
                str(toolchain.lib_dir / "libopenblas.lib"),
            ]
    else:
        command = [
            toolchain.compiler,
            "-O3",
            "-march=native",
            "-funroll-loops",
            "-fopenmp",
            "-std=gnu11",
            "-o",
            str(binary),
            str(source_file),
            "-lm",
        ]
        if toolchain.has_blas and toolchain.include_dir and toolchain.lib_dir:
            command += [
                "-DAETHER_OPENBLAS",
                f"-I{toolchain.include_dir}",
                f"-L{toolchain.lib_dir}",
                "-lopenblas",
            ]
        if os.name == "nt":
            command += ["-lws2_32"]

    started = time.perf_counter()
    proc = subprocess.run(command, capture_output=True, text=True, errors="replace")
    compile_ms = (time.perf_counter() - started) * 1000.0
    if proc.returncode != 0 or not binary.is_file():
        raise CBackendError(
            "native compilation failed\n"
            + " ".join(command)
            + "\n"
            + (proc.stdout or "")
            + (proc.stderr or "")
        )

    _copy_runtime_dll(toolchain, out_dir)
    stamp.write_text(digest, encoding="utf-8", newline="\n")
    return BuildResult(
        source_file=source_file,
        binary=binary,
        compile_ms=compile_ms,
        command=" ".join(command),
        toolchain=toolchain,
    )


HIGH_PRIORITY_CLASS = 0x00000080 if os.name == "nt" else 0


def run_binary(
    binary: Path,
    task: str | None,
    events: bool,
    silent: bool,
    timeout: float = 600.0,
    inherit: bool = False,
    high_priority: bool = False,
    base_dir: str | None = None,
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    if task:
        env["AETHER_TASK"] = task
    else:
        env.pop("AETHER_TASK", None)
    if base_dir:
        env["AETHER_BASE_DIR"] = base_dir
    else:
        env.pop("AETHER_BASE_DIR", None)
    if events:
        env["AETHER_EVENTS"] = "1"
    else:
        env.pop("AETHER_EVENTS", None)
    if silent:
        env["AETHER_SILENT"] = "1"
    else:
        env.pop("AETHER_SILENT", None)
    threads = env.get("AETHER_THREADS") or str(os.cpu_count() or 1)
    env.setdefault("OPENBLAS_NUM_THREADS", threads)
    env.setdefault("OMP_NUM_THREADS", threads)
    env["PATH"] = str(binary.parent) + os.pathsep + env.get("PATH", "")
    extra = {"creationflags": HIGH_PRIORITY_CLASS} if high_priority else {}
    return subprocess.run(
        [str(binary)],
        capture_output=not inherit,
        text=True,
        errors="replace",
        env=env,
        timeout=timeout,
        cwd=str(binary.parent),
        **extra,
    )


def parse_events(stderr: str) -> list[dict]:
    import json

    events: list[dict] = []
    for line in stderr.splitlines():
        line = line.strip()
        if not line.startswith("AETHER "):
            continue
        try:
            events.append(json.loads(line[len("AETHER ") :]))
        except ValueError:
            continue
    return events


# ---------------------------------------------------------------------- CUDA



def _cuda_plan(program: Program) -> dict:
    """Extract the training workload for the CUDA template (defaults if absent)."""
    plan = {
        "task": "Agent",
        "in_dim": 16,
        "hidden": 64,
        "out_dim": 4,
        "epochs": 5,
        "samples": 4096,
        "classes": 4,
    }
    for task in program.tasks:
        plan["task"] = task.name
        for stmt in _iter_statements(task.body):
            if stmt.kind != "assign" or not isinstance(stmt.expr, Call):
                continue
            if stmt.expr.name == "classifier":
                bound = _bind(stmt.expr, ["in", "classes", "hidden"])
                plan["in_dim"] = _int(bound.get("in"))
                plan["hidden"] = _int(bound.get("hidden"), 64)
                plan["out_dim"] = _int(bound.get("classes"))
                plan["classes"] = _int(bound.get("classes"))
            elif stmt.expr.name == "load_data":
                bound = _bind(stmt.expr, ["samples", "features", "classes"])
                plan["samples"] = _int(bound.get("samples"))
                plan["classes"] = _int(bound.get("classes"), plan["classes"])
        for stmt in _iter_statements(task.body):
            if stmt.kind == "train":
                plan["epochs"] = stmt.epochs
    return plan


def generate_cuda(program: Program, source_name: str) -> str:
    """Emit a CUDA execution plan: forward/backward SGD training kernels."""
    source = _json_string(source_name.replace("\\", "/"))
    plan = _cuda_plan(program)
    return (
        CUDA_TEMPLATE.replace("__SOURCE__", source)
        .replace("__TASK__", plan["task"])
        .replace("__IN__", str(plan["in_dim"]))
        .replace("__HIDDEN__", str(plan["hidden"]))
        .replace("__OUT_DIM__", str(plan["out_dim"]))
        .replace("__EPOCHS__", str(plan["epochs"]))
        .replace("__SAMPLES__", str(plan["samples"]))
        .replace("__CLASSES__", str(plan["classes"]))
    )


def generate_opencl(program: Program, source_name: str) -> tuple[str, str]:
    """Emit an OpenCL plan: host C (runtime-loaded ICD) + .cl kernel source."""
    source = _json_string(source_name.replace("\\", "/"))
    host = OPENCL_HOST_TEMPLATE.replace("__SOURCE__", source)
    return host, OPENCL_KERNEL_SOURCE


def compile_opencl(host_source: str, kernel_source: str, out_dir: Path, stem: str, toolchain: Toolchain) -> tuple[Path, float]:
    import hashlib

    out_dir.mkdir(parents=True, exist_ok=True)
    host_file = out_dir / f"{stem}_opencl.c"
    kernel_file = out_dir / f"{stem}_opencl.cl"
    host_file.write_text(host_source, encoding="utf-8", newline="\n")
    kernel_file.write_text(kernel_source, encoding="utf-8", newline="\n")
    binary = out_dir / (f"{stem}_opencl.exe" if os.name == "nt" else f"{stem}_opencl")
    stamp = out_dir / f"{stem}_opencl.aebuild"

    digest = hashlib.sha256((host_source + kernel_source).encode("utf-8")).hexdigest()[:24]
    if binary.is_file() and stamp.is_file() and stamp.read_text(encoding="utf-8").strip() == digest:
        return binary, 0.0

    # The host loads the ICD at runtime, so no OpenCL SDK or libs are needed.
    command = [toolchain.compiler, "-O2", "-o", str(binary), str(host_file)]
    if os.name == "nt":
        command += ["-lws2_32"]
    started = time.perf_counter()
    proc = subprocess.run(command, capture_output=True, text=True, errors="replace")
    compile_ms = (time.perf_counter() - started) * 1000.0
    if proc.returncode != 0 or not binary.is_file():
        raise CBackendError(
            "OpenCL host compilation failed\n"
            + " ".join(command)
            + "\n"
            + (proc.stdout or "")
            + (proc.stderr or "")
        )
    stamp.write_text(digest, encoding="utf-8", newline="\n")
    return binary, compile_ms


def compile_cuda(source: str, out_dir: Path, stem: str) -> tuple[Path, float, str]:
    nvcc = shutil.which("nvcc")
    if not nvcc:
        raise CBackendError("nvcc not found on PATH")
    out_dir.mkdir(parents=True, exist_ok=True)
    source_file = out_dir / f"{stem}.cu"
    source_file.write_text(source, encoding="utf-8", newline="\n")
    binary = out_dir / (f"{stem}_cuda.exe" if os.name == "nt" else f"{stem}_cuda")
    command = [nvcc, "-O3", "-arch=native", "-o", str(binary), str(source_file)]
    started = time.perf_counter()
    proc = subprocess.run(command, capture_output=True, text=True, errors="replace")
    compile_ms = (time.perf_counter() - started) * 1000.0
    if proc.returncode != 0 or not binary.is_file():
        raise CBackendError(
            "nvcc compilation failed\n"
            + " ".join(command)
            + "\n"
            + (proc.stdout or "")
            + (proc.stderr or "")
        )
    return binary, compile_ms, " ".join(command)
