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

from .ast_nodes import Call, Literal, Name, Program, Stmt, Task

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

#define AE_VERSION "0.2"
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
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1000.0 + (double)t.tv_nsec / 1e6;
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
typedef struct { int rows, out_dim, chunks; double checksum; unsigned digest; double accuracy; } StreamResult;

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

static char ae_base_dir[1024] = "";

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

    ae_event("{\"event\":\"summary\",\"chunks\":%d,\"rows\":%d,\"out_dim\":%d,\"gflops\":%.6f,\"ms\":%.6f,\"gflops_per_s\":%.6f,\"checksum\":%.6g,\"digest\":\"0x%08x\",\"accuracy\":%.6g,\"loss\":%.6g,\"kind\":\"classifier\",\"backend\":\"c\"}",
             total, d->samples, m->out_dim, gflops, elapsed, elapsed > 0.0 ? gflops / (elapsed / 1000.0) : 0.0, confidence_sum, digest, accuracy, mean_loss);
    ae_human("  [done]   %.4f accuracy | loss %.4f | %.1f ms | %.1f GFLOP/s | out[%dx%d]",
             accuracy, mean_loss, elapsed, elapsed > 0.0 ? gflops / (elapsed / 1000.0) : 0.0, d->samples, m->out_dim);
    return result;
}

static void ae_emit_text(const char *text) {
    ae_event("{\"event\":\"print\",\"text\":\"%s\",\"backend\":\"c\"}", text);
    ae_human("  [emit]   %s", text);
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


def _types(task: Task) -> dict[str, tuple[str, str]]:
    """Map variable -> (kind, detail) where detail is 'model:classifier' etc."""
    known: dict[str, tuple[str, str]] = {}
    for stmt in task.body:
        if stmt.kind != "assign" or not isinstance(stmt.expr, Call):
            continue
        call = stmt.expr
        if call.name == "tensor":
            known[stmt.name or ""] = ("tensor", "")
        elif call.name == "load_data":
            known[stmt.name or ""] = ("dataset", "")
        elif call.name == "neural_net":
            known[stmt.name or ""] = ("model", "mlp")
        elif call.name == "classifier":
            known[stmt.name or ""] = ("model", "classifier")
        elif call.name == "run":
            known[stmt.name or ""] = ("stream", "")
    return known


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


def _expr_c(node: object, known: dict[str, tuple[str, str]], chunks: int) -> str:
    if isinstance(node, Literal):
        if isinstance(node.value, str):
            return f'"{_c_string(node.value)}"'
        if isinstance(node.value, float):
            return f"{node.value!r}f"
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
    if node.name == "run":
        return _run_expr(node, known, chunks)
    raise CBackendError(f"intent '{node.name}' is not supported by the C backend yet")


_WIDTHS: dict[str, int] = {}


def _known_width(node: object, known: dict[str, tuple[str, str]]) -> int:
    if isinstance(node, Name):
        return _WIDTHS.get(node.id, 0)
    return 0


def _task_c(task: Task, chunks: int) -> list[str]:
    _WIDTHS.clear()
    known = _types(task)
    for stmt in task.body:
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

    name = task.name
    lines = [
        f"static void task_{name}(void) {{",
        "    double task_started = ae_now_ms();",
    ]
    for stmt in task.body:
        lines.extend(_stmt_c(stmt, known, chunks))
    if len(lines) == 2:
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
    stmt: Stmt, known: dict[str, tuple[str, str]], chunks: int
) -> list[str]:
    if stmt.kind == "assign":
        value = _expr_c(stmt.expr, known, chunks)
        kind, detail = known.get(stmt.name or "", ("value", ""))
        c_type = {"tensor": "Tensor", "dataset": "Dataset", "model": "Model", "stream": "StreamResult"}.get(
            kind, "double"
        )
        return [f"    {c_type} {stmt.name} = {value};"]

    if stmt.kind == "emit":
        expr = stmt.expr
        if isinstance(expr, Name) and expr.id in known:
            kind, _ = known[expr.id]
            emitter = {
                "tensor": "ae_emit_tensor",
                "stream": "ae_emit_stream",
            }.get(kind)
            if emitter:
                return [f"    {emitter}(&{expr.id});" if kind == "tensor" else f"    {emitter}({expr.id});"]
            return [f"    ae_emit_text(\"{_c_string(expr.id)}\");"]
        if isinstance(expr, Literal):
            return [f'    ae_emit_text("{_c_string(str(expr.value))}");']
        if isinstance(expr, Call) and expr.name == "run":
            return [f"    {_run_expr(expr, known, chunks)};"]
        return ["    (void)0;"]

    if stmt.kind == "return":
        return [f"    return {_expr_c(stmt.expr, known, chunks)};"]

    if stmt.kind == "train":
        return [
            f"    {{ int heals = 0, steps = 0; double accuracy = 0.0;",
            f"      ae_train(&{stmt.target}, &{stmt.data}, {stmt.epochs}, 256, 0.1f, &heals, &steps, &accuracy); }}",
        ]

    if stmt.kind == "export":
        path = _c_string((stmt.path or "").replace("\\", "/"))
        return [f'    ae_export(&{stmt.target}, "{path}");']

    if stmt.kind == "expr" and isinstance(stmt.expr, Call) and stmt.expr.name == "run":
        return [f"    {_run_expr(stmt.expr, known, chunks)};"]

    return ["    (void)0;"]


def generate_c(
    program: Program,
    source_name: str,
    tasks: list[str] | None = None,
    chunks: int = 8,
) -> str:
    _WIDTHS.clear()
    source = _json_string(source_name.replace("\\", "/"))
    header = PREAMBLE.replace("__SOURCE__", source)
    out = [header]
    names = [task.name for task in program.tasks if not tasks or task.name in tasks]
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
    out.append('    ae_event("{\\"event\\":\\"device\\",\\"device\\":\\"cpu\\",\\"machine\\":\\"native\\",\\"threads\\":%d,\\"compiler\\":\\"__CC__\\",\\"blas\\":\\"__BLAS__\\",\\"dtype\\":\\"f32\\",\\"backend\\":\\"c\\"}", threads);')
    out.append('    ae_human("  [device] auto_hardware -> c/native | %d threads | __BLAS__ | f32", threads);')
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
    out_dir.mkdir(parents=True, exist_ok=True)
    source_file = out_dir / f"{stem}.c"
    source_file.write_text(source, encoding="utf-8", newline="\n")
    binary = out_dir / (f"{stem}.exe" if os.name == "nt" else stem)

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
