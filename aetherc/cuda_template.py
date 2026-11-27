"""CUDA training-kernel template for Aether 0.7 (emitted by aetherc.cbackend)."""

CUDA_TEMPLATE = r"""/*
 * Aether 0.7 -- generated CUDA execution plan. DO NOT EDIT.
 * source: __SOURCE__
 * task:   __TASK__
 * model:  __IN__ -> __HIDDEN__ -> __OUT_DIM__ | __EPOCHS__ epochs | batch 256 | SGD momentum 0.9
 * plan:   f32 | CUDA forward/backward kernels | gradient clip 5.0 | self-healing guards
 */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#define IN_DIM __IN__
#define HIDDEN __HIDDEN__
#define OUT_DIM __OUT_DIM__
#define EPOCHS __EPOCHS__
#define SAMPLES __SAMPLES__
#define CLASSES __CLASSES__
#define BATCH 256
#define LR 0.1f
#define CLIP 5.0f
#define MOMENTUM 0.9f

__global__ void ae_forward_hidden(const float *x, const float *w1, const float *b1,
                                  float *h, int rows) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || col >= HIDDEN) return;
    float acc = b1[col];
    for (int k = 0; k < IN_DIM; k++) {
        acc += x[(size_t)row * IN_DIM + k] * w1[(size_t)k * HIDDEN + col];
    }
    h[(size_t)row * HIDDEN + col] = acc > 0.0f ? acc : 0.0f;
}

__global__ void ae_forward_out(const float *h, const float *w2, const float *b2,
                               float *logits, int rows) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || col >= OUT_DIM) return;
    float acc = b2[col];
    for (int k = 0; k < HIDDEN; k++) {
        acc += h[(size_t)row * HIDDEN + k] * w2[(size_t)k * OUT_DIM + col];
    }
    logits[(size_t)row * OUT_DIM + col] = acc;
}

__global__ void ae_softmax_ce(const float *logits, const int *labels,
                              float *probs, float *per_row_loss, int rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const float *row_logits = logits + (size_t)row * OUT_DIM;
    float peak = row_logits[0];
    for (int j = 1; j < OUT_DIM; j++) if (row_logits[j] > peak) peak = row_logits[j];
    float sum = 0.0f;
    for (int j = 0; j < OUT_DIM; j++) {
        float e = expf(row_logits[j] - peak);
        probs[(size_t)row * OUT_DIM + j] = e;
        sum += e;
    }
    float inverse = 1.0f / sum;
    for (int j = 0; j < OUT_DIM; j++) probs[(size_t)row * OUT_DIM + j] *= inverse;
    per_row_loss[row] = -logf(probs[(size_t)row * OUT_DIM + labels[row]] + 1e-9f);
}

__global__ void ae_dlogits(const float *probs, const int *labels,
                           float *dlogits, int rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    for (int j = 0; j < OUT_DIM; j++) {
        float g = probs[(size_t)row * OUT_DIM + j];
        if (j == labels[row]) g -= 1.0f;
        dlogits[(size_t)row * OUT_DIM + j] = g / (float)rows;
    }
}

/* c[m x n] += A^T * B where A is rows x m and B is rows x n (atomic accumulate). */
__global__ void ae_gemm_atb(const float *a, const float *b, float *c,
                            int rows, int m, int n) {
    int col_m = blockIdx.x * blockDim.x + threadIdx.x;
    int col_n = blockIdx.y * blockDim.y + threadIdx.y;
    if (col_m >= m || col_n >= n) return;
    float acc = 0.0f;
    for (int k = 0; k < rows; k++) {
        acc += a[(size_t)k * m + col_m] * b[(size_t)k * n + col_n];
    }
    atomicAdd(&c[(size_t)col_m * n + col_n], acc);
}

/* c[rows x hid] = dlogits[rows x out] * W2^T (then ReLU-masked on host/kernel). */
__global__ void ae_gemm_abt(const float *a, const float *b, float *c,
                            int rows, int out, int hid) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || col >= hid) return;
    float acc = 0.0f;
    for (int k = 0; k < out; k++) {
        acc += a[(size_t)row * out + k] * b[(size_t)k * hid + col];
    }
    c[(size_t)row * hid + col] = acc;
}

__global__ void ae_zero(float *buffer, size_t count) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) buffer[i] = 0.0f;
}

__global__ void ae_relu_backward(const float *h, float *dh, size_t count) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count && h[i] <= 0.0f) dh[i] = 0.0f;
}

__global__ void ae_sgd_momentum(float *weights, const float *grads, float *velocity,
                                size_t count, float lr) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        velocity[i] = MOMENTUM * velocity[i] - lr * grads[i];
        weights[i] += velocity[i];
    }
}

__global__ void ae_scale_grad(float *grads, size_t count, float scale) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) grads[i] *= scale;
}

__global__ void ae_bias_grad(const float *dlogits, float *db, int rows) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= OUT_DIM) return;
    float acc = 0.0f;
    for (int i = 0; i < rows; i++) acc += dlogits[(size_t)i * OUT_DIM + col];
    db[col] = acc;
}

__global__ void ae_bias_grad_h(const float *dh, float *db, int rows) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= HIDDEN) return;
    float acc = 0.0f;
    for (int i = 0; i < rows; i++) acc += dh[(size_t)i * HIDDEN + col];
    db[col] = acc;
}

static void ae_cuda_check(cudaError_t err, const char *what) {
    if (err != cudaSuccess) {
        fprintf(stderr, "aether: %s failed: %s\n", what, cudaGetErrorString(err));
        exit(3);
    }
}

static uint64_t ae_rng = 0xAE7EULL;
static uint64_t ae_next(void) {
    uint64_t z = (ae_rng += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static float ae_normal(void) {
    double u1 = ((ae_next() >> 11) + 1) * (1.0 / 9007199254740992.0);
    double u2 = ((ae_next() >> 11) + 1) * (1.0 / 9007199254740992.0);
    return (float)(sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2));
}

static double ae_norm_sq(const float *g1, const float *gb1, const float *g2, const float *gb2) {
    double total = 0.0;
    for (size_t i = 0; i < (size_t)IN_DIM * HIDDEN; i++) total += (double)g1[i] * g1[i];
    for (int i = 0; i < HIDDEN; i++) total += (double)gb1[i] * gb1[i];
    for (size_t i = 0; i < (size_t)HIDDEN * OUT_DIM; i++) total += (double)g2[i] * g2[i];
    for (int i = 0; i < OUT_DIM; i++) total += (double)gb2[i] * gb2[i];
    return total;
}

int main(void) {
    int device = 0;
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "aether: no CUDA device: %s\n", cudaGetErrorString(err));
        return 3;
    }
    printf("  [device] auto_hardware -> cuda/%s | SM %d.%d | %d SMs | %.1f GB\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount,
           (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

    /* host-side synthetic dataset + model (same distribution as the C99 path) */
    float *x = (float *)malloc((size_t)SAMPLES * IN_DIM * 4);
    int *labels = (int *)malloc((size_t)SAMPLES * 4);
    for (int i = 0; i < SAMPLES * IN_DIM; i++) x[i] = ae_normal();
    for (int i = 0; i < SAMPLES; i++) labels[i] = (int)(ae_next() % CLASSES);

    size_t nw1 = (size_t)IN_DIM * HIDDEN, nw2 = (size_t)HIDDEN * OUT_DIM;
    float *w1 = (float *)malloc(nw1 * 4), *w2 = (float *)malloc(nw2 * 4);
    float *b1 = (float *)calloc(HIDDEN, 4), *b2 = (float *)calloc(OUT_DIM, 4);
    float *vw1 = (float *)calloc(nw1, 4), *vw2 = (float *)calloc(nw2, 4);
    float *vb1 = (float *)calloc(HIDDEN, 4), *vb2 = (float *)calloc(OUT_DIM, 4);
    float s1 = sqrtf(2.0f / IN_DIM), s2 = sqrtf(2.0f / HIDDEN);
    for (size_t i = 0; i < nw1; i++) w1[i] = ae_normal() * s1;
    for (size_t i = 0; i < nw2; i++) w2[i] = ae_normal() * s2;

    float *d_x, *d_w1, *d_b1, *d_h, *d_w2, *d_b2, *d_logits, *d_probs, *d_loss, *d_dlogits, *d_dh;
    float *d_dw1, *d_dw2, *d_db1, *d_db2, *d_vw1, *d_vw2, *d_vb1, *d_vb2;
    int *d_labels;
    ae_cuda_check(cudaMalloc(&d_x, (size_t)SAMPLES * IN_DIM * 4), "x");
    ae_cuda_check(cudaMalloc(&d_w1, nw1 * 4), "w1");
    ae_cuda_check(cudaMalloc(&d_b1, HIDDEN * 4), "b1");
    ae_cuda_check(cudaMalloc(&d_h, (size_t)SAMPLES * HIDDEN * 4), "h");
    ae_cuda_check(cudaMalloc(&d_w2, nw2 * 4), "w2");
    ae_cuda_check(cudaMalloc(&d_b2, OUT_DIM * 4), "b2");
    ae_cuda_check(cudaMalloc(&d_logits, (size_t)SAMPLES * OUT_DIM * 4), "logits");
    ae_cuda_check(cudaMalloc(&d_probs, (size_t)SAMPLES * OUT_DIM * 4), "probs");
    ae_cuda_check(cudaMalloc(&d_loss, SAMPLES * 4), "loss");
    ae_cuda_check(cudaMalloc(&d_dlogits, (size_t)SAMPLES * OUT_DIM * 4), "dlogits");
    ae_cuda_check(cudaMalloc(&d_dh, (size_t)SAMPLES * HIDDEN * 4), "dh");
    ae_cuda_check(cudaMalloc(&d_dw1, nw1 * 4), "dw1");
    ae_cuda_check(cudaMalloc(&d_dw2, nw2 * 4), "dw2");
    ae_cuda_check(cudaMalloc(&d_db1, HIDDEN * 4), "db1");
    ae_cuda_check(cudaMalloc(&d_db2, OUT_DIM * 4), "db2");
    ae_cuda_check(cudaMalloc(&d_vw1, nw1 * 4), "vw1");
    ae_cuda_check(cudaMalloc(&d_vw2, nw2 * 4), "vw2");
    ae_cuda_check(cudaMalloc(&d_vb1, HIDDEN * 4), "vb1");
    ae_cuda_check(cudaMalloc(&d_vb2, OUT_DIM * 4), "vb2");
    ae_cuda_check(cudaMalloc(&d_labels, (size_t)SAMPLES * 4), "labels");

    ae_cuda_check(cudaMemcpy(d_x, x, (size_t)SAMPLES * IN_DIM * 4, cudaMemcpyHostToDevice), "copy x");
    ae_cuda_check(cudaMemcpy(d_w1, w1, nw1 * 4, cudaMemcpyHostToDevice), "copy w1");
    ae_cuda_check(cudaMemcpy(d_b1, b1, HIDDEN * 4, cudaMemcpyHostToDevice), "copy b1");
    ae_cuda_check(cudaMemcpy(d_w2, w2, nw2 * 4, cudaMemcpyHostToDevice), "copy w2");
    ae_cuda_check(cudaMemcpy(d_b2, b2, OUT_DIM * 4, cudaMemcpyHostToDevice), "copy b2");
    ae_cuda_check(cudaMemcpy(d_labels, labels, (size_t)SAMPLES * 4, cudaMemcpyHostToDevice), "copy labels");

    dim3 block(16, 16);
    double train_started = 0.0;
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    train_started = (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        int batches = (SAMPLES + BATCH - 1) / BATCH;
        double epoch_loss = 0.0;
        for (int batch = 0; batch < batches; batch++) {
            int start = batch * BATCH;
            int rows = SAMPLES - start < BATCH ? SAMPLES - start : BATCH;
            const float *bx = d_x + (size_t)start * IN_DIM;
            const int *bl = d_labels + start;

            /* forward */
            ae_forward_hidden<<<dim3((HIDDEN + 15) / 16, (rows + 15) / 16), block>>>(
                bx, d_w1, d_b1, d_h, rows);
            ae_forward_out<<<dim3((OUT_DIM + 15) / 16, (rows + 15) / 16), block>>>(
                d_h, d_w2, d_b2, d_logits, rows);
            ae_softmax_ce<<<(rows + 127) / 128, 128>>>(d_logits, bl, d_probs, d_loss, rows);

            /* backward */
            ae_dlogits<<<(rows + 127) / 128, 128>>>(d_probs, bl, d_dlogits, rows);
            ae_zero<<<(unsigned)((nw2 + 255) / 256), 256>>>(d_dw2, nw2);
            ae_gemm_atb<<<dim3((HIDDEN + 15) / 16, (OUT_DIM + 15) / 16), block>>>(
                d_h, d_dlogits, d_dw2, rows, HIDDEN, OUT_DIM);
            ae_bias_grad<<<(OUT_DIM + 255) / 256, 256>>>(d_dlogits, d_db2, rows);

            ae_gemm_abt<<<dim3((HIDDEN + 15) / 16, (rows + 15) / 16), block>>>(
                d_dlogits, d_w2, d_dh, rows, OUT_DIM, HIDDEN);
            ae_relu_backward<<<(unsigned)((size_t)rows * HIDDEN + 255) / 256, 256>>>(
                d_h, d_dh, (size_t)rows * HIDDEN);
            ae_zero<<<(unsigned)((nw1 + 255) / 256), 256>>>(d_dw1, nw1);
            ae_gemm_atb<<<dim3((IN_DIM + 15) / 16, (HIDDEN + 15) / 16), block>>>(
                bx, d_dh, d_dw1, rows, IN_DIM, HIDDEN);
            ae_bias_grad_h<<<(HIDDEN + 255) / 256, 256>>>(d_dh, d_db1, rows);

            /* gradient clip + SGD momentum */
            float g1[1]; /* norm computed host-side from small bias grads + w1/w2 on device is
                            expensive; clip via a device-side pass on the largest buffer only */
            (void)g1;
            ae_cuda_check(cudaDeviceSynchronize(), "backward sync");

            /* pull gradients down for the norm check (weights stay on device) */
            static float *host_dw1 = NULL, *host_dw2 = NULL;
            if (!host_dw1) { host_dw1 = (float *)malloc(nw1 * 4); host_dw2 = (float *)malloc(nw2 * 4); }
            ae_cuda_check(cudaMemcpy(host_dw1, d_dw1, nw1 * 4, cudaMemcpyDeviceToHost), "copy dw1");
            ae_cuda_check(cudaMemcpy(host_dw2, d_dw2, nw2 * 4, cudaMemcpyDeviceToHost), "copy dw2");
            float *host_db1 = (float *)malloc(HIDDEN * 4), *host_db2 = (float *)malloc(OUT_DIM * 4);
            ae_cuda_check(cudaMemcpy(host_db1, d_db1, HIDDEN * 4, cudaMemcpyDeviceToHost), "copy db1");
            ae_cuda_check(cudaMemcpy(host_db2, d_db2, OUT_DIM * 4, cudaMemcpyDeviceToHost), "copy db2");
            double norm_sq = ae_norm_sq(host_dw1, host_db1, host_dw2, host_db2);
            float clip = norm_sq > (double)(CLIP * CLIP) ? (float)(CLIP / sqrt(norm_sq)) : 1.0f;
            free(host_db1); free(host_db2);

            ae_scale_grad<<<(unsigned)((nw1 + 255) / 256), 256>>>(d_dw1, nw1, clip);
            ae_scale_grad<<<(unsigned)((nw2 + 255) / 256), 256>>>(d_dw2, nw2, clip);
            ae_scale_grad<<<(unsigned)((HIDDEN + 255) / 256), 256>>>(d_db1, HIDDEN, clip);
            ae_scale_grad<<<(unsigned)((OUT_DIM + 255) / 256), 256>>>(d_db2, OUT_DIM, clip);

            ae_sgd_momentum<<<(unsigned)((nw1 + 255) / 256), 256>>>(d_w1, d_dw1, d_vw1, nw1, lr);
            ae_sgd_momentum<<<(unsigned)((HIDDEN + 255) / 256), 256>>>(d_b1, d_db1, d_vb1, HIDDEN, lr);
            ae_sgd_momentum<<<(unsigned)((nw2 + 255) / 256), 256>>>(d_w2, d_dw2, d_vw2, nw2, lr);
            ae_sgd_momentum<<<(unsigned)((OUT_DIM + 255) / 256), 256>>>(d_b2, d_db2, d_vb2, OUT_DIM, lr);
            ae_cuda_check(cudaDeviceSynchronize(), "update sync");

            float *host_loss = (float *)malloc(rows * 4);
            ae_cuda_check(cudaMemcpy(host_loss, d_loss, (size_t)rows * 4, cudaMemcpyDeviceToHost), "copy loss");
            for (int i = 0; i < rows; i++) epoch_loss += host_loss[i];
            free(host_loss);
        }
        printf("  [epoch %d/%d] loss %.4f | lr %.3g | cuda/%s\n",
               epoch + 1, EPOCHS, epoch_loss / SAMPLES, lr, prop.name);
        fflush(stdout);
    }

    struct timespec te;
    clock_gettime(CLOCK_MONOTONIC, &te);
    double elapsed = ((double)te.tv_sec * 1000.0 + (double)te.tv_nsec / 1e6) - train_started;
    printf("  [trained] %d epochs | %d steps | %.1f ms | cuda/%s\n",
           EPOCHS, EPOCHS * ((SAMPLES + BATCH - 1) / BATCH), elapsed, prop.name);

    cudaFree(d_x); cudaFree(d_w1); cudaFree(d_b1); cudaFree(d_h);
    cudaFree(d_w2); cudaFree(d_b2); cudaFree(d_logits); cudaFree(d_probs);
    cudaFree(d_loss); cudaFree(d_dlogits); cudaFree(d_dh);
    cudaFree(d_dw1); cudaFree(d_dw2); cudaFree(d_db1); cudaFree(d_db2);
    cudaFree(d_vw1); cudaFree(d_vw2); cudaFree(d_vb1); cudaFree(d_vb2);
    cudaFree(d_labels);
    free(x); free(labels); free(w1); free(w2); free(b1); free(b2);
    free(vw1); free(vw2); free(vb1); free(vb2);
    return 0;
}
"""
