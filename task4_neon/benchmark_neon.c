
#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <arm_neon.h>

#define N 768
#define D 2048
#define REPEAT 300

static volatile float checksum = 0.0f;

double elapsed_ms(struct timespec start,
                  struct timespec end) {
    return (end.tv_sec - start.tv_sec) * 1000.0 +
           (end.tv_nsec - start.tv_nsec) / 1e6;
}

void matmul_scalar(float *out, const float *x,
                   const float *w, int n, int d) {
    for (int i = 0; i < d; i++) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++) {
            sum += w[i * n + j] * x[j];
        }
        out[i] = sum;
    }
}

void matmul_neon(float *out, const float *x,
                 const float *w, int n, int d) {
    for (int i = 0; i < d; i++) {
        float32x4_t sum_vec = vdupq_n_f32(0.0f);

        int j = 0;
        for (; j + 3 < n; j += 4) {
            float32x4_t weights =
                vld1q_f32(w + i * n + j);

            float32x4_t inputs =
                vld1q_f32(x + j);

            sum_vec = vfmaq_f32(
                sum_vec, weights, inputs);
        }

        float sum = vaddvq_f32(sum_vec);

        for (; j < n; j++) {
            sum += w[i * n + j] * x[j];
        }

        out[i] = sum;
    }
}

int main(void) {
    float *W = malloc((size_t)N * D * sizeof(float));
    float *x = malloc((size_t)N * sizeof(float));
    float *scalar_out = malloc((size_t)D * sizeof(float));
    float *neon_out = malloc((size_t)D * sizeof(float));

    if (!W || !x || !scalar_out || !neon_out) {
        fprintf(stderr, "Memory allocation failed\n");
        free(W); free(x); free(scalar_out); free(neon_out);
        return 1;
    }

    for (int i = 0; i < N; i++)
        x[i] = (i % 17 - 8) * 0.01f;

    for (int i = 0; i < N * D; i++)
        W[i] = (i % 23 - 11) * 0.01f;

    // Warm-up
    matmul_scalar(scalar_out, x, W, N, D);
    matmul_neon(neon_out, x, W, N, D);

    struct timespec start, end;

    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int r = 0; r < REPEAT; r++) {
        matmul_scalar(scalar_out, x, W, N, D);
        checksum += scalar_out[r % D];
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double scalar_ms = elapsed_ms(start, end);

    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int r = 0; r < REPEAT; r++) {
        matmul_neon(neon_out, x, W, N, D);
        checksum += neon_out[r % D];
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double neon_ms = elapsed_ms(start, end);

    float max_error = 0.0f;
    for (int i = 0; i < D; i++) {
        float error = fabsf(scalar_out[i] - neon_out[i]);
        if (error > max_error)
            max_error = error;
    }

    printf("Matrix dimensions: %d x %d\n", D, N);
    printf("Iterations: %d\n", REPEAT);
    printf("Scalar total: %.3f ms\n", scalar_ms);
    printf("NEON total: %.3f ms\n", neon_ms);
    printf("Scalar average: %.6f ms\n", scalar_ms / REPEAT);
    printf("NEON average: %.6f ms\n", neon_ms / REPEAT);
    printf("Speedup: %.3fx\n", scalar_ms / neon_ms);
    printf("Maximum absolute error: %.9f\n", max_error);
    printf("Checksum: %.6f\n", checksum);

    free(W);
    free(x);
    free(scalar_out);
    free(neon_out);

    return 0;
}
