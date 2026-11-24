
#include <stdio.h>
#include <arm_neon.h>
#include <math.h>
#include <stdlib.h>

// Original scalar matrix-vector multiplication
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

// Arm NEON matrix-vector multiplication
void matmul_neon(float *out, const float *x,
                 const float *w, int n, int d) {
    for (int i = 0; i < d; i++) {
        float32x4_t sum_vec = vdupq_n_f32(0.0f);

        int j = 0;
        for (; j + 3 < n; j += 4) {
            float32x4_t weights = vld1q_f32(w + i * n + j);
            float32x4_t inputs = vld1q_f32(x + j);

            sum_vec = vfmaq_f32(sum_vec, weights, inputs);
        }

        float sum = vaddvq_f32(sum_vec);

        // Handle remaining elements if n is not divisible by 4
        for (; j < n; j++) {
            sum += w[i * n + j] * x[j];
        }

        out[i] = sum;
    }
}


int main(void) {
    const int n = 7;
    const int d = 3;

    float W[d * n];
    float x[n];
    float scalar_out[d];
    float neon_out[d];

    // Create test input with positive, negative,
    // and fractional floating-point values.
    for (int j = 0; j < n; j++) {
        x[j] = (j - 3) * 0.25f;
    }

    for (int i = 0; i < d * n; i++) {
        W[i] = ((i % 9) - 4) * 0.15f;
    }

    matmul_scalar(scalar_out, x, W, n, d);
    matmul_neon(neon_out, x, W, n, d);

    float max_error = 0.0f;

    printf("Row    Scalar        NEON          Abs Error\n");
    printf("------------------------------------------------\n");

    for (int i = 0; i < d; i++) {
        float error = fabsf(scalar_out[i] - neon_out[i]);

        if (error > max_error) {
            max_error = error;
        }

        printf("%-6d %-13.8f %-13.8f %.10f\n",
               i, scalar_out[i], neon_out[i], error);
    }

    printf("\nMaximum absolute error: %.10f\n", max_error);

    if (max_error < 1e-5f) {
        printf("Correctness Test: PASS\n");
    } else {
        printf("Correctness Test: FAIL\n");
        return 1;
    }

    return 0;
}

