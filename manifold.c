/**
 * manifold.c — Quarter-Square Matmul kernel + manifold math primitives.
 *
 * The fake 32-word vocabulary has been removed.
 * All text generation is now in gemma_infer.c using real Gemma 2 2B weights.
 */
#include <stdio.h>
#include <stdlib.h>
#include <dispatch/dispatch.h>
#include "manifold.h"

#ifdef MANIFOLD_SIMD
#include <arm_neon.h>
#endif

/* Memory-Mapped Quarter-Square Lookup Table (L1 Cache friendly) */
static uint32_t SQ_TABLE[SQ_TABLE_SIZE];

void manifold_init(void) {
    /* Initialize SQ_TABLE[n] = n^2 without hardware multipliers */
    SQ_TABLE[0] = 0;
    for (uint32_t n = 1; n < SQ_TABLE_SIZE; n++) {
        /* n^2 = (n-1)^2 + 2(n-1) + 1 */
        SQ_TABLE[n] = SQ_TABLE[n - 1] + ((n - 1) << 1) + 1;
    }
}

uint16_t manifold_phase_rotate(uint16_t current_phase, uint16_t steps) {
    return (current_phase + steps) & ARC_MASK;
}

uint16_t manifold_antipodal_invert(uint16_t current_phase) {
    return (current_phase + ANTIPODAL_INVERSION_NODE) & ARC_MASK;
}

uint16_t manifold_spatial_translate(uint16_t current_dist, uint16_t shift) {
    return (current_dist + shift) & DIST_MASK;
}

uint16_t manifold_qsm_multiply(uint16_t vec_A, uint16_t vec_B) {
    uint16_t sum  = vec_A + vec_B;
    uint16_t diff = (vec_A >= vec_B) ? (vec_A - vec_B) : (vec_B - vec_A);
    uint32_t sq_sum  = SQ_TABLE[sum];
    uint32_t sq_diff = SQ_TABLE[diff];
    return (uint16_t)((sq_sum - sq_diff) >> 2);
}

uint16_t manifold_state_collapse(uint16_t x, uint16_t y, uint16_t z, uint16_t u) {
    return (x & DIST_MASK) +
           (y & DIST_MASK) +
           (z & DIST_MASK) +
           (u & OP_MASK);
}

int32_t manifold_qsm_row_dot(const uint8_t* Xbar_w, const uint8_t* x_bar_x,
                              uint16_t n_blocks, uint16_t block_size) {
    int32_t  total_sum   = 0;
    uint32_t offset_const = (uint32_t)(16384 * block_size);

    for (uint16_t b = 0; b < n_blocks; b++) {
        uint32_t A_b = 0, B_b = 0, C_b = 0;
        const uint8_t *w_block = Xbar_w + b * block_size;
        const uint8_t *x_block = x_bar_x + b * block_size;

        for (uint16_t i = 0; i < block_size; i++) {
            A_b += manifold_qsm_multiply(w_block[i], x_block[i]);
            B_b += w_block[i];
            C_b += x_block[i];
        }
        /* P_b = A_b + 16384*block_size - 128*(B_b+C_b) */
        int32_t P_b = (int32_t)A_b + offset_const - (int32_t)(128 * (B_b + C_b));
        total_sum  += P_b;
    }
    return total_sum;
}

void manifold_qsm_matmul(const uint8_t* Xbar, const uint8_t* x_bar,
                          int32_t* out, uint16_t out_feat,
                          uint16_t n_blocks, uint16_t block_size) {
#ifdef MANIFOLD_SIMD
    uint32_t in_feat = (uint32_t)n_blocks * block_size;

    // Pre-bias x_bar into a stack-allocated buffer to avoid malloc/free overhead.
    // Max in_feat for Gemma 2 2B is 9216. 16384 bytes is safe and fits on stack.
    int8_t x_s8_buf[16384];
    uint8x16_t bias = vdupq_n_u8(128);
    uint32_t i = 0;
    for (; i + 15 < in_feat; i += 16) {
        uint8x16_t x_u8 = vld1q_u8(x_bar + i);
        int8x16_t x_s8 = vreinterpretq_s8_u8(veorq_u8(x_u8, bias));
        vst1q_s8(x_s8_buf + i, x_s8);
    }
    for (; i < in_feat; i++) {
        x_s8_buf[i] = (int8_t)((int32_t)x_bar[i] - 128);
    }

    const int8_t *x_s8_ptr = x_s8_buf;

    dispatch_queue_t queue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0);

    // Chunk size: dynamically size to around 32 chunks to minimize scheduling overhead
    uint32_t chunk_size = out_feat / 32;
    if (chunk_size < 32) chunk_size = 32;
    uint32_t num_chunks = (out_feat + chunk_size - 1) / chunk_size;

    dispatch_apply(num_chunks, queue, ^(size_t chunk_idx) {
        uint32_t r_start = chunk_idx * chunk_size;
        uint32_t r_end = r_start + chunk_size;
        if (r_end > out_feat) r_end = out_feat;

        uint32_t r = r_start;
        // Core loop: 8-row unrolling
        for (; r + 7 < r_end; r += 8) {
            const uint8_t *w_row0 = Xbar + (r + 0) * in_feat;
            const uint8_t *w_row1 = Xbar + (r + 1) * in_feat;
            const uint8_t *w_row2 = Xbar + (r + 2) * in_feat;
            const uint8_t *w_row3 = Xbar + (r + 3) * in_feat;
            const uint8_t *w_row4 = Xbar + (r + 4) * in_feat;
            const uint8_t *w_row5 = Xbar + (r + 5) * in_feat;
            const uint8_t *w_row6 = Xbar + (r + 6) * in_feat;
            const uint8_t *w_row7 = Xbar + (r + 7) * in_feat;

            int32x4_t acc0 = vdupq_n_s32(0);
            int32x4_t acc1 = vdupq_n_s32(0);
            int32x4_t acc2 = vdupq_n_s32(0);
            int32x4_t acc3 = vdupq_n_s32(0);
            int32x4_t acc4 = vdupq_n_s32(0);
            int32x4_t acc5 = vdupq_n_s32(0);
            int32x4_t acc6 = vdupq_n_s32(0);
            int32x4_t acc7 = vdupq_n_s32(0);

            uint32_t i = 0;
            for (; i + 15 < in_feat; i += 16) {
                int8x16_t x_s8 = vld1q_s8(x_s8_ptr + i);

                // Load raw weights
                uint8x16_t w_u8_0 = vld1q_u8(w_row0 + i);
                uint8x16_t w_u8_1 = vld1q_u8(w_row1 + i);
                uint8x16_t w_u8_2 = vld1q_u8(w_row2 + i);
                uint8x16_t w_u8_3 = vld1q_u8(w_row3 + i);
                uint8x16_t w_u8_4 = vld1q_u8(w_row4 + i);
                uint8x16_t w_u8_5 = vld1q_u8(w_row5 + i);
                uint8x16_t w_u8_6 = vld1q_u8(w_row6 + i);
                uint8x16_t w_u8_7 = vld1q_u8(w_row7 + i);

                // Bias weights on the fly (XOR by 128)
                int8x16_t w_s8_0 = vreinterpretq_s8_u8(veorq_u8(w_u8_0, bias));
                int8x16_t w_s8_1 = vreinterpretq_s8_u8(veorq_u8(w_u8_1, bias));
                int8x16_t w_s8_2 = vreinterpretq_s8_u8(veorq_u8(w_u8_2, bias));
                int8x16_t w_s8_3 = vreinterpretq_s8_u8(veorq_u8(w_u8_3, bias));
                int8x16_t w_s8_4 = vreinterpretq_s8_u8(veorq_u8(w_u8_4, bias));
                int8x16_t w_s8_5 = vreinterpretq_s8_u8(veorq_u8(w_u8_5, bias));
                int8x16_t w_s8_6 = vreinterpretq_s8_u8(veorq_u8(w_u8_6, bias));
                int8x16_t w_s8_7 = vreinterpretq_s8_u8(veorq_u8(w_u8_7, bias));

                acc0 = vdotq_s32(acc0, w_s8_0, x_s8);
                acc1 = vdotq_s32(acc1, w_s8_1, x_s8);
                acc2 = vdotq_s32(acc2, w_s8_2, x_s8);
                acc3 = vdotq_s32(acc3, w_s8_3, x_s8);
                acc4 = vdotq_s32(acc4, w_s8_4, x_s8);
                acc5 = vdotq_s32(acc5, w_s8_5, x_s8);
                acc6 = vdotq_s32(acc6, w_s8_6, x_s8);
                acc7 = vdotq_s32(acc7, w_s8_7, x_s8);
            }

            int32_t sum0 = vaddvq_s32(acc0);
            int32_t sum1 = vaddvq_s32(acc1);
            int32_t sum2 = vaddvq_s32(acc2);
            int32_t sum3 = vaddvq_s32(acc3);
            int32_t sum4 = vaddvq_s32(acc4);
            int32_t sum5 = vaddvq_s32(acc5);
            int32_t sum6 = vaddvq_s32(acc6);
            int32_t sum7 = vaddvq_s32(acc7);

            for (; i < in_feat; i++) {
                int32_t xi = (int32_t)x_s8_ptr[i];
                sum0 += (int32_t)(int8_t)(w_row0[i] ^ 128) * xi;
                sum1 += (int32_t)(int8_t)(w_row1[i] ^ 128) * xi;
                sum2 += (int32_t)(int8_t)(w_row2[i] ^ 128) * xi;
                sum3 += (int32_t)(int8_t)(w_row3[i] ^ 128) * xi;
                sum4 += (int32_t)(int8_t)(w_row4[i] ^ 128) * xi;
                sum5 += (int32_t)(int8_t)(w_row5[i] ^ 128) * xi;
                sum6 += (int32_t)(int8_t)(w_row6[i] ^ 128) * xi;
                sum7 += (int32_t)(int8_t)(w_row7[i] ^ 128) * xi;
            }

            out[r + 0] = sum0;
            out[r + 1] = sum1;
            out[r + 2] = sum2;
            out[r + 3] = sum3;
            out[r + 4] = sum4;
            out[r + 5] = sum5;
            out[r + 6] = sum6;
            out[r + 7] = sum7;
        }

        // 4-row fallback
        for (; r + 3 < r_end; r += 4) {
            const uint8_t *w_row0 = Xbar + (r + 0) * in_feat;
            const uint8_t *w_row1 = Xbar + (r + 1) * in_feat;
            const uint8_t *w_row2 = Xbar + (r + 2) * in_feat;
            const uint8_t *w_row3 = Xbar + (r + 3) * in_feat;

            int32x4_t acc0 = vdupq_n_s32(0);
            int32x4_t acc1 = vdupq_n_s32(0);
            int32x4_t acc2 = vdupq_n_s32(0);
            int32x4_t acc3 = vdupq_n_s32(0);

            uint32_t i = 0;
            for (; i + 15 < in_feat; i += 16) {
                int8x16_t x_s8 = vld1q_s8(x_s8_ptr + i);

                uint8x16_t w_u8_0 = vld1q_u8(w_row0 + i);
                uint8x16_t w_u8_1 = vld1q_u8(w_row1 + i);
                uint8x16_t w_u8_2 = vld1q_u8(w_row2 + i);
                uint8x16_t w_u8_3 = vld1q_u8(w_row3 + i);

                int8x16_t w_s8_0 = vreinterpretq_s8_u8(veorq_u8(w_u8_0, bias));
                int8x16_t w_s8_1 = vreinterpretq_s8_u8(veorq_u8(w_u8_1, bias));
                int8x16_t w_s8_2 = vreinterpretq_s8_u8(veorq_u8(w_u8_2, bias));
                int8x16_t w_s8_3 = vreinterpretq_s8_u8(veorq_u8(w_u8_3, bias));

                acc0 = vdotq_s32(acc0, w_s8_0, x_s8);
                acc1 = vdotq_s32(acc1, w_s8_1, x_s8);
                acc2 = vdotq_s32(acc2, w_s8_2, x_s8);
                acc3 = vdotq_s32(acc3, w_s8_3, x_s8);
            }

            int32_t sum0 = vaddvq_s32(acc0);
            int32_t sum1 = vaddvq_s32(acc1);
            int32_t sum2 = vaddvq_s32(acc2);
            int32_t sum3 = vaddvq_s32(acc3);

            for (; i < in_feat; i++) {
                int32_t xi = (int32_t)x_s8_ptr[i];
                sum0 += (int32_t)(int8_t)(w_row0[i] ^ 128) * xi;
                sum1 += (int32_t)(int8_t)(w_row1[i] ^ 128) * xi;
                sum2 += (int32_t)(int8_t)(w_row2[i] ^ 128) * xi;
                sum3 += (int32_t)(int8_t)(w_row3[i] ^ 128) * xi;
            }

            out[r + 0] = sum0;
            out[r + 1] = sum1;
            out[r + 2] = sum2;
            out[r + 3] = sum3;
        }

        // Remainder rows
        for (; r < r_end; r++) {
            const uint8_t *w_row = Xbar + r * in_feat;
            int32x4_t acc = vdupq_n_s32(0);
            uint32_t i = 0;

            for (; i + 15 < in_feat; i += 16) {
                int8x16_t x_s8 = vld1q_s8(x_s8_ptr + i);
                uint8x16_t w_u8 = vld1q_u8(w_row + i);
                int8x16_t w_s8 = vreinterpretq_s8_u8(veorq_u8(w_u8, bias));
                acc = vdotq_s32(acc, w_s8, x_s8);
            }

            int32_t row_sum = vaddvq_s32(acc);
            for (; i < in_feat; i++) {
                row_sum += (int32_t)(int8_t)(w_row[i] ^ 128) * (int32_t)x_s8_ptr[i];
            }
            out[r] = row_sum;
        }
    });
#else
    uint32_t offset_const = (uint32_t)(16384 * block_size);
    uint32_t stride_bytes = (uint32_t)n_blocks * block_size;
    uint16_t block_mask   = block_size - 1;

    dispatch_queue_t queue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0);

    dispatch_apply(out_feat, queue, ^(size_t r) {
        const uint8_t *w_row = Xbar + r * stride_bytes;
        int32_t row_sum = 0;

        for (uint16_t b = 0; b < n_blocks; b++) {
            const uint8_t *w_block = w_row   + b * block_size;
            const uint8_t *x_block = x_bar   + b * block_size;
            uint32_t A_b = 0, B_b = 0, C_b = 0;

            for (uint16_t j = 0; j < block_size; j++) {
                /* stride-7 coprime walk for cache-friendly access pattern */
                uint16_t i    = (j * 7) & block_mask;
                uint16_t w_val = w_block[i];
                uint16_t x_val = x_block[i];
                uint16_t sum  = w_val + x_val;
                uint16_t diff = (w_val >= x_val) ? (w_val - x_val) : (x_val - w_val);
                A_b += (SQ_TABLE[sum] - SQ_TABLE[diff]) >> 2;
                B_b += w_val;
                C_b += x_val;
            }
            int32_t P_b = (int32_t)A_b + offset_const - (int32_t)(128 * (B_b + C_b));
            row_sum += P_b;
        }
        out[r] = row_sum;
    });
#endif
}

void manifold_stochastic_diffusion(manifold_state_t* state, uint8_t dimension, int16_t step) {
    if      (dimension == 0) state->theta = manifold_phase_rotate(state->theta, (uint16_t)step);
    else if (dimension == 1) state->x     = manifold_spatial_translate(state->x, (uint16_t)step);
    else if (dimension == 2) state->y     = manifold_spatial_translate(state->y, (uint16_t)step);
    else if (dimension == 3) state->z     = manifold_spatial_translate(state->z, (uint16_t)step);
    else if (dimension == 4) state->u     = (uint16_t)((state->u + step) & OP_MASK);
}

/** Returns -1, 0, or +1 for the shortest direction on a masked ring */
int stepDir1D(int target, int current, int mask) {
    if (target == current) return 0;
    int range = mask + 1;
    int diff  = (target - current + range) & mask;
    return diff < (range >> 1) ? 1 : -1;
}

int manifold_simd_enabled(void) {
#ifdef MANIFOLD_SIMD
    return 1;
#else
    return 0;
#endif
}
