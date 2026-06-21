#ifndef MANIFOLD_H
#define MANIFOLD_H

#include <stdint.h>

/* Real Gemma 2 2B inference engine (ONIX/HPQ4 + manifold_qsm_matmul) */
#include "gemma_infer.h"

/* Bitmasks for registers */
#define ARC_MASK  0x01FF
#define DIST_MASK 0x07FF
#define OP_MASK   0x01FF

/* Critical Nodes */
#define VACUUM_OFFSET 0
#define ANTIPODAL_INVERSION_NODE 256 /* 0x0100 */

/* Lookup table definitions */
#define SQ_TABLE_SIZE 4095

/**
 * @brief Representation of a point in the 4D discrete manifold.
 */
typedef struct {
    uint16_t theta; /* Arc (θ) - Angular Phase / Degrees (0-511) */
    uint16_t x;     /* Distance X - Spatial Translation (0-2047) */
    uint16_t y;     /* Distance Y - Spatial Translation (0-2047) */
    uint16_t z;     /* Distance Z - Spatial Translation (0-2047) */
    uint16_t u;     /* Operator U - Directional State History (0-511) */
} manifold_state_t;

/**
 * @brief Initializes the Memory-Mapped Quarter-Square LUT (SQ_TABLE).
 *        Computes squares using only bit-shifts and additions (no multiplication).
 */
void manifold_init(void);

/**
 * @brief Phase Rotation (Angular Shift).
 *        Replaces complex rotation matrices and trigonometry.
 * @param current_phase Current angular phase θ (0-511).
 * @param steps Phase steps to rotate.
 * @return Rotated phase within [0, 511].
 */
uint16_t manifold_phase_rotate(uint16_t current_phase, uint16_t steps);

/**
 * @brief Antipodal Inversion.
 *        Inverts the phase by 180 degrees (256 steps).
 * @param current_phase Current angular phase θ (0-511).
 * @return Inverted phase within [0, 511].
 */
uint16_t manifold_antipodal_invert(uint16_t current_phase);

/**
 * @brief Spatial Translation.
 *        Translates distance in a dimension.
 * @param current_dist Current spatial coordinate (0-2047).
 * @param shift Translation step.
 * @return Translated coordinate within [0, 2047].
 */
uint16_t manifold_spatial_translate(uint16_t current_dist, uint16_t shift);

/**
 * @brief Quarter-Square Vector Multiplication.
 *        Computes vec_A * vec_B using lookups.
 * @param vec_A First scale factor/vector dimension (0-2047).
 * @param vec_B Second scale factor/vector dimension (0-2047).
 * @return Product (uint16_t).
 */
uint16_t manifold_qsm_multiply(uint16_t vec_A, uint16_t vec_B);

/**
 * @brief 4D State Collapse.
 *        Reduces the superposition field into a single localized memory address.
 * @param x Spatial coordinate X.
 * @param y Spatial coordinate Y.
 * @param z Spatial coordinate Z.
 * @param u State history coordinate U.
 * @return Unified 16-bit address index (0-6652).
 */
uint16_t manifold_state_collapse(uint16_t x, uint16_t y, uint16_t z, uint16_t u);

/**
 * @brief Quarter-Square Dot Product for an entire row.
 *        Computes the integer dot product of a row composed of blocks of size block_size.
 *        Uses the offset-binary expansion formula:
 *        P_b = A_b + 16384 * block_size - 128 * (B_b + C_b)
 * @param Xbar_w Weight array of size n_blocks * block_size (unsigned offset-binary [0-255]).
 * @param x_bar_x Activation array of size n_blocks * block_size (unsigned offset-binary [0-255]).
 * @param n_blocks Number of blocks in the row.
 * @param block_size Number of elements per block.
 * @return Signed 32-bit integer dot product.
 */
int32_t manifold_qsm_row_dot(const uint8_t* Xbar_w, const uint8_t* x_bar_x, uint16_t n_blocks, uint16_t block_size);

/**
 * @brief Performs QSM Matrix-Vector Multiplication with QCM 128x113 Tiling.
 *        Computes out = Xbar @ x_bar using a coprime block walk and stride-7 elements.
 * @param Xbar Pointer to weight matrix [out_feat, n_blocks, block_size].
 * @param x_bar Pointer to activation vector [n_blocks * block_size].
 * @param out Pointer to preallocated output buffer [out_feat].
 * @param out_feat Number of rows.
 * @param n_blocks Number of blocks per row.
 * @param block_size Number of elements per block.
 */
void manifold_qsm_matmul(const uint8_t* Xbar, const uint8_t* x_bar, int32_t* out, uint16_t out_feat, uint16_t n_blocks, uint16_t block_size);

/**
 * @brief Performs a stochastic diffusion step on a state.
 *        Updates the state by randomly translating or rotating along
 *        one of the 5 coordinate axes using strict register masks.
 * @param state Pointer to the state to update.
 * @param dimension Dimension index (0 = theta, 1 = x, 2 = y, 3 = z, 4 = u).
 * @param step Stride step of the diffusion.
 */
void manifold_stochastic_diffusion(manifold_state_t* state, uint8_t dimension, int16_t step);

/**
 * @brief Returns 1 if SIMD is enabled, 0 otherwise.
 */
int manifold_simd_enabled(void);

#endif /* MANIFOLD_H */
