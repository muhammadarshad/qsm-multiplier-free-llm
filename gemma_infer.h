/**
 * gemma_infer.h — Gemma 2 2B autoregressive inference engine (C / Apple Silicon).
 *
 * Architecture constants (Gemma2-2B):
 *   hidden_size:       2304
 *   n_layers:          26
 *   n_q_heads:         8, n_kv_heads: 4, head_dim: 256, gqa_ratio: 2
 *   intermediate_size: 9216
 *   vocab_size:        256000
 *   rope_theta:        10000.0
 *   attn_softcap:      50.0
 *   final_softcap:     30.0
 *   rms_eps:           1e-6
 *
 * Weight sources (all mmap'd, never copied):
 *   ONIX file:   all 26-layer projection matrices (Xbar/S/Z in HPQ4)
 *   norms.bin:   26 × 4 layer norms + 1 final norm  (float16 → loaded as f32)
 *   embed.bin:   [256000, 2304] float16 embedding table (mmap'd)
 *
 * dequant formula per output row r:
 *   output[r] = int32_accumulator[r] × 2^S[r] / Z[r]  ×  act_scale
 *   where act_scale = 2^⌈log2(max|x| / 127)⌉
 */
#ifndef GEMMA_INFER_H
#define GEMMA_INFER_H

#include <stdint.h>
#include <stddef.h>
#include "gemma_onix.h"

/* ── Architecture constants ──────────────────────────────────────────────── */
#define G2_N_LAYERS     26
#define G2_HIDDEN       2304
#define G2_N_Q_HEADS    8
#define G2_N_KV_HEADS   4
#define G2_HEAD_DIM     256
#define G2_INTER        9216
#define G2_VOCAB        256000
#define G2_GQA_RATIO    (G2_N_Q_HEADS / G2_N_KV_HEADS)   /* 2 */
#define G2_ROPE_THETA   10000.0f
#define G2_ATTN_CAP     50.0f
#define G2_FINAL_CAP    30.0f
#define G2_RMS_EPS      1e-6f
#define G2_EMBED_SCALE  48.0f   /* sqrt(2304) */

/* KV cache depth (ring buffer per layer) */
#define G2_KV_CAP       512

/* ── Per-layer norm weights (float32, loaded from norms.bin float16) ──────── */
typedef struct {
    float pre_attn [G2_HIDDEN];
    float post_attn[G2_HIDDEN];
    float pre_mlp  [G2_HIDDEN];
    float post_ff  [G2_HIDDEN];
} GemmaLayerNorms;

/* ── KV ring buffer for one layer ────────────────────────────────────────── */
typedef struct {
    /* [G2_KV_CAP, G2_N_KV_HEADS, G2_HEAD_DIM] */
    float k[G2_KV_CAP][G2_N_KV_HEADS][G2_HEAD_DIM];
    float v[G2_KV_CAP][G2_N_KV_HEADS][G2_HEAD_DIM];
} GemmaKVCache;

/* ── Full model state ─────────────────────────────────────────────────────── */
typedef struct {
    /* ONIX file (weights mmap'd) */
    OnixFile onix;

    /* Layer projection tensor views (zero-copy into ONIX mmap) */
    OnixTensorView q_proj[G2_N_LAYERS];
    OnixTensorView k_proj[G2_N_LAYERS];
    OnixTensorView v_proj[G2_N_LAYERS];
    OnixTensorView o_proj[G2_N_LAYERS];
    OnixTensorView gate_proj[G2_N_LAYERS];
    OnixTensorView up_proj[G2_N_LAYERS];
    OnixTensorView down_proj[G2_N_LAYERS];

    /* Layer norms (float32 heap, loaded from norms.bin) */
    GemmaLayerNorms norms[G2_N_LAYERS];
    float           final_norm[G2_HIDDEN];

    /* Embedding table (mmap'd float16, accessed read-only) */
    const uint16_t *embed_f16; /* [G2_VOCAB × G2_HIDDEN] float16            */
    void           *embed_mmap;
    size_t          embed_mmap_len;

    /* KV cache (heap-allocated, zeroed at load) */
    GemmaKVCache   *kv[G2_N_LAYERS];

    /* Current sequence position */
    int position;

    /* Ready flag */
    int loaded;
} GemmaModel;

/* ── Public API ──────────────────────────────────────────────────────────── */

/**
 * Return sizeof(GemmaModel). Python ctypes should call this first to allocate
 * a correctly-sized buffer before calling gemma_load().
 */
size_t gemma_sizeof(void);

/**
 * Load model from the three weight files.
 * Returns 0 on success, -1 on error (check stderr).
 *
 * @param model      Pre-allocated GemmaModel (or heap pointer)
 * @param onix_path  Path to gemma2_2b.onix
 * @param norms_path Path to norms.bin
 * @param embed_path Path to embed.bin
 */
int gemma_load(GemmaModel *model,
               const char *onix_path,
               const char *norms_path,
               const char *embed_path);

/**
 * Free all resources (mmap, KV cache heap).
 */
void gemma_free(GemmaModel *model);

/**
 * Reset KV cache and position counter (start a new conversation).
 */
void gemma_reset(GemmaModel *model);

/**
 * Run one autoregressive forward pass for a single token.
 * Updates KV cache and position counter internally.
 *
 * @param model    Loaded model
 * @param token_id Input token id [0, G2_VOCAB)
 * @param logits   Output buffer [G2_VOCAB] float32 — CALLER must allocate
 */
void gemma_forward(GemmaModel *model, int token_id, float *logits);

/**
 * Sample the next token (greedy argmax).
 * Applies final_logit_softcap before argmax.
 *
 * @param logits [G2_VOCAB] float32 from gemma_forward
 * @return token_id with highest softcapped logit
 */
int gemma_argmax(const float *logits);

/**
 * Callback type for gemma_generate.
 * Called after each token is generated.
 * Return 0 to continue, non-zero to stop early.
 */
typedef int (*gemma_token_cb)(int token_id, void *userdata);

/**
 * Generate up to max_tokens tokens starting from prompt_ids.
 * Calls cb after every generated token (not prompt tokens).
 *
 * @param model       Loaded model
 * @param prompt_ids  Array of token ids for the prompt
 * @param prompt_len  Number of prompt tokens
 * @param max_tokens  Maximum new tokens to generate
 * @param cb          Token callback (may be NULL)
 * @param userdata    Passed through to cb
 * @return Number of tokens generated
 */
int gemma_generate(GemmaModel *model,
                   const int *prompt_ids, int prompt_len,
                   int max_tokens,
                   gemma_token_cb cb, void *userdata);

#endif /* GEMMA_INFER_H */
