/**
 * gemma_infer.c — Gemma 2 2B autoregressive inference engine.
 *
 * Full forward pass entirely in C:
 *   embed_lookup → 26× [RMSNorm → GQA-attn + RoPE + softcap → MLP(GEGLU)] → final-norm → logits
 *
 * Weights: ONIX mmap (HPQ4 uint8) + norms.bin (f16→f32) + embed.bin (f16 mmap)
 * Matmul:  manifold_qsm_matmul() via libmanifold.dylib (GCD-parallel, no BLAS)
 * Memory:  ~55 MB heap (KV cache) + 2 GB mmap (weights, never copied)
 */

#include "gemma_infer.h"
#include "gemma_onix.h"
#include "manifold.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <dispatch/dispatch.h>

#if defined(__ARM_NEON__) || defined(__ARM_NEON)
  #include <arm_neon.h>
#endif

/* ── Forward declarations of internal helpers ─────────────────────────────── */
static void hpq_project(const OnixTensorView *w, const float *x_f32,
                         float *out_f32, int32_t *scratch_i32);
static void rmsnorm_inplace(float *x, const float *gamma, int n);
static void rope_apply(float *qk, int n_heads, int head_dim,
                       int position, float theta);
static float gelu_tanh(float x);
static uint32_t f16_to_bits(const uint16_t *src);
static float f16_to_f32(uint16_t h);

/* ═══════════════════════════════════════════════════════════════════════════
   ONIX FILE — open, mmap, find tensor
   ═══════════════════════════════════════════════════════════════════════════ */

/* Read little-endian integers from a byte buffer without alignment risk. */
static inline uint32_t rd_u32(const uint8_t *b, size_t off) {
    return (uint32_t)b[off]
         | ((uint32_t)b[off+1] << 8)
         | ((uint32_t)b[off+2] << 16)
         | ((uint32_t)b[off+3] << 24);
}
static inline uint64_t rd_u64(const uint8_t *b, size_t off) {
    return (uint64_t)rd_u32(b, off) | ((uint64_t)rd_u32(b, off+4) << 32);
}

int onix_open(OnixFile *f, const char *path) {
    memset(f, 0, sizeof(*f));
    f->fd = open(path, O_RDONLY);
    if (f->fd < 0) { perror("onix_open: open"); return -1; }

    struct stat st;
    if (fstat(f->fd, &st) < 0) { perror("onix_open: fstat"); close(f->fd); return -1; }

    f->mmap_len  = (size_t)st.st_size;
    f->mmap_base = mmap(NULL, f->mmap_len, PROT_READ, MAP_PRIVATE, f->fd, 0);
    if (f->mmap_base == MAP_FAILED) { perror("onix_open: mmap"); close(f->fd); return -1; }

    const uint8_t *b = (const uint8_t *)f->mmap_base;

    /* Verify magic "ONIX" */
    if (memcmp(b, ONIX_MAGIC, 4) != 0) {
        fprintf(stderr, "onix_open: bad magic\n");
        munmap(f->mmap_base, f->mmap_len); close(f->fd); return -1;
    }

    /*
     * Header byte offsets (confirmed against Rust-written binary):
     *   0  magic[4]
     *   4  version    u16
     *   6  quant_type u8
     *   7  _reserved0 u8
     *   8  n_tensors  u32
     *  12  model_type [32]
     *  44  num_layers u32
     *  48  hidden     u32
     *  52  vocab_size u32
     *  56  num_heads  u32
     *  60  block_size u32
     *  64  _reserved1 [4]
     *  68  index_offset u64
     *  76  data_offset  u64
     *  84  file_size    u64
     *  92  crc32        u32
     * [96..255] pad
     */
    f->n_tensors   = rd_u32(b, 8);
    f->data_offset = rd_u64(b, 76);

    /* Advise sequential access for initial index scan */
    madvise(f->mmap_base,
            ONIX_HEADER_SIZE + f->n_tensors * ONIX_INDEX_ENTRY_SIZE,
            MADV_SEQUENTIAL);

    return 0;
}

void onix_close(OnixFile *f) {
    if (f->mmap_base && f->mmap_base != MAP_FAILED)
        munmap(f->mmap_base, f->mmap_len);
    if (f->fd > 0) close(f->fd);
    memset(f, 0, sizeof(*f));
}

int onix_find(const OnixFile *f, const char *name, OnixTensorView *out) {
    const uint8_t *base_ptr = (const uint8_t *)f->mmap_base;

    /*
     * Index entry byte offsets (192 bytes each, start at ONIX_HEADER_SIZE=256):
     *   0   name[128]     null-terminated string
     * 128   offset    u64  (from data_offset)
     * 136   out_feat  u32
     * 140   n_blocks  u32
     * 144   block_size u32
     * 148   xbar_len  u64
     * 156   s_len     u64
     * 164   z_len     u64
     * 172   _pad[20]
     */
    for (uint32_t i = 0; i < f->n_tensors; i++) {
        const uint8_t *e = base_ptr + ONIX_HEADER_SIZE
                         + (size_t)i * ONIX_INDEX_ENTRY_SIZE;
        if (strncmp((const char *)e, name, 127) == 0) {
            uint64_t tensor_offset = rd_u64(e, 128);
            uint32_t out_feat      = rd_u32(e, 136);
            uint32_t n_blocks      = rd_u32(e, 140);
            uint32_t block_size    = rd_u32(e, 144);
            uint64_t xbar_len      = rd_u64(e, 148);
            uint64_t s_len         = rd_u64(e, 156);
            uint64_t z_len         = rd_u64(e, 164);

            const uint8_t *data = base_ptr + f->data_offset + tensor_offset;
            out->xbar       = data;
            out->s_row      = (const int8_t *)(data + xbar_len);
            out->z_row      = (const uint8_t *)(data + xbar_len + s_len);
            out->out_feat   = out_feat;
            out->n_blocks   = n_blocks;
            out->block_size = block_size;
            madvise((void *)data, (size_t)(xbar_len + s_len + z_len), MADV_RANDOM);
            return 0;
        }
    }
    return -1;
}

/* ═══════════════════════════════════════════════════════════════════════════
   LOAD
   ═══════════════════════════════════════════════════════════════════════════ */

int gemma_load(GemmaModel *model,
               const char *onix_path,
               const char *norms_path,
               const char *embed_path)
{
    memset(model, 0, sizeof(*model));

    /* ── 1. Open ONIX ───────────────────────────────────────────────────── */
    fprintf(stderr, "[gemma_load] Opening ONIX: %s\n", onix_path);
    if (onix_open(&model->onix, onix_path) < 0) return -1;
    fprintf(stderr, "[gemma_load] ONIX: %u tensors, data_offset=%llu\n",
            model->onix.n_tensors, (unsigned long long)model->onix.data_offset);

    /* ── 2. Resolve all 26-layer tensor views ───────────────────────────── */
    char key[160];
    for (int li = 0; li < G2_N_LAYERS; li++) {
#define LOAD_PROJ(field, suffix) \
        snprintf(key, sizeof(key), "model.layers.%d." suffix, li); \
        if (onix_find(&model->onix, key, &model->field[li]) < 0) { \
            fprintf(stderr, "[gemma_load] Missing tensor: %s\n", key); \
            onix_close(&model->onix); return -1; \
        }
        LOAD_PROJ(q_proj,    "self_attn.q_proj")
        LOAD_PROJ(k_proj,    "self_attn.k_proj")
        LOAD_PROJ(v_proj,    "self_attn.v_proj")
        LOAD_PROJ(o_proj,    "self_attn.o_proj")
        LOAD_PROJ(gate_proj, "mlp.gate_proj")
        LOAD_PROJ(up_proj,   "mlp.up_proj")
        LOAD_PROJ(down_proj, "mlp.down_proj")
#undef LOAD_PROJ
    }
    fprintf(stderr, "[gemma_load] All projection tensors resolved.\n");

    /* ── 3. Load norms.bin ──────────────────────────────────────────────── */
    fprintf(stderr, "[gemma_load] Loading norms: %s\n", norms_path);
    {
        int fd = open(norms_path, O_RDONLY);
        if (fd < 0) { perror("norms open"); onix_close(&model->onix); return -1; }

        struct stat st; fstat(fd, &st);
        uint8_t *ndata = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
        if (ndata == MAP_FAILED) { perror("norms mmap"); onix_close(&model->onix); return -1; }

        if (memcmp(ndata, "NORM", 4) != 0) {
            fprintf(stderr, "[gemma_load] Bad norms magic\n");
            munmap(ndata, st.st_size); onix_close(&model->onix); return -1;
        }

        /* Header: magic(4) + version(4) + n_layers(4) + hidden(4) = 16 bytes */
        size_t off    = 16;
        size_t hidden_b = G2_HIDDEN * 2; /* float16 bytes per vector */

        /* Inline f16→f32 conversion */
#define READ_NORM(dst) \
        { const uint16_t *src16 = (const uint16_t *)(ndata + off); \
          for (int _i = 0; _i < G2_HIDDEN; _i++) (dst)[_i] = f16_to_f32(src16[_i]); \
          off += hidden_b; }

        for (int li = 0; li < G2_N_LAYERS; li++) {
            READ_NORM(model->norms[li].pre_attn)
            READ_NORM(model->norms[li].post_attn)
            READ_NORM(model->norms[li].pre_mlp)
            READ_NORM(model->norms[li].post_ff)
            off += 8; /* 8-byte sentinel per layer */
        }
        READ_NORM(model->final_norm)
#undef READ_NORM

        munmap(ndata, st.st_size);
    }
    fprintf(stderr, "[gemma_load] Norms loaded.\n");

    /* ── 4. mmap embed.bin ──────────────────────────────────────────────── */
    fprintf(stderr, "[gemma_load] Mapping embed: %s\n", embed_path);
    {
        int fd = open(embed_path, O_RDONLY);
        if (fd < 0) { perror("embed open"); onix_close(&model->onix); return -1; }
        struct stat st; fstat(fd, &st);

        size_t expected = (size_t)G2_VOCAB * G2_HIDDEN * 2;
        if ((size_t)st.st_size < expected) {
            fprintf(stderr, "[gemma_load] embed.bin too small (%zu < %zu)\n",
                    (size_t)st.st_size, expected);
            close(fd); onix_close(&model->onix); return -1;
        }

        model->embed_mmap     = mmap(NULL, (size_t)st.st_size,
                                     PROT_READ, MAP_PRIVATE, fd, 0);
        model->embed_mmap_len = (size_t)st.st_size;
        close(fd);
        if (model->embed_mmap == MAP_FAILED) {
            perror("embed mmap"); onix_close(&model->onix); return -1;
        }
        model->embed_f16 = (const uint16_t *)model->embed_mmap;
        /* Advise random (token lookup pattern) */
        madvise(model->embed_mmap, model->embed_mmap_len, MADV_RANDOM);
    }
    fprintf(stderr, "[gemma_load] Embed mapped (%zu MB).\n",
            model->embed_mmap_len / (1024*1024));

    /* ── 5. Allocate KV cache ────────────────────────────────────────────── */
    for (int li = 0; li < G2_N_LAYERS; li++) {
        model->kv[li] = calloc(1, sizeof(GemmaKVCache));
        if (!model->kv[li]) {
            fprintf(stderr, "[gemma_load] KV cache alloc failed at layer %d\n", li);
            gemma_free(model); return -1;
        }
    }

    /* ── 6. Init manifold QSM LUT ────────────────────────────────────────── */
    manifold_init();

    /* No load-time pre-biasing. Weights remain read-only and will be biased on-the-fly in matmul. */

    model->position = 0;
    model->loaded   = 1;
    fprintf(stderr, "[gemma_load] Ready. KV cap=%d, %.1f MB heap.\n",
            G2_KV_CAP, G2_N_LAYERS * sizeof(GemmaKVCache) / 1e6f);
    return 0;
}

/* Return sizeof(GemmaModel) so Python ctypes can allocate the right buffer. */
size_t gemma_sizeof(void) { return sizeof(GemmaModel); }

void gemma_free(GemmaModel *model) {

    onix_close(&model->onix);
    if (model->embed_mmap && model->embed_mmap != MAP_FAILED)
        munmap(model->embed_mmap, model->embed_mmap_len);
    for (int li = 0; li < G2_N_LAYERS; li++)
        free(model->kv[li]);
    memset(model, 0, sizeof(*model));
}

void gemma_reset(GemmaModel *model) {
    model->position = 0;
    for (int li = 0; li < G2_N_LAYERS; li++)
        if (model->kv[li]) memset(model->kv[li], 0, sizeof(GemmaKVCache));
}

/* ═══════════════════════════════════════════════════════════════════════════
   FORWARD PASS
   ═══════════════════════════════════════════════════════════════════════════ */

void gemma_forward(GemmaModel *model, int token_id, float *logits) {
    /* Scratch buffers on stack/heap for intermediate activations.
       All sized for the largest tensor (G2_INTER = 9216). */
    static float   hidden[G2_HIDDEN];
    static float   normed[G2_HIDDEN];
    static float   q_buf [G2_N_Q_HEADS  * G2_HEAD_DIM];  /* 2048 */
    static float   k_buf [G2_N_KV_HEADS * G2_HEAD_DIM];  /* 1024 */
    static float   v_buf [G2_N_KV_HEADS * G2_HEAD_DIM];  /* 1024 */
    static float   attn_out[G2_N_Q_HEADS * G2_HEAD_DIM]; /* 2048 */
    static float   o_proj_out[G2_HIDDEN];
    static float   gate_out[G2_INTER];
    static float   up_out  [G2_INTER];
    static float   down_out[G2_HIDDEN];
    static int32_t scratch_i32[G2_INTER]; /* largest output dim */

    int pos = model->position;

    /* ── 1. Embedding lookup (float16 → float32, scaled by sqrt(hidden)) ── */
    {
        const uint16_t *row = model->embed_f16 + (size_t)token_id * G2_HIDDEN;
        for (int i = 0; i < G2_HIDDEN; i++)
            hidden[i] = f16_to_f32(row[i]) * G2_EMBED_SCALE;
    }

    /* ── 2. Transformer layers ───────────────────────────────────────────── */
    for (int li = 0; li < G2_N_LAYERS; li++) {
        const GemmaLayerNorms *nrm = &model->norms[li];
        GemmaKVCache          *kvc = model->kv[li];

        /* — 2a. Pre-attention RMSNorm ————————————————————————————————————— */
        memcpy(normed, hidden, sizeof(float) * G2_HIDDEN);
        rmsnorm_inplace(normed, nrm->pre_attn, G2_HIDDEN);

        /* — 2b. Q/K/V projections ————————————————————————————————————————— */
        hpq_project(&model->q_proj[li], normed, q_buf,   scratch_i32);
        hpq_project(&model->k_proj[li], normed, k_buf,   scratch_i32);
        hpq_project(&model->v_proj[li], normed, v_buf,   scratch_i32);

        /* — 2c. RoPE ————————————————————————————————————————————————————— */
        rope_apply(q_buf, G2_N_Q_HEADS,  G2_HEAD_DIM, pos, G2_ROPE_THETA);
        rope_apply(k_buf, G2_N_KV_HEADS, G2_HEAD_DIM, pos, G2_ROPE_THETA);

        /* — 2d. Write K/V into ring cache ————————————————————————————————— */
        int slot = pos % G2_KV_CAP;
        {
            const float *kp = k_buf;
            const float *vp = v_buf;
            for (int kh = 0; kh < G2_N_KV_HEADS; kh++) {
                memcpy(kvc->k[slot][kh], kp, sizeof(float) * G2_HEAD_DIM);
                memcpy(kvc->v[slot][kh], vp, sizeof(float) * G2_HEAD_DIM);
                kp += G2_HEAD_DIM;
                vp += G2_HEAD_DIM;
            }
        }

        /* — 2e. Grouped Query Attention ——————————————————————————————————— */
        int active  = (pos + 1 < G2_KV_CAP) ? pos + 1 : G2_KV_CAP;
        int win_start = (pos + 1) - active;          /* oldest abs position   */
        float scale = 1.0f / sqrtf((float)G2_HEAD_DIM);

        dispatch_queue_t dq = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0);
        dispatch_apply(G2_N_Q_HEADS, dq, ^(size_t qh) {
            float head_scores[G2_KV_CAP];
            int kvh = qh / G2_GQA_RATIO;
            const float *q_head = q_buf + qh * G2_HEAD_DIM;

            /* Compute attention scores for all cached positions */
            float max_score = -1e30f;
            for (int t = 0; t < active; t++) {
                int kslot = (win_start + t) % G2_KV_CAP;
                const float *k_head = kvc->k[kslot][kvh];
                float s = 0.0f;
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
                float32x4_t acc0 = vdupq_n_f32(0.0f);
                float32x4_t acc1 = vdupq_n_f32(0.0f);
                float32x4_t acc2 = vdupq_n_f32(0.0f);
                float32x4_t acc3 = vdupq_n_f32(0.0f);
                for (int d = 0; d < G2_HEAD_DIM; d += 16) {
                    float32x4_t q0 = vld1q_f32(q_head + d);
                    float32x4_t q1 = vld1q_f32(q_head + d + 4);
                    float32x4_t q2 = vld1q_f32(q_head + d + 8);
                    float32x4_t q3 = vld1q_f32(q_head + d + 12);
                    float32x4_t k0 = vld1q_f32(k_head + d);
                    float32x4_t k1 = vld1q_f32(k_head + d + 4);
                    float32x4_t k2 = vld1q_f32(k_head + d + 8);
                    float32x4_t k3 = vld1q_f32(k_head + d + 12);
                    acc0 = vfmaq_f32(acc0, q0, k0);
                    acc1 = vfmaq_f32(acc1, q1, k1);
                    acc2 = vfmaq_f32(acc2, q2, k2);
                    acc3 = vfmaq_f32(acc3, q3, k3);
                }
                float32x4_t acc = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
                s = vaddvq_f32(acc);
#else
                for (int d = 0; d < G2_HEAD_DIM; d++) s += q_head[d] * k_head[d];
#endif
                s *= scale;
                /* Attention logit softcap: tanh(s / cap) * cap */
                s = tanhf(s / G2_ATTN_CAP) * G2_ATTN_CAP;
                head_scores[t] = s;
                if (s > max_score) max_score = s;
            }

            /* Softmax */
            float sum_exp = 0.0f;
            for (int t = 0; t < active; t++) {
                head_scores[t] = expf(head_scores[t] - max_score);
                sum_exp  += head_scores[t];
            }
            float inv_sum = 1.0f / sum_exp;

            /* Weighted sum of V */
            float *out_head = attn_out + qh * G2_HEAD_DIM;
            memset(out_head, 0, sizeof(float) * G2_HEAD_DIM);
            for (int t = 0; t < active; t++) {
                int vslot = (win_start + t) % G2_KV_CAP;
                const float *v_head = kvc->v[vslot][kvh];
                float w = head_scores[t] * inv_sum;
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
                float32x4_t w_vec = vdupq_n_f32(w);
                for (int d = 0; d < G2_HEAD_DIM; d += 16) {
                    float32x4_t o0 = vld1q_f32(out_head + d);
                    float32x4_t o1 = vld1q_f32(out_head + d + 4);
                    float32x4_t o2 = vld1q_f32(out_head + d + 8);
                    float32x4_t o3 = vld1q_f32(out_head + d + 12);
                    float32x4_t v0 = vld1q_f32(v_head + d);
                    float32x4_t v1 = vld1q_f32(v_head + d + 4);
                    float32x4_t v2 = vld1q_f32(v_head + d + 8);
                    float32x4_t v3 = vld1q_f32(v_head + d + 12);
                    vst1q_f32(out_head + d,      vfmaq_f32(o0, w_vec, v0));
                    vst1q_f32(out_head + d + 4,  vfmaq_f32(o1, w_vec, v1));
                    vst1q_f32(out_head + d + 8,  vfmaq_f32(o2, w_vec, v2));
                    vst1q_f32(out_head + d + 12, vfmaq_f32(o3, w_vec, v3));
                }
#else
                for (int d = 0; d < G2_HEAD_DIM; d++)
                    out_head[d] += w * v_head[d];
#endif
            }
        });

        /* — 2f. O-projection + post-attn RMSNorm + residual ———————————— */
        hpq_project(&model->o_proj[li], attn_out, o_proj_out, scratch_i32);

        memcpy(normed, o_proj_out, sizeof(float) * G2_HIDDEN);
        rmsnorm_inplace(normed, nrm->post_attn, G2_HIDDEN);

        /* residual: hidden += normed(o_proj) */
        for (int i = 0; i < G2_HIDDEN; i++) hidden[i] += normed[i];

        /* — 2g. Pre-MLP RMSNorm ————————————————————————————————————————— */
        memcpy(normed, hidden, sizeof(float) * G2_HIDDEN);
        rmsnorm_inplace(normed, nrm->pre_mlp, G2_HIDDEN);

        /* — 2h. MLP: gate × GELU(gate) * up → down ——————————————————————— */
        hpq_project(&model->gate_proj[li], normed, gate_out, scratch_i32);
        hpq_project(&model->up_proj  [li], normed, up_out,   scratch_i32);

        for (int i = 0; i < G2_INTER; i++)
            gate_out[i] = gelu_tanh(gate_out[i]) * up_out[i];

        hpq_project(&model->down_proj[li], gate_out, down_out, scratch_i32);

        /* — 2i. Post-FF RMSNorm + residual ——————————————————————————————— */
        rmsnorm_inplace(down_out, nrm->post_ff, G2_HIDDEN);

        for (int i = 0; i < G2_HIDDEN; i++) hidden[i] += down_out[i];
    }

    /* ── 3. Final RMSNorm ────────────────────────────────────────────────── */
    rmsnorm_inplace(hidden, model->final_norm, G2_HIDDEN);

    /* ── 4. Logits = hidden @ embed_table.T  (tied lm_head) ─────────────── */
    /*                                                                         */
    /*  Chunked dispatch: 256 tasks × 1024 rows each instead of 256K tasks.   */
    /*  Eliminates GCD scheduler overhead while keeping all P-cores saturated. */
    /*  Each chunk processes a contiguous slice of the embedding table with    */
    /*  NEON-vectorised f16→f32 FMA (8 elements/cycle on Apple Silicon).       */
    if (logits) {
        #define LOGIT_CHUNK 1024          /* rows per GCD task           */
        #define LOGIT_TASKS (G2_VOCAB / LOGIT_CHUNK) /* 256 tasks        */

        const float   *h_ptr    = hidden;
        const uint16_t *emb     = model->embed_f16;
        const float    cap_inv  = 1.0f / G2_FINAL_CAP;
        const float    cap      = G2_FINAL_CAP;

        dispatch_queue_t dq = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0);
        dispatch_apply(LOGIT_TASKS, dq, ^(size_t chunk) {
            size_t row_start = chunk * LOGIT_CHUNK;
            size_t row_end   = row_start + LOGIT_CHUNK;

            for (size_t tok = row_start; tok < row_end; tok++) {
                const uint16_t *row = emb + tok * G2_HIDDEN;
                float dot = 0.0f;

#if defined(__ARM_NEON__) || defined(__ARM_NEON)
                float32x4_t acc0 = vdupq_n_f32(0.0f);
                float32x4_t acc1 = vdupq_n_f32(0.0f);
                float32x4_t acc2 = vdupq_n_f32(0.0f);
                float32x4_t acc3 = vdupq_n_f32(0.0f);
                int i = 0;
                for (; i <= G2_HIDDEN - 16; i += 16) {
                    float16x8_t h0 = vld1q_f16((const __fp16 *)(row + i));
                    float16x8_t h1 = vld1q_f16((const __fp16 *)(row + i + 8));
                    float32x4_t f0 = vcvt_f32_f16(vget_low_f16(h0));
                    float32x4_t f1 = vcvt_f32_f16(vget_high_f16(h0));
                    float32x4_t f2 = vcvt_f32_f16(vget_low_f16(h1));
                    float32x4_t f3 = vcvt_f32_f16(vget_high_f16(h1));
                    float32x4_t x0 = vld1q_f32(h_ptr + i);
                    float32x4_t x1 = vld1q_f32(h_ptr + i + 4);
                    float32x4_t x2 = vld1q_f32(h_ptr + i + 8);
                    float32x4_t x3 = vld1q_f32(h_ptr + i + 12);
                    acc0 = vfmaq_f32(acc0, x0, f0);
                    acc1 = vfmaq_f32(acc1, x1, f1);
                    acc2 = vfmaq_f32(acc2, x2, f2);
                    acc3 = vfmaq_f32(acc3, x3, f3);
                }
                float32x4_t acc = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
                dot = vaddvq_f32(acc);
                for (; i < G2_HIDDEN; i++) dot += h_ptr[i] * f16_to_f32(row[i]);
#else
                for (int i = 0; i < G2_HIDDEN; i++) dot += h_ptr[i] * f16_to_f32(row[i]);
#endif
                logits[tok] = tanhf(dot * cap_inv) * cap;
            }
        });

        #undef LOGIT_CHUNK
        #undef LOGIT_TASKS
    }

    model->position = pos + 1;
}

/* ═══════════════════════════════════════════════════════════════════════════
   GENERATION LOOP
   ═══════════════════════════════════════════════════════════════════════════ */

int gemma_argmax(const float *logits) {
    int best = 0;
    float best_val = logits[0];
    for (int i = 1; i < G2_VOCAB; i++) {
        if (logits[i] > best_val) { best_val = logits[i]; best = i; }
    }
    return best;
}

int gemma_generate(GemmaModel *model,
                   const int *prompt_ids, int prompt_len,
                   int max_tokens,
                   gemma_token_cb cb, void *userdata)
{
    /* Heap-allocate logits (G2_VOCAB * 4 = ~1 MB) */
    float *logits = malloc(sizeof(float) * G2_VOCAB);
    if (!logits) { fprintf(stderr, "gemma_generate: OOM\n"); return 0; }

    /* Prefill: run prompt tokens through model (skip logits except last) */
    for (int i = 0; i < prompt_len - 1; i++)
        gemma_forward(model, prompt_ids[i], NULL);
    if (prompt_len > 0)
        gemma_forward(model, prompt_ids[prompt_len - 1], logits);

    /* Autoregressive decode */
    int n_generated = 0;
    int next_token  = gemma_argmax(logits);

    for (int step = 0; step < max_tokens; step++) {
        if (cb && cb(next_token, userdata) != 0) break;
        n_generated++;

        /* EOS tokens: 1 = <eos>, 107 = <end_of_turn> */
        if (next_token == 1 || next_token == 107) break;

        gemma_forward(model, next_token, logits);
        next_token = gemma_argmax(logits);
    }

    free(logits);
    return n_generated;
}

/* ═══════════════════════════════════════════════════════════════════════════
   INTERNAL HELPERS
   ═══════════════════════════════════════════════════════════════════════════ */

/**
 * HPQ4 projection: x_f32 → out_f32 via manifold_qsm_matmul.
 *
 * Step 1: quantize x_f32 to offset-binary uint8 (x_bar).
 *         act_scale = 2^ceil(log2(max|x|/127)), x_bar = round(x/act_scale)+128
 * Step 2: call manifold_qsm_matmul → int32 accumulator per output row.
 * Step 3: dequantize: out[r] = i32[r] * 2^S[r] / Z[r] * act_scale
 */
static void hpq_project(const OnixTensorView *w, const float *x_f32,
                         float *out_f32, int32_t *scratch_i32)
{
    int in_feat = (int)(w->n_blocks * w->block_size);

    /* ── Quantize activation ─────────────────────────────────────────────── */
    float abs_max = 0.0f;
    for (int i = 0; i < in_feat; i++) {
        float v = fabsf(x_f32[i]);
        if (v > abs_max) abs_max = v;
    }

    /* Zero activation → zero output */
    if (abs_max < 1e-30f) {
        memset(out_f32, 0, sizeof(float) * w->out_feat);
        return;
    }

    /* act_scale = 2^ceil(log2(abs_max/127)) */
    float log2_scale = ceilf(log2f(abs_max / 127.0f));
    float act_scale  = exp2f(log2_scale);
    float inv_scale  = 1.0f / act_scale;

    /* Build x_bar = round(x/act_scale) + 128, clamped [0,255] */
    /* Stack is fine for in_feat ≤ 9216 */
    uint8_t x_bar[9216];
    int i = 0;
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
    float32x4_t inv_scale_vec = vdupq_n_f32(inv_scale);
    int32x4_t bias_vec = vdupq_n_s32(128);
    for (; i <= in_feat - 16; i += 16) {
        float32x4_t x0 = vld1q_f32(x_f32 + i);
        float32x4_t x1 = vld1q_f32(x_f32 + i + 4);
        float32x4_t x2 = vld1q_f32(x_f32 + i + 8);
        float32x4_t x3 = vld1q_f32(x_f32 + i + 12);

        int32x4_t r0 = vaddq_s32(vcvtaq_s32_f32(vmulq_f32(x0, inv_scale_vec)), bias_vec);
        int32x4_t r1 = vaddq_s32(vcvtaq_s32_f32(vmulq_f32(x1, inv_scale_vec)), bias_vec);
        int32x4_t r2 = vaddq_s32(vcvtaq_s32_f32(vmulq_f32(x2, inv_scale_vec)), bias_vec);
        int32x4_t r3 = vaddq_s32(vcvtaq_s32_f32(vmulq_f32(x3, inv_scale_vec)), bias_vec);

        int16x8_t q16_0 = vcombine_s16(vqmovn_s32(r0), vqmovn_s32(r1));
        int16x8_t q16_1 = vcombine_s16(vqmovn_s32(r2), vqmovn_s32(r3));

        uint8x8_t q8_0 = vqmovun_s16(q16_0);
        uint8x8_t q8_1 = vqmovun_s16(q16_1);

        vst1q_u8(x_bar + i, vcombine_u8(q8_0, q8_1));
    }
#endif
    for (; i < in_feat; i++) {
        int q = (int)roundf(x_f32[i] * inv_scale) + 128;
        if (q < 0)   q = 0;
        if (q > 255) q = 255;
        x_bar[i] = (uint8_t)q;
    }

    /* ── QSM matmul via manifold kernel ─────────────────────────────────── */
    manifold_qsm_matmul(w->xbar, x_bar, scratch_i32,
                         (uint16_t)w->out_feat,
                         (uint16_t)w->n_blocks,
                         (uint16_t)w->block_size);

    /* ── Dequantize ──────────────────────────────────────────────────────── */
    const int8_t  *s = w->s_row;
    const uint8_t *z = w->z_row;
    for (uint32_t r = 0; r < w->out_feat; r++) {
        int   denom = (int)z[r];
        if (denom == 0) denom = 1;  /* guard: D=1 fast path */
        float w_scale = exp2f((float)s[r]) / (float)denom;
        out_f32[r] = (float)scratch_i32[r] * w_scale * act_scale;
    }
}

/**
 * RMSNorm in-place: x = x / rms(x) * (1 + gamma)
 * Gemma uses additive-bias form: multiply by (1 + γ), not raw γ.
 */
static void rmsnorm_inplace(float *x, const float *gamma, int n) {
    float ss = 0.0f;
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
    float32x4_t ss_acc0 = vdupq_n_f32(0.0f);
    float32x4_t ss_acc1 = vdupq_n_f32(0.0f);
    float32x4_t ss_acc2 = vdupq_n_f32(0.0f);
    float32x4_t ss_acc3 = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i <= n - 16; i += 16) {
        float32x4_t x0 = vld1q_f32(x + i);
        float32x4_t x1 = vld1q_f32(x + i + 4);
        float32x4_t x2 = vld1q_f32(x + i + 8);
        float32x4_t x3 = vld1q_f32(x + i + 12);
        ss_acc0 = vfmaq_f32(ss_acc0, x0, x0);
        ss_acc1 = vfmaq_f32(ss_acc1, x1, x1);
        ss_acc2 = vfmaq_f32(ss_acc2, x2, x2);
        ss_acc3 = vfmaq_f32(ss_acc3, x3, x3);
    }
    float32x4_t ss_acc = vaddq_f32(vaddq_f32(ss_acc0, ss_acc1), vaddq_f32(ss_acc2, ss_acc3));
    ss = vaddvq_f32(ss_acc);
    for (; i < n; i++) ss += x[i] * x[i];
#else
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
#endif

    float rms = sqrtf(ss / (float)n + G2_RMS_EPS);
    float inv = 1.0f / rms;

#if defined(__ARM_NEON__) || defined(__ARM_NEON)
    float32x4_t inv_vec = vdupq_n_f32(inv);
    float32x4_t one_vec = vdupq_n_f32(1.0f);
    i = 0;
    for (; i <= n - 16; i += 16) {
        float32x4_t x0 = vld1q_f32(x + i);
        float32x4_t x1 = vld1q_f32(x + i + 4);
        float32x4_t x2 = vld1q_f32(x + i + 8);
        float32x4_t x3 = vld1q_f32(x + i + 12);

        float32x4_t g0 = vld1q_f32(gamma + i);
        float32x4_t g1 = vld1q_f32(gamma + i + 4);
        float32x4_t g2 = vld1q_f32(gamma + i + 8);
        float32x4_t g3 = vld1q_f32(gamma + i + 12);

        float32x4_t scale0 = vmulq_f32(inv_vec, vaddq_f32(one_vec, g0));
        float32x4_t scale1 = vmulq_f32(inv_vec, vaddq_f32(one_vec, g1));
        float32x4_t scale2 = vmulq_f32(inv_vec, vaddq_f32(one_vec, g2));
        float32x4_t scale3 = vmulq_f32(inv_vec, vaddq_f32(one_vec, g3));

        vst1q_f32(x + i,      vmulq_f32(x0, scale0));
        vst1q_f32(x + i + 4,  vmulq_f32(x1, scale1));
        vst1q_f32(x + i + 8,  vmulq_f32(x2, scale2));
        vst1q_f32(x + i + 12, vmulq_f32(x3, scale3));
    }
    for (; i < n; i++) x[i] = x[i] * inv * (1.0f + gamma[i]);
#else
    for (int i = 0; i < n; i++) x[i] = x[i] * inv * (1.0f + gamma[i]);
#endif
}

/**
 * Apply RoPE rotations with precomputed cos/sin tables.
 * Tables are built once on first call and reused every decode step.
 * Covers positions 0..G2_KV_CAP-1 × dims 0..head_dim/2-1.
 */
static void rope_apply(float *qk, int n_heads, int head_dim,
                        int position, float theta)
{
    /* ── Precomputed table: [8192][G2_HEAD_DIM/2] cos and sin ───────── */
    static float rope_cos[8192][G2_HEAD_DIM / 2];
    static float rope_sin[8192][G2_HEAD_DIM / 2];
    static int   rope_built = 0;

    if (!rope_built) {
        for (int pos = 0; pos < 8192; pos++) {
            for (int i = 0; i < G2_HEAD_DIM / 2; i++) {
                float freq  = 1.0f / powf(G2_ROPE_THETA, (2.0f * i) / G2_HEAD_DIM);
                float angle = (float)pos * freq;
                rope_cos[pos][i] = cosf(angle);
                rope_sin[pos][i] = sinf(angle);
            }
        }
        rope_built = 1;
    }

    int half = head_dim / 2;
    /* Use a slice of the table appropriate for this head_dim (always ≤ G2_HEAD_DIM/2) */
    const float *cos_row = rope_cos[position % 8192];
    const float *sin_row = rope_sin[position % 8192];

    for (int h = 0; h < n_heads; h++) {
        float *head = qk + h * head_dim;
        for (int i = 0; i < half; i++) {
            float c  = cos_row[i];
            float s  = sin_row[i];
            float x0 = head[i];
            float x1 = head[i + half];
            head[i]        = x0 * c - x1 * s;
            head[i + half] = x0 * s + x1 * c;
        }
    }
}

/**
 * GELU with tanh approximation (matches PyTorch gelu_pytorch_tanh).
 * gelu(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715*x³)))
 */
static float gelu_tanh(float x) {
    static const float SQRT2PI = 0.7978845608f;
    float inner = SQRT2PI * (x + 0.044715f * x * x * x);
    /* Clamp before tanh to avoid overflow */
    if (inner >  20.0f) inner =  20.0f;
    if (inner < -20.0f) inner = -20.0f;
    return 0.5f * x * (1.0f + tanhf(inner));
}

/**
 * Convert IEEE 754 float16 to float32.
 * Handles normal, subnormal, inf, nan, signed zero.
 */
static float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t man  = h & 0x3FF;
    uint32_t result;

    if (exp == 0) {
        if (man == 0) {
            result = sign; /* ±0 */
        } else {
            /* Subnormal: normalize */
            exp = 1;
            while (!(man & 0x400)) { man <<= 1; exp--; }
            man &= 0x3FF;
            result = sign | ((exp + (127 - 15)) << 23) | (man << 13);
        }
    } else if (exp == 31) {
        result = sign | 0x7F800000 | (man << 13); /* ±inf or nan */
    } else {
        result = sign | ((exp + (127 - 15)) << 23) | (man << 13);
    }

    float f;
    memcpy(&f, &result, 4);
    return f;
}
