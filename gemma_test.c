/**
 * gemma_test.c — Smoke test for the Gemma 2 2B C inference engine.
 *
 * Runs:
 *   1. ONIX open + tensor resolution
 *   2. norms.bin + embed.bin load
 *   3. Single forward pass (1 token) and argmax
 *   4. 5-token greedy generation with a hard-coded prompt (Paris question)
 *   5. Timing report
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "gemma_infer.h"

/* Hard-coded Gemma2 tokenizer output for:
 * "<bos><start_of_turn>user\nWhat is the capital of France?<end_of_turn>\n<start_of_turn>model\n"
 * (same fallback token list as run_gemma_manifold.py)
 */
static const int PROMPT[] = {
    2, 106, 1645, 108, 4421, 603, 573, 4016, 575, 7837, 1373, 107, 108, 106, 2516, 108
};
static const int PROMPT_LEN = 16;

/* Paths to weights */
#define ONIX_PATH  "/Users/marshad/Projects/hpq-kernel-rust/gemma2_2b.onix"
#define NORMS_PATH "/Users/marshad/Projects/mprc-scratchpad/hpq_kernel/weights_2b/norms.bin"
#define EMBED_PATH "/Users/marshad/Projects/mprc-scratchpad/hpq_kernel/weights_2b/embed.bin"

/* Token callback: print id and timing */
static int on_token(int token_id, void *userdata) {
    int *count = (int *)userdata;
    printf("  token[%d] = %d\n", *count, token_id);
    (*count)++;
    return 0;
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(void) {
    printf("==========================================================\n");
    printf("  Gemma 2 2B — C Inference Engine Smoke Test\n");
    printf("==========================================================\n\n");

    /* ── 1. Allocate and load model ──────────────────────────────────────── */
    GemmaModel *model = calloc(1, sizeof(GemmaModel));
    if (!model) { fprintf(stderr, "OOM\n"); return 1; }

    printf("[1] Loading model weights...\n");
    double t0 = now_sec();

    if (gemma_load(model, ONIX_PATH, NORMS_PATH, EMBED_PATH) < 0) {
        fprintf(stderr, "FAILED: gemma_load\n");
        free(model); return 1;
    }

    double t_load = now_sec() - t0;
    printf("[1] Load time: %.2f s\n\n", t_load);

    /* ── 2. Single forward pass ───────────────────────────────────────────── */
    printf("[2] Single forward pass (token_id=2, <bos>)...\n");
    float *logits = malloc(sizeof(float) * G2_VOCAB);
    if (!logits) { fprintf(stderr, "OOM logits\n"); free(model); return 1; }

    t0 = now_sec();
    gemma_forward(model, 2, logits);
    double t_fwd = now_sec() - t0;

    int top = gemma_argmax(logits);
    printf("[2] Forward: %.2f s  |  argmax=%d  logit[%d]=%.4f\n\n",
           t_fwd, top, top, logits[top]);

    /* ── 3. Generate 8 tokens from the Paris prompt ───────────────────────── */
    printf("[3] Generating 8 tokens from Paris prompt (%d tok prefill)...\n",
           PROMPT_LEN);
    gemma_reset(model);
    int count = 0;

    double t_prefill_start = now_sec();
    /* Prefill phase (process prompt tokens) */
    float *prefill_logits = malloc(sizeof(float) * G2_VOCAB);
    if (!prefill_logits) { fprintf(stderr, "OOM prefill\n"); return 1; }
    for (int i = 0; i < PROMPT_LEN - 1; i++) {
        gemma_forward(model, PROMPT[i], NULL);
    }
    if (PROMPT_LEN > 0) {
        gemma_forward(model, PROMPT[PROMPT_LEN - 1], prefill_logits);
    }
    double t_prefill = now_sec() - t_prefill_start;

    /* Decode phase (autoregressive generation) */
    int next_token = gemma_argmax(prefill_logits);
    free(prefill_logits);

    double t_decode_start = now_sec();
    int n = 0;
    for (int step = 0; step < 8; step++) {
        if (on_token(next_token, &count) != 0) break;
        n++;
        if (next_token == 1 || next_token == 107) break;
        gemma_forward(model, next_token, logits);
        next_token = gemma_argmax(logits);
    }
    double t_decode = now_sec() - t_decode_start;

    printf("\n[3] Prefill completed in %.2f s (%.2f tok/s)\n",
           t_prefill, (double)PROMPT_LEN / t_prefill);
    printf("[3] Generated %d tokens in %.2f s (%.2f tok/s decode)\n\n",
           n, t_decode, (double)n / t_decode);

    /* ── 4. Throughput summary ────────────────────────────────────────────── */
    printf("==========================================================\n");
    printf("  SUMMARY\n");
    printf("  Load:       %.2f s\n", t_load);
    printf("  Prefill:    %.2f s  (%.2f tok/s, %d tokens)\n",
           t_prefill, (double)PROMPT_LEN / t_prefill, PROMPT_LEN);
    printf("  Decode:     %.2f s  (%.2f tok/s, %d tokens)\n",
           t_decode, (double)n / t_decode, n);
    printf("  Throughput: %.2f tok/s (decode)\n", (double)n / t_decode);
    printf("==========================================================\n");

    free(logits);
    gemma_free(model);
    free(model);
    return 0;
}
