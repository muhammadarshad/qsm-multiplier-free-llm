/**
 * gemma_onix.h — ONIX binary model format reader (zero-copy mmap).
 *
 * ONIX layout (all little-endian):
 *   [0..256]         Header (OnixHeader, 256 bytes)
 *   [256..256+N*192] Index entries (N = header.n_tensors, 192 bytes each)
 *   [data_offset..]  Tensor data (16384-byte page-aligned per tensor)
 *
 * Per tensor data block: [xbar_len bytes Xbar] [s_len bytes S] [z_len bytes Z]
 *   Xbar: uint8  [out_feat × n_blocks × block_size]  offset-binary weights
 *   S:    int8   [out_feat]                            per-row exponent
 *   Z:    uint8  [out_feat]                            per-row denominator
 */
#ifndef GEMMA_ONIX_H
#define GEMMA_ONIX_H

#include <stdint.h>
#include <stddef.h>

#define ONIX_HEADER_SIZE      256
#define ONIX_INDEX_ENTRY_SIZE 192
#define ONIX_MAGIC            "ONIX"

/* ── On-disk header (256 bytes) ──────────────────────────────────────────── */
typedef struct {
    char     magic[4];         /* "ONIX"                                     */
    uint16_t version;          /* 1                                           */
    uint8_t  quant_type;       /* 0x01 = HPQ4_U8                             */
    uint8_t  _reserved0;
    uint32_t n_tensors;
    char     model_type[32];   /* e.g. "gemma2"                              */
    uint32_t num_layers;
    uint32_t hidden_size;
    uint32_t vocab_size;
    uint32_t num_heads;
    uint32_t block_size;       /* uniform block_size (may be overridden)     */
    uint8_t  _reserved1[4];
    uint64_t index_offset;     /* always 256                                 */
    uint64_t data_offset;      /* page-aligned start of tensor data          */
    uint64_t file_size;
    uint32_t header_crc32;
    uint8_t  _pad[160];
} OnixHeader;

/* ── On-disk index entry (192 bytes) ─────────────────────────────────────── */
typedef struct {
    char     name[128];        /* null-terminated tensor key                 */
    uint64_t offset;           /* from data_offset (NOT from file start)     */
    uint32_t out_feat;
    uint32_t n_blocks;
    uint32_t block_size;
    uint64_t xbar_len;         /* out_feat × n_blocks × block_size           */
    uint64_t s_len;            /* out_feat                                   */
    uint64_t z_len;            /* out_feat                                   */
    uint8_t  _pad[20];
} OnixIndexEntry;

/* ── In-memory view into one tensor's mmap'd data ────────────────────────── */
typedef struct {
    const uint8_t  *xbar;       /* [out_feat × n_blocks × block_size]        */
    const int8_t   *s_row;      /* [out_feat]  per-row 2^s exponent          */
    const uint8_t  *z_row;      /* [out_feat]  per-row denominator           */
    uint32_t        out_feat;
    uint32_t        n_blocks;
    uint32_t        block_size;
} OnixTensorView;

/* ── Opened ONIX file ────────────────────────────────────────────────────── */
typedef struct {
    void     *mmap_base; /* mmap pointer                                      */
    size_t    mmap_len;  /* total mmap size                                   */
    int       fd;        /* file descriptor                                   */
    uint64_t  data_offset;
    uint32_t  n_tensors;
} OnixFile;

/* Open and mmap an ONIX file. Returns 0 on success, -1 on error. */
int  onix_open(OnixFile *f, const char *path);

/* Close and unmap. */
void onix_close(OnixFile *f);

/**
 * Find tensor by name and return a zero-copy view.
 * Returns 0 on success, -1 if not found.
 */
int onix_find(const OnixFile *f, const char *name, OnixTensorView *out);

#endif /* GEMMA_ONIX_H */
