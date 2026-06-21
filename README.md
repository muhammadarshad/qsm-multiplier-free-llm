# Resurrecting Quarter-Square Arithmetic for Multiplier-Free Large Language Model Inference

This repository contains the official implementation of the Quarter-Square Multiplication (QSM) paradigm for multiplier-free Large Language Model (LLM) inference. 

QSM replaces power-hungry hardware multiplier arrays with L1-cache-friendly lookup tables and adders, achieving a ~20% logic gate area reduction and up to 60% dynamic energy savings in hardware accelerators while preserving 100% mathematical parity (zero perplexity degradation) compared to standard quantized baselines.

## Repository Contents

* `manifold.c`, `manifold.h`: Core QSM matrix multiplication kernels (`libmanifold.dylib`) with optimized 8-row loop unrolling and on-the-fly SIMD weight biasing.
* `gemma_infer.c`, `gemma_infer.h`, `gemma_onix.h`: Autoregressive inference engine executing Gemma 2 2B forward passes natively on CPU.
* `gemma_test.c`: Benchmark smoke test verifying token generation and decode throughput.
* `Makefile`: Compilation tasks for ARM64 Apple Silicon.
* `verify_lossless_qsm.py`: Exhaustive 8-bit integer space verification proving division by 4 in QSM is mathematically lossless.
* `verify_scaling_parity.py`: Dynamic scaling parity verification checking C engine output against float baselines on Gemma-4 8B projections.
* `QSM_PROOF.md`: Mathematical proof of lossless QSM division.
* `PAPER_DRAFT.md`: Full publication draft text.
* `hardware/`: Verilog RTL source files and testbenches comparing QSM vs. standard MAC blocks.

## Getting Started

### 1. Requirements
* macOS (ARM64 Apple Silicon) or Linux with NEON/SIMD support.
* Clang or GCC compiler with OpenMP/GCD capabilities.
* Python 3 with Numpy (for verification scripts).

### 2. Compilation
Compile the shared library:
```bash
make
```

### 3. Verification & Testing
Run the lossless arithmetic check:
```bash
python3 verify_lossless_qsm.py
```

Run the smoke test to measure decode throughput:
```bash
make test
```

## Authors

* **Muhammad Arshad** (Independent Researcher) - ORCID: [0009-0002-1314-0494](https://orcid.org/0009-0002-1314-0494) - GitHub: [muhammadarshad](https://github.com/muhammadarshad)
