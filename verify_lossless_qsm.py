#!/usr/bin/env python3
"""
verify_lossless_qsm.py — Exhaustively verifies that Quarter-Square Multiplication
                         on uint8 values has exactly zero mathematical error.
"""

import numpy as np

def build_qsm_table():
    # Size 511 is sufficient for sum of two uint8s (max 510)
    table = np.zeros(511, dtype=np.uint32)
    for n in range(511):
        # n^2 calculated exactly
        table[n] = n * n
    return table

def qsm_multiply(a, b, table):
    # Sum and absolute difference
    s = int(a) + int(b)
    d = abs(int(a) - int(b))
    
    # Lookups
    sq_sum = table[s]
    sq_diff = table[d]
    
    # Division by 4 via shift
    diff = sq_sum - sq_diff
    
    # Assert that the diff of squares is indeed a multiple of 4
    assert diff % 4 == 0, f"Error: diff {diff} is not divisible by 4 for inputs ({a}, {b})"
    
    return int(diff >> 2)

def main():
    print("=" * 60)
    # 1. Initialize lookup table
    print("[init] Building QSM Lookup Table...")
    table = build_qsm_table()
    print(f"Table built with {len(table)} entries. Footprint: {table.nbytes} bytes.")
    
    # 2. Exhaustive search over uint8 space
    print("\n[test] Running exhaustive verification over uint8 space...")
    total_combinations = 256 * 256
    failures = 0
    
    for a in range(256):
        for b in range(256):
            expected = a * b
            actual = qsm_multiply(a, b, table)
            if actual != expected:
                print(f"Mismatch for ({a}, {b}): expected {expected}, got {actual}")
                failures += 1
                
    if failures == 0:
        print(f"✅ Success! Tested all {total_combinations:,} combinations. Error count: 0.")
        print("QSM is 100% mathematically exact and lossless in integer space.")
    else:
        print(f"❌ Failed! Found {failures} mismatches.")
        exit(1)
    print("=" * 60)

if __name__ == "__main__":
    main()
