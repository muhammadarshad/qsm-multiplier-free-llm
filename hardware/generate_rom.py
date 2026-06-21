#!/usr/bin/env python3
import os

os.makedirs("hardware", exist_ok=True)
with open("hardware/sq_rom.hex", "w") as f:
    for n in range(511):
        f.write(f"{n*n:05x}\n")
print("[init] Generated hardware/sq_rom.hex successfully.")
