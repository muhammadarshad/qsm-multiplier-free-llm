#!/usr/bin/env python3
"""
verify_scaling_parity.py — Dynamically loads weights from the 8B model (gemma4_8b.onix),
                            runs linear projections of varying dimensions using our
                            C QSM kernel, and compares against the floating-point baseline.
"""

import ctypes
import os
import struct
import numpy as np

# Paths
ONIX_PATH = "/Users/marshad/Projects/hpq-kernel-rust/gemma4_8b.onix"
LIB_PATH = "./libmanifold.dylib"

# ── ctypes initialization ────────────────────────────────────────────────────
if not os.path.exists(LIB_PATH):
    print(f"Error: {LIB_PATH} not found. Run 'make' first.")
    exit(1)

lib = ctypes.CDLL(LIB_PATH)
lib.manifold_init()

lib.manifold_qsm_matmul.argtypes = [
    ctypes.POINTER(ctypes.c_uint8),  # Xbar_w
    ctypes.POINTER(ctypes.c_uint8),  # x_bar_x
    ctypes.POINTER(ctypes.c_int32),  # out
    ctypes.c_uint16,                 # out_feat
    ctypes.c_uint16,                 # n_blocks
    ctypes.c_uint16,                 # block_size
]
lib.manifold_qsm_matmul.restype = None

# ── ONIX Parser ──────────────────────────────────────────────────────────────
class OnixReader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.file = open(filepath, "rb")
        
        # 1. Parse header (256 bytes)
        header = self.file.read(256)
        if header[:4] != b"ONIX":
            raise ValueError("Invalid ONIX magic")
            
        self.version = struct.unpack_from("<H", header, 4)[0]
        self.n_tensors = struct.unpack_from("<I", header, 8)[0]
        self.model_type = header[12:44].decode().strip("\x00")
        self.num_layers = struct.unpack_from("<I", header, 44)[0]
        self.hidden = struct.unpack_from("<I", header, 48)[0]
        self.vocab_size = struct.unpack_from("<I", header, 52)[0]
        self.num_heads = struct.unpack_from("<I", header, 56)[0]
        self.block_size = struct.unpack_from("<I", header, 60)[0]
        
        self.index_offset = struct.unpack_from("<Q", header, 68)[0]
        self.data_offset = struct.unpack_from("<Q", header, 76)[0]
        
        print(f"[onix] Model Type: {self.model_type}")
        print(f"[onix] Layers: {self.num_layers}, Hidden: {self.hidden}, Vocab: {self.vocab_size}")
        print(f"[onix] Tensors in file: {self.n_tensors}")

        # 2. Parse index entries
        self.tensors = {}
        self.file.seek(256)
        for _ in range(self.n_tensors):
            entry = self.file.read(192)
            name = entry[:128].decode().strip("\x00")
            offset = struct.unpack_from("<Q", entry, 128)[0]
            out_feat = struct.unpack_from("<I", entry, 136)[0]
            n_blocks = struct.unpack_from("<I", entry, 140)[0]
            block_size = struct.unpack_from("<I", entry, 144)[0]
            xbar_len = struct.unpack_from("<Q", entry, 148)[0]
            s_len = struct.unpack_from("<Q", entry, 156)[0]
            z_len = struct.unpack_from("<Q", entry, 164)[0]
            
            self.tensors[name] = {
                "offset": offset,
                "out_feat": out_feat,
                "n_blocks": n_blocks,
                "block_size": block_size,
                "xbar_len": xbar_len,
                "s_len": s_len,
                "z_len": z_len
            }

    def load_tensor(self, name):
        meta = self.tensors.get(name)
        if not meta:
            raise KeyError(f"Tensor {name} not found")
            
        self.file.seek(self.data_offset + meta["offset"])
        xbar_data = self.file.read(meta["xbar_len"])
        s_data = self.file.read(meta["s_len"])
        z_data = self.file.read(meta["z_len"])
        
        Xbar = np.frombuffer(xbar_data, dtype=np.uint8).reshape(meta["out_feat"], meta["n_blocks"], meta["block_size"])
        S = np.frombuffer(s_data, dtype=np.int8)
        Z = np.frombuffer(z_data, dtype=np.uint8)
        
        return Xbar, S, Z

    def close(self):
        self.file.close()

# ── Dynamic validation logic ─────────────────────────────────────────────────
def verify_tensor_projection(reader, name):
    print(f"\n--- Testing Tensor: {name} ---")
    Xbar, S, Z = reader.load_tensor(name)
    out_feat, n_blocks, block_size = Xbar.shape
    in_feat = n_blocks * block_size
    print(f"Dimensions: {out_feat} x {in_feat} (blocks: {n_blocks}, block_size: {block_size})")

    # 1. Generate random input activation
    np.random.seed(42)  # For reproducibility
    x_f32 = np.random.randn(in_feat).astype(np.float32) * 2.0
    
    # 2. Baseline floating-point quantization projection
    abs_max = np.abs(x_f32).max()
    s_exp = int(np.ceil(np.log2(abs_max / 127.0)))
    act_scale = 2.0 ** s_exp
    x_int = np.round(x_f32 / act_scale).clip(-128, 127).astype(np.int32)
    
    W_float = (Xbar.reshape(out_feat, in_feat).astype(np.float32) - 128.0)
    w_scales = (2.0 ** S.astype(np.float32)) / np.where(Z == 0, 1.0, Z.astype(np.float32))
    expected = (W_float @ x_int) * w_scales * act_scale

    # 3. C QSM library execution
    x_bar = (x_int + 128).astype(np.uint8)
    out_int32 = np.zeros(out_feat, dtype=np.int32)
    
    x_ptr = x_bar.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    
    w_ptr = np.ascontiguousarray(Xbar).ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        
    out_ptr = out_int32.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
    
    lib.manifold_qsm_matmul(w_ptr, x_ptr, out_ptr, out_feat, n_blocks, block_size)
    
    # 4. Dequantize C output
    actual = out_int32.astype(np.float32) * w_scales * act_scale

    # 5. Assert equality
    max_diff = np.abs(actual - expected).max()
    print(f"Max absolute difference: {max_diff:.6e}")
    assert max_diff < 1e-5, f"Validation failed for {name}! Mismatch exceeds threshold."
    print("✅ Parity Verified!")

def main():
    print("=" * 70)
    print(" Dynamic Scaling Verification: QSM C Kernel on 8B Model Projections")
    print("=" * 70)
    
    if not os.path.exists(ONIX_PATH):
        print(f"Error: {ONIX_PATH} not found.")
        exit(1)
        
    reader = OnixReader(ONIX_PATH)
    
    # Select sample projection layers of different shapes from Gemma 4 8B
    # Let's inspect the names of first few tensors to make sure we load the right names
    print("\nSelecting tensors for verification...")
    tensors_to_test = []
    
    # Look for self_attn.q_proj, self_attn.o_proj, and mlp.down_proj layers from layer 0 and layer 15
    candidates = [
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.mlp.down_proj",
        "model.layers.15.self_attn.q_proj",
        "model.layers.15.mlp.down_proj"
    ]
    for c in candidates:
        if c in reader.tensors:
            tensors_to_test.append(c)
            
    if not tensors_to_test:
        # Fallback: select first 3 tensors in index
        tensors_to_test = list(reader.tensors.keys())[:3]
        
    print(f"Tensors selected: {tensors_to_test}")
    
    for name in tensors_to_test:
        verify_tensor_projection(reader, name)
        
    reader.close()
    print("\n" + "=" * 70)
    print("✅ Success! All dynamic scaling tests passed with exact parity.")
    print("=" * 70)

if __name__ == "__main__":
    main()
