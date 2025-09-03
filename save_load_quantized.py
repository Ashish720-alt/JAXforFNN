# save_load_quantized.py
import numpy as np
from quantization import quantize_int8_for_export, dequantize_int8

def save_quantized_params(params, filename):
    """
    Overwrites existing file. Saves per-layer:
      - W{i}_q: int8 weights
      - W{i}_scale: float32 scale for W
      - b{i}: float32 biases (kept in float to keep it simple)
    """
    arrays = {}
    for i, (W, b) in enumerate(params):
        W_np = np.array(W)  # device -> host
        b_np = np.array(b)
        W_q, W_scale = quantize_int8_for_export(W_np, num_bits=8)
        arrays[f"W{i}_q"] = W_q
        arrays[f"W{i}_scale"] = np.array(W_scale, dtype=np.float32)
        arrays[f"b{i}"] = b_np.astype(np.float32)
    np.savez(filename, **arrays)  # overwrites

def load_quantized_params(filename):
    """
    Loads int8 weights and dequantizes back to float32 for inference with your existing forward().
    Returns list of (W_float32, b_float32).
    """
    data = np.load(filename)
    # infer layer count
    n_layers = len([k for k in data.files if k.startswith("W") and k.endswith("_q")])
    params = []
    for i in range(n_layers):
        W_q = data[f"W{i}_q"]
        W_scale = data[f"W{i}_scale"].item() if data[f"W{i}_scale"].shape == () else float(data[f"W{i}_scale"])
        b = data[f"b{i}"]
        W = dequantize_int8(W_q, W_scale)
        params.append((W, b))
    return params

def save_unquantized_params(params, filename, dtype=np.float32):
    arrays = {}
    for i, (W, b) in enumerate(params):
        W_np = np.array(W).astype(dtype, copy=False)
        b_np = np.array(b).astype(dtype, copy=False)
        arrays[f"W{i}"] = W_np
        arrays[f"b{i}"] = b_np
    np.savez(filename, **arrays)  # overwrite


def load_unquantized_params(filename):
    data = np.load(filename)
    # infer number of layers from stored keys
    n_layers = len([k for k in data.files if k.startswith("W") and not k.endswith("_q")])
    params = []
    for i in range(n_layers):
        W = data[f"W{i}"]
        b = data[f"b{i}"]
        params.append((W, b))
    return params