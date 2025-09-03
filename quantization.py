# quantization.py
from typing import Tuple
import jax
import jax.numpy as jnp
import numpy as np

'''
What it does:
    Forward value: exactly round(x).
    Backward gradient: behaves like the identity (d/dx ≈ 1 everywhere), i.e., the incoming gradient passes straight through.

Why it works:
    Let y = x + stop_grad(round(x) - x).
    Forward: y = round(x) because x + (round(x) - x) = round(x).
    Backward: d/dx [x] = 1, d/dx [stop_grad(...)] = 0 ⇒ total gradient ≈ 1.
    
    
Note: jax.lax.stop_gradient basically stops the gradient flow through that
'''
# --------- Straight-Through Estimator (STE) helpers ---------
def _ste_round(x: jnp.ndarray) -> jnp.ndarray: #This is basically the approximation for differentiating round function
    # Forward: round(x); Backward: 1 (like identity)
    return x + jax.lax.stop_gradient(jnp.round(x) - x)





'''
Intended behavior:
    Forward value: clip(x, lo, hi).
    Backward gradient:
    1 where lo < x < hi (inside the range),
    0 where x ≤ lo or x ≥ hi (saturated).

Example:
    Let x = [ -2.0, -0.5, 0.3, 1.7 ], lo = -1, hi = 1.
    clipped = [ -1.0, -0.5, 0.3, 1.0 ]
    Return: x + stop_grad(clipped - x) → forward equals clipped
    Gradients: pass through (1) for -0.5 and 0.3; zero for -2.0 and 1.7.
'''
def _ste_clip(x: jnp.ndarray, lo: float, hi: float) -> jnp.ndarray: #This is basically the approximation for differentiating clip function
    # Forward: clip; Backward: pass gradients inside range, zero outside
    clipped = jnp.clip(x, lo, hi)
    return x + jax.lax.stop_gradient(clipped - x)





# --------- Fake quant (QAT) used during training ---------
def fake_quantize(x: jnp.ndarray, num_bits: int = 8, eps: float = 1e-12) -> jnp.ndarray:
    """
    Symmetric per-tensor fake quantization to int8-ish range with STE.
    Forward: x_q = scale * round(clip(x/scale))
    Backward: STE lets grads flow.
    """
    qmin = -(2 ** (num_bits - 1))
    qmax =  (2 ** (num_bits - 1)) - 1

    max_abs = jnp.max(jnp.abs(x))
    scale  = jnp.where(max_abs < eps, jnp.array(1.0, x.dtype), max_abs / qmax)
    
    '''jnp.where(condition, x, y) is a vectorized if-else. 
            Input: condition: boolean array (or scalar) ; x, y: arrays (or scalars) of compatible shape 
            For each element: If condition[i] is True, take x[i]. Else, take y[i]
    This case: 
            Here max_abs < eps is a scalar boolean. 
            If it's True → scale = 1.0 
            If it's False → scale = max_abs / qmax
            This avoids dividing by 0 when all elements in x are tiny.
    '''
    
    y = x / scale 
    y = _ste_clip(y, qmin, qmax) #This is clip and round, but we use _ste_clip and _ste_round to tell jax how to compute gradients for this!
    y = _ste_round(y)
    x_q = y * scale
    return x_q

# --------- Export-time real quantization (int arrays + scales) ---------
def quantize_int8_for_export(x: np.ndarray, num_bits: int = 8, eps: float = 1e-12) -> Tuple[np.ndarray, np.float32]:
    """
    Real quantization to int8 with symmetric per-tensor scale.
    Returns (q_int8, scale). Dequantize with: x ≈ scale * q_int8.astype(float32)
    """
    assert x.dtype.kind in "fc", "Expected float array for quantization"
    qmin = -(2 ** (num_bits - 1))
    qmax =  (2 ** (num_bits - 1)) - 1
    max_abs = float(np.max(np.abs(x)))
    scale = np.float32(1.0 if max_abs < eps else max_abs / qmax)
    q = np.clip(np.round(x / scale), qmin, qmax).astype(np.int8)
    return q, scale

def dequantize_int8(q: np.ndarray, scale: np.float32) -> np.ndarray:
    return (q.astype(np.float32) * np.float32(scale))