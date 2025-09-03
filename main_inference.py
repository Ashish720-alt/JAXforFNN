import jax
import jax.numpy as jnp

from FNN_architecture import forward  # or use accuracy if you prefer per-batch means
from load_dataset import load_dataset
from save_load_quantized import load_quantized_params, load_unquantized_params
import config as conf

def _to_jax_params(params):
    """Convert list of (W, b) numpy arrays to JAX arrays."""
    return [(jnp.array(W), jnp.array(b)) for (W, b) in params]

def evaluate_on_test(params, batch_size=128):
    """Return overall accuracy across the entire test set."""
    total_correct = 0
    total = 0
    for x_np, y_np in load_dataset("test", batch_size=batch_size, shuffle=False):
        x = jnp.array(x_np)
        y = jnp.array(y_np)
        logits = forward(params, x)
        preds = jnp.argmax(logits, axis=1)
        total_correct += int(jnp.sum(preds == y))
        total += y.shape[0]
    return total_correct / total if total > 0 else float("nan")

def main():
    if conf.QAT_ENABLE:
        # Loads int8 weights + scales and dequantizes to float32
        qparams_np = load_quantized_params("weights_int8.npz")
    else:
        # Loads raw float weights
        qparams_np = load_unquantized_params("weights_unquantized_int8.npz")

    # Convert numpy arrays to JAX arrays for inference
    qparams = _to_jax_params(qparams_np)

    test_acc = evaluate_on_test(qparams, batch_size=128)
    print(f"Test accuracy with loaded params: {test_acc:.4f}")

if __name__ == "__main__":
    main()