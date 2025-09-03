import jax
import jax.numpy as jnp

from FNN_architecture import forward
from load_dataset import load_dataset
from save_load_quantized import load_quantized_params, load_unquantized_params
import config as conf

def _to_jax_params(params):
    """Convert list of (W, b) numpy arrays to JAX arrays."""
    return [(jnp.array(W), jnp.array(b)) for (W, b) in params]

def evaluate_on_test_loss(params, batch_size=128):
    """Return average cross-entropy loss across the entire test set."""
    total_loss = 0.0
    total = 0
    for x_np, y_np in load_dataset("test", batch_size=batch_size, shuffle=False):
        x = jnp.array(x_np)
        y = jnp.array(y_np)

        # Forward pass
        logits = forward(params, x)
        labels = jax.nn.one_hot(y, 10)

        # Cross-entropy loss for the batch
        batch_loss = -jnp.sum(labels * jax.nn.log_softmax(logits), axis=1)
        total_loss += float(jnp.sum(batch_loss))
        total += y.shape[0]

    return total_loss / total if total > 0 else float("nan")

def main():
    if conf.QAT_ENABLE:
        # Loads int8 weights + scales and dequantizes to float32
        qparams_np = load_quantized_params("weights_int8.npz")
    else:
        # Loads raw float weights
        qparams_np = load_unquantized_params("weights_unquantized_int8.npz")

    # Convert numpy arrays to JAX arrays for inference
    qparams = _to_jax_params(qparams_np)

    test_loss = evaluate_on_test_loss(qparams, batch_size=128)
    print(f"Validation loss with loaded params: {test_loss:.6f}")

if __name__ == "__main__":
    main()
