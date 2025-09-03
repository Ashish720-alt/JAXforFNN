import jax
from jax import grad, jit
import jax.numpy as jnp
from FNN_architecture import loss_fn, accuracy
import config as conf
from load_dataset import load_dataset
import numpy as np

'''
jit = Just-In-Time compilation.
It takes your pure Python function and compiles it into highly optimized XLA code (runs on CPU/GPU/TPU).
The first time you call update, JAX traces the function, builds a computation graph, and compiles it.
Subsequent calls run the compiled version, which is much faster than Python.
So in effect:
Without @jit: JAX runs line-by-line in Python (slower).
With @jit: JAX fuses ops, eliminates Python overhead, and runs at near-C speed.
'''


#One gradient descent update step.    
@jit
def update(params, x, y, lr):
    '''
    JAX grad explained:

    - grad is JAX's automatic differentiation function.
    - You give it a Python function (e.g. loss_fn) and it returns another function
    that computes the gradient w.r.t. the first argument of the loss_fn i.e. params.

    So:
        grad(loss_fn)          -> new function f(params, x, y) = ∂loss_fn/∂params
        grad(loss_fn)(params, x, y)  -> evaluates ∂loss_fn/∂params(params, x, y) internally.

    - In our code, params is not a single array.
    It is a PyTree (JAX's term for nested structures of arrays), here:
        [(W1, b1), (W2, b2), (W3, b3), ...]
    where each Wi and bi are JAX arrays.

    - JAX is PyTree-aware:
    grad walks through every leaf array in params, computes ∂loss/∂Wi and ∂loss/∂bi,
    and returns the gradients in the same structure:
        [(dW1, db1), (dW2, db2), (dW3, db3), ...]

    - Each dWi and dbi has the same shape as Wi and bi, so zip(params, grads)
    lines them up perfectly for the update step.
    '''

    '''
    Understanding grad(loss_fn) in our 3-layer FNN:

    1. Forward pass we coded:
    - Hidden layers: h = tanh(Wx + b)
    - Final layer:   z = W h + b   (logits, no activation here)
    - Loss:          cross-entropy with softmax (via log_softmax)

    So the network is: affine → tanh → affine → tanh → affine → softmax → CE loss.

    2. What grad(loss_fn) does:
    - JAX traces all ops inside loss_fn (dot products, tanh, log_softmax, mean).
    - It applies the chain rule (reverse-mode autodiff) to compute:
        ∂L/∂Wi , ∂L/∂bi   for every layer i
    - You don't write backprop manually — JAX builds it automatically.

    Example chain:
        dL/dz   = softmax(z) - y
        dL/dW3  = h2^T · dL/dz
        dL/db3  = dL/dz
        dL/dh2  = (dL/dz) · W3^T
        dL/du2  = dL/dh2 ⊙ (1 - tanh^2(u2))
        etc. (back to W2,b2 and W1,b1)

    3. Does JAX assume tanh + softmax?
    - No. JAX only differentiates the exact operations you code in loss_fn.
    - In our case, we wrote tanh + log_softmax, so grad follows that.
    - If we swapped tanh → relu or CE → MSE, grad would propagate through
        those instead (no extra changes needed).

    In plain words:
    JAX does not assume softmax or cross-entropy — it just applies the chain rule
    through whatever functions you coded. In our FNN, that happens to be
    tanh hidden layers and softmax + cross-entropy at the output.
    '''

    
    grads = grad(loss_fn)(params, x, y)
    new_params = [(W - lr*dW, b - lr*db)
                  for (W, b), (dW, db) in zip(params, grads)]
    return new_params


def _clone_params(params):
    # Deep-copy a PyTree of arrays so "best" isn’t mutated by future updates
    return jax.tree_util.tree_map(lambda a: jnp.array(a, copy=True), params)

def _eval_on_dataset(params, split, batch_size=128):
    """
    Compute average loss & accuracy over the dataset split.
    Recreates the generator each call to avoid exhaustion.
    """
    losses = []
    accs = []
    for x, y in load_dataset(split, batch_size=batch_size, shuffle=False):
        # loss
        l = loss_fn(params, x, y)
        losses.append(np.array(l))  # move to host for numeric ops
        # accuracy
        a = accuracy(params, x, y)
        accs.append(np.array(a))
    if len(losses) == 0:
        return np.nan, np.nan
    return float(np.mean(losses)), float(np.mean(accs))



#lr is the learning rate
def train(params, lr=conf.LR, epochs=conf.MAX_EPOCHS, batch_size=conf.BATCH_SIZE, patience=conf.LOSS_PATIENCE, min_delta=conf.LOSS_MIN_DELTA, monitor=conf.METRIC):
    best_params = _clone_params(params)
    best_metric = np.inf if monitor == "val_loss" else -np.inf
    epochs_without_improvement = 0

    epoch = 1
    while (epoch < epochs + 1):
        # ---- Training epoch (fresh generator) ----
        for x, y in load_dataset("train", batch_size=batch_size, shuffle=True):
            params = update(params, x, y, lr)

        # ---- Validation / early-stopping metrics ----
        val_loss, val_acc = _eval_on_dataset(params, split="test", batch_size=batch_size)

        # Console log too
        print(f"Epoch {epoch:02d} | val_loss: {val_loss:.6f} | val_acc: {val_acc:.4f}")

        # ---- Check improvement ----
        if monitor == "val_loss":
            improved = (best_metric - val_loss) > min_delta
            current = val_loss
        else:  # monitor == "val_acc"
            improved = (val_acc - best_metric) > min_delta
            current = val_acc

        if improved:
            best_metric = current
            best_params = _clone_params(params)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # ---- Early stop? ----
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch} (no {monitor} improvement for {patience} epochs).")
            break

        epoch+=1

    return best_params, best_metric, epoch