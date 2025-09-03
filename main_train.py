# main.py
import jax
from FNN_architecture import init_params
from training import train
from save_load_quantized import save_quantized_params, save_unquantized_params
import config as conf
import matplotlib.pyplot as plt
import jax.numpy as jnp

def main():
    key = jax.random.PRNGKey(42) #randomized key required only for initializing the MLP parameters
    layer_sizes = [784, 256, 128, 10]  # 2 hidden layers, input layer dimension = 28*28 = 784
    params = init_params(key, layer_sizes)
    # Train
    trained_params, _, _ = train(params)
    if (conf.QAT_ENABLE):
        save_quantized_params(trained_params, "weights_int8.npz") 
    else:
        save_unquantized_params(trained_params, "weights_unquantized_int8.npz") 
    
def lr_ablation():
    # Recall to set conf.QAT_ENABLE
    def clone_params(p):
        # make brand-new arrays so nothing is aliased
        return jax.tree_util.tree_map(lambda a: jnp.array(a, copy=True), p)

    key = jax.random.PRNGKey(42) 
    layer_sizes = [784, 256, 128, 10]  
    params0 = init_params(key, layer_sizes)
    graph_vals = {}
    
    for learn_rate in [1, 1e-1, 1e-2, 1e-3, 1e-4]:
        print(f"\n\nLearn rate is {learn_rate}.\n")
        params = clone_params(params0)
        # NOTE: now we assume train returns (params, best_val_loss, epochs)
        _, best_val_loss, epochs = train(params, lr=learn_rate, epochs=conf.MAX_EPOCHS)
        graph_vals[learn_rate] = (best_val_loss, epochs)

    #---- Print IR results -----
    print(graph_vals)

    # --- Plotting ---
    lrs = list(graph_vals.keys())
    val_losses = [graph_vals[lr][0] for lr in lrs]
    epochs_used = [graph_vals[lr][1] for lr in lrs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # === Plot 1: Epochs vs Learning Rate ===
    ax1.plot(lrs, epochs_used, marker="o")
    ax1.set_xscale("log")
    ax1.set_xlabel("Learning rate (log scale)")
    ax1.set_ylabel("Epochs until stop")
    ax1.set_title("Epochs vs Learning Rate")
    ax1.grid(True, alpha=0.3)

    # === Plot 2: Validation Loss vs Learning Rate ===
    ax2.plot(lrs, val_losses, marker="o")
    ax2.set_xscale("log")
    ax2.set_yscale("log")  # highlight small values despite outliers
    ax2.set_xlabel("Learning rate (log scale)")
    ax2.set_ylabel("Validation loss (log scale)")
    ax2.set_title("Validation Loss vs Learning Rate (log y)")
    ax2.grid(True, which="both", alpha=0.3)

    # Annotate points with their values
    for x, y in zip(lrs, val_losses):
        ax2.annotate(f"{y:.3g}", (x, y), textcoords="offset points", xytext=(0, 6),
                     ha="center", fontsize=8)

    # Inset to zoom into small-loss region
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    inset = inset_axes(ax2, width="55%", height="55%", loc="upper right", borderpad=1)
    inset.plot(lrs, val_losses, marker="o")
    inset.set_xscale("log")
    inset.set_ylim(min(val_losses)*0.8, max([v for v in val_losses if v < 1])*1.2)
    inset.set_xlim(min(lrs), max(lrs))
    inset.grid(True, alpha=0.3)
    inset.tick_params(labelsize=8)
    inset.set_title("Zoom", fontsize=9)

    plt.tight_layout()
    plt.savefig("lr_ablation.png")   # save figure with both subplots

    return graph_vals


if __name__ == "__main__":
    if (conf.LR_ABLATION_MODE):
        lr_ablation() #Set conf.QAT_ENABLE to True
    else:
        main() #Compute for both QAT and non-quantized version
