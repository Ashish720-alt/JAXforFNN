# main.py
import jax
from FNN_architecture import init_params
from training import train
from save_load_quantized import save_quantized_params, save_unquantized_params
import config as conf
import matplotlib.pyplot as plt

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
    #Recall to set conf.QAT_ENABLE
    key = jax.random.PRNGKey(42) 
    layer_sizes = [784, 256, 128, 10]  
    params = init_params(key, layer_sizes)

    graph_vals = {}
    
    for learn_rate in [1, 1e-1, 1e-2, 1e-3, 1e-4]:
        print(f"\n\nLearn rate is {learn_rate}.\n")
        _, best_acc, epochs = train(params, lr = learn_rate, epochs= conf.MAX_EPOCHS)
        graph_vals[learn_rate] = (best_acc, epochs)

    # --- Plotting ---
    lrs = list(graph_vals.keys())
    best_accs = [graph_vals[lr][0] for lr in lrs]
    epochs_used = [graph_vals[lr][1] for lr in lrs]

    plt.figure(figsize=(10,4))

    # Plot 1: Epochs vs learning rate
    plt.subplot(1,2,1)
    plt.plot(lrs, epochs_used, marker="o")
    plt.xscale("log")
    plt.xlabel("Learning rate (log scale)")
    plt.ylabel("Epochs until stop")
    plt.title("Epochs vs Learning Rate")

    # Plot 2: Best accuracy vs learning rate
    plt.subplot(1,2,2)
    plt.plot(lrs, best_accs, marker="o")
    plt.xscale("log")
    plt.xlabel("Learning rate (log scale)")
    plt.ylabel("Best Validation Accuracy")
    plt.title("Best Acc vs Learning Rate")

    plt.tight_layout()
    plt.savefig("lr_ablation.png")   # save instead of show

    return graph_vals


if __name__ == "__main__":
    if (conf.LR_ABLATION_MODE):
        lr_ablation() #Set conf.QAT_ENABLE to True
    else:
        main() #Compute for both QAT and non-quantized version
