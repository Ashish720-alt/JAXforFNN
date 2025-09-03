import matplotlib.pyplot as plt
import numpy as np

# Data
'''
QAT_Training -> Epoch 35 | val_loss: 0.059968 | val_vloss: 0.9816
QAT_Inference -> Validation loss with loaded params: 0.060633
'''
quantized_vloss_loss = 1.11  # as percentage
quantized_space = 238.5      # in kB

'''
no_quantization_training -> Epoch 27 | val_loss: 0.061590 | val_vloss: 0.9794
no_quantization_inference -> Validation loss with loaded params: 0.062269
'''
unquantized_vloss_loss = 1.10  # as percentage
unquantized_space = 942.0      # in kB

# Total parameter count (for [784, 256, 128, 10])
total_params = 235146  

labels = ["Quantized", "Unquantized"]
vloss_losses = [quantized_vloss_loss, unquantized_vloss_loss]
sizes = [quantized_space, unquantized_space]

x = np.arange(len(labels))  # the label locations
width = 0.5  # bar width

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# --- Subplot 1: Validation Loss (%) ---
rects1 = ax1.bar(x, vloss_losses, width, color="skyblue")
ax1.set_ylabel("Validation Loss Change (%)")
ax1.set_title("Validation Loss")
ax1.set_xticks(x)
ax1.set_xticklabels(labels)

for rect in rects1:
    height = rect.get_height()
    ax1.annotate(f"{height:.2f}",
                 xy=(rect.get_x() + rect.get_width() / 2, height),
                 xytext=(0, 3),
                 textcoords="offset points",
                 ha="center", va="bottom")

# --- Subplot 2: Model Size (kB) ---
rects2 = ax2.bar(x, sizes, width, color="lightgreen")
ax2.set_ylabel("Model Size (kB)")
ax2.set_title("Model Size")
ax2.set_xticks(x)
ax2.set_xticklabels(labels)

for rect in rects2:
    height = rect.get_height()
    ax2.annotate(f"{height:.0f}",
                 xy=(rect.get_x() + rect.get_width() / 2, height),
                 xytext=(0, 3),
                 textcoords="offset points",
                 ha="center", va="bottom")

# --- Add total parameter count as a suptitle or text box ---
fig.suptitle(f"Total Parameters ≈ {total_params:,}", fontsize=12, fontweight="bold")

# Alternative: put as caption text below plots
# fig.text(0.5, 0.01, f"Total Parameters ≈ {total_params:,}", ha="center", fontsize=10)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # leave room for suptitle
plt.savefig("quant_vs_unquant_subplots.png")
plt.close()
