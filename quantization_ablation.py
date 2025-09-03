import matplotlib.pyplot as plt
import numpy as np

# Data
quantized_acc_change = -1.4 
quantized_space = 1  # MB

unquantized_acc_change = 0 
unquantized_space = 4  # MB

labels = ["Quantized", "Unquantized"]
acc_changes = [quantized_acc_change, unquantized_acc_change]
sizes = [quantized_space, unquantized_space]

x = np.arange(len(labels))  # the label locations
width = 0.35  # width of each bar

fig, ax = plt.subplots(figsize=(6,4))

# Bars
rects1 = ax.bar(x - width/2, acc_changes, width, label="Accuracy Change (%)")
rects2 = ax.bar(x + width/2, sizes, width, label="Model Size (MB)")

# Labels and formatting
ax.set_ylabel("Value")
ax.set_title("Quantized vs Unquantized: Accuracy Change & Model Size")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Annotate bars with values
def autolabel(rects, fmt="{:.1f}"):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(fmt.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center", va="bottom")

autolabel(rects1)
autolabel(rects2, fmt="{:.0f}")

plt.tight_layout()
plt.savefig("quant_vs_unquant_hist.png")
plt.close()