# 🚀 Run Code
1. **Training**  
   - First set the flag `QAT_ENABLE` in [`config.py`](config.py) depending on whether you want quantization.  
   - Then run:  
     ```bash
     python3 main_train.py
     ```  
   - This will train the weights and store them in:  
     - Quantized: **`weights_int8.npz`**  
     - Unquantized: **`weights_unquantized_int8.npz`**

2. **Inference**  
   - Again, set `QAT_ENABLE` depending on whether you want to load quantized vs unquantized weights.  
   - Then run:  
     ```bash
     python3 main_inference.py
     ```

3. **Learning Rate Ablation Study**  
   - Set `LR_ABLATION_MODE = True` in [`config.py`](config.py)  
   - Then run:  
     ```bash
     python3 main_train.py
     ```

---

# 📊 Results
- 📈 See **[`lr_ablation.png`](lr_ablation.png)** for results of the ablation study on learning rate, comparing *epochs* vs *validation loss*.  
- 📉 See **[`quant_vs_unquant_subplots.png`](quant_vs_unquant_subplots.png)** for the ablation study comparing *validation loss* and *model size* of quantized vs unquantized weights.

---

# 🧠 Theory
1. Used **[JAX](https://github.com/google/jax)** framework for training a simple MLP architecture.  
2. Weights initialized with **Xavier–He initialization**.  
3. Trained a **2-hidden-layer MLP** with **~235,000 parameters** using **SGD + early stopping** to classify MNIST digits *(0–9)*.  
4. Conducted ablation study on learning rates → identified `0.1` as the sweet spot with the best validation loss and least epochs. (📊 see [`lr_ablation.png`](lr_ablation.png))  
5. Achieved **~98.16% validation accuracy** after 35 epochs.  
6. Applied **symmetric QAT (Quantization Aware Training)** to quantize weights from FP32 → INT8.  
7. At inference, quantized weights are loaded and evaluated.  
8. 📉 Compared **validation loss** & 📦 **storage size** between quantized and unquantized models:  
   - Validation loss drop:  
     - Quantized: **1.11%**  
     - Unquantized: **1.10%**  
   - Storage size:  
     - Quantized: **238 kB**  
     - Unquantized: **942 kB**  
   - Compression: ~**3.96× smaller**  
   - (📊 see [`quant_vs_unquant_subplots.png`](quant_vs_unquant_subplots.png))

---

# ⚠️ Architecture Limitations
💻 Trained on **CPU only** due to GPU incompatibility:  
- Laptop GPU: **NVIDIA GeForce GTX 1050 Max-Q, 3GB**  
- Missing support for **cuDNN 8.9 / CUDA 12.2** required by JAX.  

---

# 🔮 For Later
1. Dive deeper into how **`jax.grad`** works internally (automatic differentiation theory).  
2. Implement **AdamW** optimizer instead of plain SGD.  

---
