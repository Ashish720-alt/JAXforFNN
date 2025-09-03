# JAXforFNN
1. Using JAX framework for the training of a simple MLP architecture

2. Initialized weights using Xavier-He initialization.

3. Trained an 2 hidden layer MLP with SGD + early stop to identify handwritten digits from 0-9.

4. Trained on 200 epochs to get validation accuracy > 95%.

5. Used QAT (Quantization Aware Training) to quantize the weights and store them in weights.npz - used symmetric quantization.

6. At inference time, the quantized weights are loaded and MLP runs.

7. Did an ablation study wrt learning rates to see which learning rate value takes least epochs for training and achieves best accuracy.

8. Did an ablation study to measure the error on the validation set at inference time for quantized vs non-quantized weights.


# Architecture Limitations:
Although personal laptop has an NVIDIA GPU specifically GeForce GTX 1050 Max-Q, 3GB; it doesn't have required cuDNN 8.9 interface for CUDA 12.2, so couldn't use the GPU. So Trained model using JAX-CPU. 

# For later: 
1. Read on how jax.grad works INTERNALLY - how does automated differentiation work?

2. Implement AdamW instead of SGD
