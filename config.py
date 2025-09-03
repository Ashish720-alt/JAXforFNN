DEBUG_MODE = False

LR_ABLATION_MODE = True

# --- Training ---
LR = 1e-3 #Learning Rate
MAX_EPOCHS = 1000 #Upper bound on epochs, early stopping implemented!
BATCH_SIZE = 128
METRIC = "val_loss" #"val_loss" or "val_acc"
LOSS_PATIENCE = 5  # require at least this much improvement to reset patience
LOSS_MIN_DELTA = 5*1e-4


# --- Quantization Aware Training toggles ---
QAT_ENABLE = True   # turn on/off QAT
W_BITS = 8
A_BITS = 8