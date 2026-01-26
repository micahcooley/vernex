---
description: Train Vernex model on NVIDIA GPU (RTX 4080/4090)
---

// turbo-all
# NVIDIA Training Workflow

Use this if you have access to an NVIDIA GPU. It is ~100x faster than CPU training.

## 1. Environment Setup
```bash
# Create venv
python3 -m venv .venv
source .venv/bin/activate

# Install NVIDIA-optimized PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install requirements
pip install -r requirements.txt
```

## 2. Sync Existing Progress
Ensure the `model/tokenizer.json` and any previous checkpoints are in the `model/` directory.

## 3. Run Training
The scripts automatically detect CUDA and use FP16 Mixed Precision on NVIDIA.

```bash
# Finish Tool Tuning in minutes
python3 src/train_tools.py

# Quantize for local use
python3 src/quantize.py
```

## 4. Why the 4080?
- **VRAM**: 16GB is more than enough for the 500M model.
- **Speed**: AMP (Mixed Precision) works perfectly on NVIDIA, offering a massive speed boost.
- **Stability**: No aperture violations or driver hangs common on older AMD RDNA1 cards.
