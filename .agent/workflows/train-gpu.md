---
description: Train Vernex model on AMD GPU
---

# Training Vernex (Stable CPU Pipeline)

Training: **CPU** (GPU training crashes on RX 5700 XT/ROCm 5.7).
Inference/Chat: **GPU** (Works perfectly!).

## Prerequisites
```bash
cd /home/micah/Desktop/vernex
source .venv/bin/activate
```

## Generate Training Data
// turbo
```bash
python3 src/cpp_juce_skia_data.py
```

## Train Tokenizer
// turbo
```bash
python3 src/tokenizer.py
```

## Train Model (CPU)
// turbo
```bash
# Guaranteed to work, takes ~8 hours.
HIP_VISIBLE_DEVICES=-1 python3 src/train.py
```

## Run Dual-Agent (GPU Enabled)
// turbo
```bash
# Agent runs inference on GPU (fast!)
HSA_OVERRIDE_GFX_VERSION=10.3.0 python3 src/dual_agent.py
```
