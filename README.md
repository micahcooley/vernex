# Vernex

**Vernex** is a custom-built, efficient local AI coding assistant designed to run on consumer hardware.

## Architecture
- **Type**: Decoder-only Transformer
- **Features**: RoPE, RMSNorm, SwiGLU, Grouped Query Attention (GQA).
- **Parameters**: ~500M (customizable).

## Quick Start
1. **Install Requirements**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train Tokenizer**
   ```bash
   python src/tokenizer.py
   ```

3. **Train Model (Nano version)**
   ```bash
   python src/train.py
   ```

4. **Chat**
   ```bash
   python src/generate.py
   ```

## Directory Structure
- `src/model.py`: The neural network definition.
- `src/train.py`: Training loop.
- `src/tokenizer.py`: BPE tokenizer training.
- `src/generate.py`: Inference script.
