# Vernex AI

Local AI assistant for C++/JUCE/Skia audio software development. GPU-accelerated, web search, dual-agent critique system.

## Features

- **500M Parameter Model** - Understands context, minimal hallucination
- **Dual-Agent System** - Agent-2 critiques Agent-1 in real-time
- **Web Search** - DuckDuckGo search for verification
- **OpenAI-Compatible API** - Works with VS Code, Cursor, etc.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt

# For AMD GPU (ROCm):
pip install torch --index-url https://download.pytorch.org/whl/rocm5.7

# For NVIDIA GPU:
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 2. Generate Training Data
```bash
# Basic training data
python src/data_gen.py

# C++/JUCE/Skia focused (recommended)
python src/cpp_juce_skia_data.py
```

### 3. Train Tokenizer
```bash
python src/tokenizer.py
```

### 4. Train Model
```bash
python src/train.py
```
GPU is auto-detected. Training uses FP16 mixed precision for speed.

### 5. Run & Sync

**On Main PC (Training & Full Usage):**
```bash
python3 src/dual_agent.py        # Run with local weights
```

**Syncing to Laptop (Git LFS):**
Since model files are large (>400MB), we use **Git LFS**. To pull models onto your laptop:
1. Install Git LFS: `git lfs install`
2. Pull latest: `git pull origin main`

**On Laptop (CPU Optimized):**
```bash
# Run the 120M Super-Nano model specifically
python3 src/dual_agent.py

# Once the 500M quantized model is ready:
python3 src/dual_agent.py --mobile
```

### 6. Chat UI
Open `chat.html` in browser, or connect to `http://localhost:8000/v1`

## Model Architecture

| Config | Value |
|--------|-------|
| Parameters | ~500M |
| Hidden Dim | 1024 |
| Layers | 24 |
| Heads | 16 |
| Context | 512 tokens |

## Training Data Mix

| Category | Percent | Contents |
|----------|---------|----------|
| Code | 70% | JUCE, Skia, C++ DSP |
| Debug | 12% | Memory leaks, threading, audio glitches |
| FIM | 10% | Fill-in-the-middle completion |
| Critic | 5% | Agent-2 critique examples |
| Chat | 3% | Basic greetings |

## Dual-Agent System

```
User Query → Agent-1 → Response
                ↓
           Agent-2 (Critic)
                ↓
           "Wrong because..." or "Verified."
```

Agent-2 catches:
- Wrong API usage
- Outdated version info
- Unsafe threading practices
- Memory leaks

## Files

| File | Purpose |
|------|---------|
| `train.py` | GPU training with AMP |
| `cpp_juce_skia_data.py` | Real code training data |
| `dual_agent.py` | Dual-agent orchestrator |
| `server.py` | OpenAI-compatible API |
| `model.py` | 500M architecture |

## License

MIT
