# Vernex AI

A specialized AI assistant for audio software development. Runs locally, searches the web, and integrates with your editor.

## Features

- **500M Parameter Model** - Large enough to understand context and avoid hallucinations
- **Bundled Web Search** - DuckDuckGo search with no external dependencies
- **OpenAI-Compatible API** - Works with VS Code (Continue), Cursor, and more
- **Chat UI** - Simple HTML interface at `chat.html`

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Training Data
```bash
python src/data_gen.py
```

### 3. Train the Tokenizer
```bash
python src/tokenizer.py
```

### 4. Train the Model
```bash
python src/train.py
```

**GPU Training (AMD RX 5700 XT):**
```bash
# Install PyTorch with ROCm support first:
pip install torch --index-url https://download.pytorch.org/whl/rocm5.7

# Then run training - it will auto-detect GPU
python src/train.py
```

### 5. Start the Server
```bash
python src/server.py
```

### 6. Chat
Open `chat.html` in your browser, or connect your editor to `http://localhost:8000/v1`.

## Model Architecture

- **Parameters**: ~500M
- **Hidden Dim**: 1024
- **Layers**: 24
- **Heads**: 16
- **Context Length**: 512 tokens

## Training Data

The model is trained on:
- 40% Conversational (greetings, simple Q&A)
- 25% Audio debugging scenarios
- 20% Code generation (C++/JUCE)
- 15% Fill-in-the-Middle (FIM)

## License

MIT
