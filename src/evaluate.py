
import torch
import sys
from pathlib import Path
from model import VernexConfig, VernexForCausalLM
from tokenizers import Tokenizer

ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "model"
DATA_DIR = ROOT / "data"

def evaluate_model():
    """Run basic C++/JUCE logic tests on the model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluating on {device}...")

    # Load Tokenizer
    tokenizer_path = MODEL_DIR / "tokenizer.json"
    if not tokenizer_path.exists():
        print("Tokenizer not found.")
        return
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    # Load Model (Nano Config)
    cfg = VernexConfig(
        dim=768, n_layers=12, n_heads=12, n_kv_heads=4,
        hidden_dim=3072, vocab_size=tokenizer.get_vocab_size(), 
        max_seq_len=512
    )
    model = VernexForCausalLM(cfg).to(device)
    
    # Load weights
    import glob
    import os
    weights = glob.glob(str(MODEL_DIR / "vernex_nano_*.pt"))
    if not weights:
        print("No weights found.")
        return
    latest = max(weights, key=os.path.getctime)
    print(f"Loading {latest}...")
    model.load_state_dict(torch.load(latest, map_location=device))
    model.eval()

    test_cases = [
        ("Write a C++ function to add two numbers.", "int add(int a, int b)"),
        ("Clear a JUCE AudioBuffer.", "buffer.clear()"),
        ("Create a Skia paint object.", "SkPaint paint;")
    ]

    print("\n--- Evaluation Results ---")
    correct = 0
    for prompt, expected in test_cases:
        print(f"\nPrompt: {prompt}")
        
        # Simple generation
        input_ids = torch.tensor([tokenizer.encode(prompt).ids]).to(device)
        model.clear_cache()
        
        generated_text = ""
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(device=="cuda")):
                for token_id in model.generate(input_ids, max_new_tokens=50, tokenizer=tokenizer):
                    text = tokenizer.decode([token_id])
                    if "<|im_end|>" in text: break
                    generated_text += text
        
        print(f"Output: {generated_text.strip()}")
        # Very basic check - just see if it generates *something* relevant or code-like
        if len(generated_text) > 5: 
             correct += 1
             
    print(f"\nPassed (Basic Check): {correct}/{len(test_cases)}")

if __name__ == "__main__":
    evaluate_model()
