"""
Fast generation with KV cache for 20+ TPS on CPU.
"""
import torch
import torch.nn.functional as F
from model import VernexConfig, VernexForCausalLM
from tokenizers import Tokenizer
import os, glob, time
from pathlib import Path

# Resolve paths
ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "model"

@torch.inference_mode()
def generate_fast(prompt, max_tokens=100):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load tokenizer and model
    tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    cfg = VernexConfig(dim=1024, n_layers=24, n_heads=16, n_kv_heads=8, vocab_size=tok.get_vocab_size(), max_seq_len=512)
    model = VernexForCausalLM(cfg).to(device)
    
    # Try to load weights
    paths = glob.glob(str(MODEL_DIR / "vernex_nano_*.pt"))
    if paths:
        try:
            model.load_state_dict(torch.load(max(paths, key=os.path.getctime), map_location=device))
            print(f"Loaded weights on {device}")
        except: 
            print("Using random weights")
    
    model.eval()
    
    # Encode prompt
    ids = tok.encode(prompt).ids
    input_ids = torch.tensor([ids], device=device)
    
    # Generate with KV cache
    generated = []
    start = time.time()
    
    model.clear_cache()
    
    for i in range(max_tokens):
        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            logits, _ = model(input_ids, start_pos=len(ids) + i - 1 if i > 0 else 0, use_cache=True)
            next_logits = logits[0, -1, :]
            
            # Greedy decode (fastest)
            next_token = torch.argmax(next_logits).item()
            
            # For next iteration, only input the new token
            input_ids = torch.tensor([[next_token]], device=device)
            generated.append(next_token)
            
            # Stop if EOS
            text = tok.decode([next_token])
            if "<|im_end|>" in text:
                break
    
    elapsed = time.time() - start
    tps = len(generated) / elapsed if elapsed > 0 else 0
    
    output = tok.decode(generated)
    return output, tps

if __name__ == "__main__":
    import sys
    prompt = sys.argv[1] if len(sys.argv) > 1 else "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
    
    print("Generating...")
    output, tps = generate_fast(prompt)
    print(f"\nOutput: {output}")
    print(f"\nSpeed: {tps:.1f} tokens/second")
