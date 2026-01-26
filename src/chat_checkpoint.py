import torch
from pathlib import Path
from model import VernexConfig, VernexForCausalLM
from tokenizers import Tokenizer
import sys
import colorama
from colorama import Fore, Style

colorama.init()

ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "model"

def chat():
    # Force GPU for inference to avoid fighting CPU training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{Fore.CYAN}Starting Inference on: {device}{Style.RESET_ALL}")

    # Load Tokenizer
    tokenizer = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))

    # Load Nano Config
    cfg = VernexConfig(
        dim=768, n_layers=12, n_heads=12, n_kv_heads=4,
        hidden_dim=3072, vocab_size=tokenizer.get_vocab_size(), 
        max_seq_len=512
    )
    
    model = VernexForCausalLM(cfg).to(device)
    
    # Check for latest weights
    import glob
    import os
    weights = glob.glob(str(MODEL_DIR / "vernex_nano_*.pt"))
    if not weights:
        print(f"{Fore.RED}No checkpoints found yet!{Style.RESET_ALL}")
        return
    
    latest = max(weights, key=os.path.getctime)
    print(f"{Fore.GREEN}Loaded checkpoint: {os.path.basename(latest)}{Style.RESET_ALL}")
    model.load_state_dict(torch.load(latest, map_location=device))
    model.eval()

    print(f"\n{Fore.YELLOW}Vernex (120M) is ready. (Type 'exit' to quit){Style.RESET_ALL}")
    
    while True:
        prompt = input(f"\n{Fore.BLUE}You: {Style.RESET_ALL}")
        if prompt.lower() in ["exit", "quit"]:
            break
            
        input_ids = torch.tensor([tokenizer.encode(prompt).ids]).to(device)
        model.clear_cache()
        
        print(f"{Fore.MAGENTA}Vernex: {Style.RESET_ALL}", end="", flush=True)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(device=="cuda")):
                for token_id in model.generate(input_ids, max_new_tokens=100, tokenizer=tokenizer):
                    text = tokenizer.decode([token_id])
                    if "<|im_end|>" in text:
                        break
                    print(text, end="", flush=True)
        print()

if __name__ == "__main__":
    chat()
