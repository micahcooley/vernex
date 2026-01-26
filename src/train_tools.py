import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from model import VernexConfig, VernexForCausalLM
from tokenizers import Tokenizer
import os
import time
from pathlib import Path

# Resolve paths relative to project root
ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "model"
DATA_DIR = ROOT / "data"

class ToolDataset(Dataset):
    def __init__(self, txt_file, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        with open(txt_file, 'r', encoding='utf-8') as f:
            self.text = f.read()
        
        # Split by <|im_end|> to keep samples intact
        raw_samples = self.text.split("<|im_end|>\n")
        
        self.samples = []
        for s in raw_samples:
            if not s.strip(): continue
            s += "<|im_end|>" # add back the token
            enc = tokenizer.encode(s).ids
            # Truncate if too long (unlikely for these samples)
            if len(enc) > max_len: enc = enc[:max_len]
            self.samples.append(enc)
            
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        chunk = self.samples[idx]
        
        # Consistent padding to max_len
        # We need a fixed size for the stack to work (unless we use a custom collate)
        # Pad with 0 (assuming 0 is safe/ignore index, or just not impactful)
        needed = 512 - len(chunk)
        if needed > 0:
            chunk = chunk + [0] * needed
        elif needed < 0:
            chunk = chunk[:512]
            
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

def train_tools():
    device = "cuda" if torch.cuda.is_available() else "cpu" # Default to CUDA if possible (for 500M fine-tune)
    # Wait.. we are running on CPU for the big model, stick to CPU to be safe? 
    # Actually, fine-tuning is faster, but user has GPU issues. Stick to CPU or follow 500M config.
    # The queue script sets HIP_VISIBLE_DEVICES=-1, so this will naturally use CPU.
    
    print(f"Fine-Tuning on device: {device}")
    
    # Load Tokenizer
    tokenizer = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))

    # Config must match 500M Base
    cfg = VernexConfig(
        dim=1024, n_layers=24, n_heads=16, n_kv_heads=8,
        hidden_dim=4096, vocab_size=tokenizer.get_vocab_size(), 
        max_seq_len=512
    )
    
    model = VernexForCausalLM(cfg).to(device)
    
    # Load the FINAL 500M model
    # We look for the latest checkpoint from the base training
    import glob
    base_weights = glob.glob(str(MODEL_DIR / "vernex_base_*.pt"))
    if not base_weights:
        print("ERROR: No Base (500M) weights found! Skipping fine-tune.")
        return

    latest_base = max(base_weights, key=os.path.getctime)
    print(f"Loading Base Model: {latest_base}")
    model.load_state_dict(torch.load(latest_base, map_location=device))
    
    # Compile?
    if hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
        except: pass

    # Low Learning Rate for Fine-Tuning
    optimizer = optim.AdamW(model.parameters(), lr=1e-5) 
    
    # Load Tool Corpus
    dataset = ToolDataset(str(DATA_DIR / "tool_corpus.txt"), tokenizer)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    model.train()
    print("Starting Tool-Use Fine-Tuning...")
    
    # Train for just 1-2 epochs to check-in the behavior
    # Train for just 1 epoch to check-in the behavior (Optimized for CPU)
    for epoch in range(1):
        total_loss = 0
        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, loss = model(x, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")
        
        # Save specially marked checkpoint
        torch.save(model.state_dict(), str(MODEL_DIR / f"vernex_tooltuned_epoch_{epoch}.pt"))
        print(f"Saved Tool-Tuned Checkpoint for Epoch {epoch}")

if __name__ == "__main__":
    train_tools()
