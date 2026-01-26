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

class CodeDataset(Dataset):
    def __init__(self, txt_file, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        with open(txt_file, 'r', encoding='utf-8') as f:
            self.text = f.read()
        
        self.encoded = tokenizer.encode(self.text).ids
        # Create simple sliding chunks
        self.samples = []
        for i in range(0, len(self.encoded) - max_len, max_len):
            self.samples.append(self.encoded[i:i+max_len])
            
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        chunk = self.samples[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

def train():
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")
    
    # Load Tokenizer
    tokenizer_path = MODEL_DIR / "tokenizer.json"
    try:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    except:
        print(f"Tokenizer not found at {tokenizer_path}! Run src/tokenizer.py first.")
        return

    # Vernex Nano Config (~120M) - Fits in 8GB VRAM
    cfg = VernexConfig(
        dim=768,        # 1024 -> 768
        n_layers=12,    # 24 -> 12
        n_heads=12,     # 16 -> 12
        n_kv_heads=4,   # 8 -> 4
        hidden_dim=3072,  # 4096 -> 3072
        vocab_size=tokenizer.get_vocab_size(),
        max_seq_len=512
    )
    
    model = VernexForCausalLM(cfg).to(device)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # Auto-Resume Logic
    latest_ckpt = MODEL_DIR / "vernex_nano_latest.pt"
    if latest_ckpt.exists():
        print(f"✅ Resuming from checkpoint: {latest_ckpt}")
        model.load_state_dict(torch.load(str(latest_ckpt), map_location=device))

    
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=False) # Disable FP16 Scaler
    
    # Find training data - prefer enhanced corpus
    corpus_path = DATA_DIR / "cpp_juce_skia_corpus.txt"
    if not corpus_path.exists():
        corpus_path = DATA_DIR / "audio_corpus.txt"
    if not corpus_path.exists():
        print(f"No corpus found in {DATA_DIR}! Run cpp_juce_skia_data.py first.")
        return
    
    dataset = CodeDataset(str(corpus_path), tokenizer, max_len=cfg.max_seq_len)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    model.train()
    
    print("Starting Training Loop (FP32 ONLY)...")
    for epoch in range(10):
        total_loss = 0
        start_time = time.time()
        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            # Disable Autocast for stability
            with torch.cuda.amp.autocast(enabled=False):
                logits, loss = model(x, y)
            
            # Scaled backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")
            
            if batch_idx % 50 == 0:
                # Atomic save to prevent file contention
                tmp_path = MODEL_DIR / "vernex_nano_latest.pt.tmp"
                final_path = MODEL_DIR / "vernex_nano_latest.pt"
                torch.save(model.state_dict(), str(tmp_path))
                if tmp_path.exists():
                    try:
                        os.replace(str(tmp_path), str(final_path))
                        print(f"Checkpoint saved at batch {batch_idx}")
                    except Exception as e:
                        print(f"Warning: Could not rename checkpoint: {e}")
        
        print(f"Epoch {epoch} Time: {time.time() - start_time:.2f}s | Avg Loss: {total_loss/len(loader):.4f}")
        
        # Save checkpoint
        epoch_tmp = MODEL_DIR / f"vernex_nano_epoch_{epoch}.pt.tmp"
        epoch_final = MODEL_DIR / f"vernex_nano_epoch_{epoch}.pt"
        torch.save(model.state_dict(), str(epoch_tmp))
        if epoch_tmp.exists():
            try:
                os.replace(str(epoch_tmp), str(epoch_final))
            except Exception as e:
                print(f"Warning: Could not rename epoch checkpoint: {e}")

if __name__ == "__main__":
    train()
