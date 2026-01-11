import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from model import VernexConfig, VernexForCausalLM
from tokenizers import Tokenizer
import os
import time

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
    try:
        tokenizer = Tokenizer.from_file("c:/vernex/model/tokenizer.json")
    except:
        print("Tokenizer not found! Run src/tokenizer.py first.")
        return

    # 500M Config (user requirement)
    cfg = VernexConfig(
        dim=1024,       # 500M model
        n_layers=24,    # 500M model
        n_heads=16,     # 500M model
        n_kv_heads=8,   # 500M model
        hidden_dim=4096,  # 500M model
        vocab_size=tokenizer.get_vocab_size(),
        max_seq_len=512
    )
    
    model = VernexForCausalLM(cfg).to(device)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda")) # FP16 Scaler
    
    dataset = CodeDataset("c:/vernex/data/audio_corpus.txt", tokenizer, max_len=cfg.max_seq_len)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    model.train()
    
    print("Starting Training Loop (AMP Enabled)...")
    for epoch in range(10):
        total_loss = 0
        start_time = time.time()
        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            # Autocast for FP16 speed
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits, loss = model(x, y)
            
            # Scaled backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")
            
            if batch_idx % 50 == 0:
                torch.save(model.state_dict(), "c:/vernex/model/vernex_nano_latest.pt")
                print(f"Checkpoint saved at batch {batch_idx}")
        
        print(f"Epoch {epoch} Time: {time.time() - start_time:.2f}s | Avg Loss: {total_loss/len(loader):.4f}")
        
        # Save checkpoint
        torch.save(model.state_dict(), f"c:/vernex/model/vernex_nano_epoch_{epoch}.pt")

if __name__ == "__main__":
    train()
