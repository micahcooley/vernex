"""
Fine-tune Vernex Nano on conversational data.
Preserves coding skills while adding chat abilities.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from model import VernexConfig, VernexForCausalLM
from tokenizers import Tokenizer
import os
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "model"
DATA_DIR = ROOT / "data"

class ConversationDataset(Dataset):
    def __init__(self, txt_file, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        with open(txt_file, 'r', encoding='utf-8') as f:
            self.text = f.read()
        
        # Split by <|im_end|> to keep samples intact
        raw_samples = self.text.split("<|im_end|>\n")
        
        self.samples = []
        for s in raw_samples:
            if not s.strip(): continue
            s += "<|im_end|>"
            enc = tokenizer.encode(s).ids
            if len(enc) > max_len: enc = enc[:max_len]
            if len(enc) < 5: continue  # Skip too-short samples
            self.samples.append(enc)
            
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        chunk = self.samples[idx]
        
        # Pad to fixed size
        needed = 512 - len(chunk)
        if needed > 0:
            chunk = chunk + [0] * needed
        elif needed < 0:
            chunk = chunk[:512]
            
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

def train_conversation():
    device = "cpu"  # Safe CPU training
    print(f"Fine-tuning on device: {device}")
    
    tokenizer = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))

    # Nano config (must match trained model)
    cfg = VernexConfig(
        vocab_size=tokenizer.get_vocab_size(),
        dim=768,
        n_layers=12,
        n_heads=12,
        n_kv_heads=4,
        hidden_dim=3072,
        max_seq_len=512
    )
    
    model = VernexForCausalLM(cfg).to(device)
    
    # Load existing Nano checkpoint
    import glob
    nano_weights = glob.glob(str(MODEL_DIR / "vernex_nano_*.pt"))
    if not nano_weights:
        print("ERROR: No Nano weights found!")
        return
    
    latest_nano = max(nano_weights, key=os.path.getctime)
    print(f"Loading Nano Model: {latest_nano}")
    model.load_state_dict(torch.load(latest_nano, map_location=device))
    
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # Very low learning rate to preserve existing knowledge
    optimizer = optim.AdamW(model.parameters(), lr=5e-6)
    
    # Load conversation corpus
    dataset = ConversationDataset(str(DATA_DIR / "conversation_corpus.txt"), tokenizer)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    model.train()
    print(f"Starting Conversational Fine-Tuning ({len(dataset)} samples)...")
    
    for epoch in range(3):  # 3 epochs should be enough
        total_loss = 0
        start_time = time.time()
        
        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(loader)
        elapsed = time.time() - start_time
        print(f"Epoch {epoch} Complete | Time: {elapsed:.1f}s | Avg Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        torch.save(model.state_dict(), str(MODEL_DIR / f"vernex_nano_chat_epoch_{epoch}.pt"))
    
    # Save final "chat" version
    torch.save(model.state_dict(), str(MODEL_DIR / "vernex_nano_chat.pt"))
    print(f"✅ Saved conversational model to: {MODEL_DIR / 'vernex_nano_chat.pt'}")

if __name__ == "__main__":
    train_conversation()
