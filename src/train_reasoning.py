"""
Vernex Adaptive Reasoning Fine-tuning
Chain-of-thought, step-by-step reasoning
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

class MultiCorpusDataset(Dataset):
    def __init__(self, corpora_configs, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.chunks = []
        
        for cfg in corpora_configs:
            path = cfg['path']
            weight = cfg.get('weight', 1)
            if not Path(path).exists():
                print(f"Skipping {path} - not found")
                continue
            print(f"Loading {path} (weight: {weight})...")
            
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            tokens = tokenizer.encode(text).ids
            for _ in range(weight):
                for i in range(0, len(tokens) - max_len, max_len // 2):
                    self.chunks.append(tokens[i:i+max_len])
        
        print(f"Total dataset chunks: {len(self.chunks)}")
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training Reasoning on: {device}")
    
    tokenizer = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    
    cfg = VernexConfig(
        dim=768, n_layers=12, n_heads=12, n_kv_heads=4,
        hidden_dim=3072, vocab_size=tokenizer.get_vocab_size(), max_seq_len=512
    )
    
    model = VernexForCausalLM(cfg).to(device)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    base_ckpt = MODEL_DIR / "vernex_nano_latest.pt"
    if base_ckpt.exists():
        print(f"Loading checkpoint: {base_ckpt}")
        model.load_state_dict(torch.load(str(base_ckpt), map_location=device))
    
    corpora = [
        {'path': DATA_DIR / "reasoning_corpus.txt", 'weight': 20},
        {'path': DATA_DIR / "cpp_juce_skia_corpus.txt", 'weight': 1},
    ]
    
    dataset = MultiCorpusDataset(corpora, tokenizer, max_len=cfg.max_seq_len)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    model.train()
    
    print("Starting Reasoning Fine-tuning...")
    for epoch in range(5):
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
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")
            
            if batch_idx % 100 == 0:
                tmp_path = MODEL_DIR / "vernex_reasoning_latest.pt.tmp"
                final_path = MODEL_DIR / "vernex_reasoning_latest.pt"
                torch.save(model.state_dict(), str(tmp_path))
                if tmp_path.exists():
                    os.replace(str(tmp_path), str(final_path))
                    print(f"Checkpoint saved at batch {batch_idx}")
        
        print(f"Epoch {epoch} Time: {time.time() - start_time:.2f}s | Avg Loss: {total_loss/len(loader):.4f}")
    
    torch.save(model.state_dict(), str(MODEL_DIR / "vernex_reasoning_final.pt"))
    torch.save(model.state_dict(), str(MODEL_DIR / "vernex_nano_latest.pt"))
    print("Reasoning training complete!")

if __name__ == "__main__":
    train()
