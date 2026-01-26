"""
Vernex Micro Training Script
30M parameter model - Beginner-style critic/reviewer
Asks questions, spots potential issues, provides feedback
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import VernexConfig, VernexForCausalLM
from tokenizers import Tokenizer
import os
import time
from pathlib import Path

# Resolve paths relative to project root
ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "model"
DATA_DIR = ROOT / "data"

class MultiCorpusDataset(Dataset):
    """Dataset that mixes multiple files with weighting/oversampling."""
    def __init__(self, corpora_configs, tokenizer, max_len=256):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.chunks = []
        
        for cfg in corpora_configs:
            path = cfg['path']
            weight = cfg.get('weight', 1)
            print(f"Loading {path} (weight: {weight})...")
            
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Oversample by repeating the text
            tokens = tokenizer.encode(text * weight).ids
            
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
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training Vernex Micro on device: {device}")
    
    # Load tokenizer
    tokenizer_path = MODEL_DIR / "tokenizer.json"
    if not tokenizer_path.exists():
        print("No tokenizer found!")
        return
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    # Vernex Micro Config (~14M params)
    cfg = VernexConfig(
        dim=384,
        n_layers=6,
        n_heads=6,
        n_kv_heads=2,
        hidden_dim=1536,
        vocab_size=tokenizer.get_vocab_size(),
        max_seq_len=256
    )
    
    model = VernexForCausalLM(cfg).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {param_count/1e6:.2f}M")
    
    # Data Mixing Configuration
    corpora = [
        # 10x oversample for chat to make it chat-focused
        {'path': DATA_DIR / "conversation_corpus.txt", 'weight': 10}, 
        # Weighted code focus
        {'path': DATA_DIR / "cpp_juce_skia_corpus.txt", 'weight': 1},
        # Synthetic critic data (if it exists)
        {'path': DATA_DIR / "critic_corpus.txt", 'weight': 15},
    ]
    
    # Filter to existing files
    corpora = [c for c in corpora if Path(c['path']).exists()]
    
    dataset = MultiCorpusDataset(corpora, tokenizer, max_len=cfg.max_seq_len)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Auto-Resume Logic
    latest_ckpt = MODEL_DIR / "vernex_micro_latest.pt"
    start_epoch = 0
    if latest_ckpt.exists():
        print(f"✅ Resuming from checkpoint: {latest_ckpt}")
        try:
            checkpoint = torch.load(str(latest_ckpt), map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                start_epoch = checkpoint.get('epoch', 0) + 1
            else:
                model.load_state_dict(checkpoint)
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
    
    optimizer = optim.AdamW(model.parameters(), lr=5e-4)
    model.train()
    
    print("Starting Vernex Micro Training...")
    num_epochs = 15  # More epochs for smaller model
    
    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0
        batch_count = 0
        start_time = time.time()
        
        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits, loss = model(x, targets=y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")
            
            if batch_idx % 100 == 0 and batch_idx > 0:
                # Save checkpoint - Atomic
                tmp_path = MODEL_DIR / "vernex_micro_latest.pt.tmp"
                final_path = MODEL_DIR / "vernex_micro_latest.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }, str(tmp_path))
                if tmp_path.exists():
                    try:
                        os.replace(str(tmp_path), str(final_path))
                        print(f"Checkpoint saved at batch {batch_idx}")
                    except Exception as e:
                        print(f"Warning: Could not rename checkpoint: {e}")
        
        # End of epoch
        elapsed = time.time() - start_time
        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
        print(f"✅ Epoch {epoch} complete | Avg Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")
        
        # Save epoch checkpoint - Atomic
        epoch_tmp = MODEL_DIR / f"vernex_micro_epoch_{epoch}.pt.tmp"
        epoch_final = MODEL_DIR / f"vernex_micro_epoch_{epoch}.pt"
        latest_tmp = MODEL_DIR / "vernex_micro_latest.pt.tmp"
        latest_final = MODEL_DIR / "vernex_micro_latest.pt"
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, str(epoch_tmp))
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, str(latest_tmp))

        if epoch_tmp.exists():
            try:
                os.replace(str(epoch_tmp), str(epoch_final))
            except Exception as e:
                print(f"Warning: Could not rename epoch checkpoint: {e}")
        
        if latest_tmp.exists():
            try:
                os.replace(str(latest_tmp), str(latest_final))
            except Exception as e:
                print(f"Warning: Could not rename latest checkpoint: {e}")
    
    print("🎉 Vernex Micro training complete!")

if __name__ == "__main__":
    train()
