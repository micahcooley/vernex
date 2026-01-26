"""
Vernex Pro Training Script (380M Parameters)
Focus: Role-based masking (no hallucinating user), high-diveristy, loops removal.
"""
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from model import VernexConfig, VernexForCausalLM
from tokenizers import Tokenizer
import os
import time
from pathlib import Path
import collections

# Resolve paths
ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "model"
DATA_DIR = ROOT / "data"

# === TRAINING SETTINGS ===
LABEL_SMOOTHING = 0.05  # Slight smoothing
REPETITION_PENALTY_NDIM = 3  # Look back at 3-grams for loops
REPETITION_PENALTY_WEIGHT = 1.0  # MUCH higher weight
DROPOUT_RATE = 0.1

class MultiCorpusDataset(Dataset):
    def __init__(self, corpora_configs, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.chunks = []
        self.masks = []
        
        # Identify special sequences
        # <|im_start|>assistant = [2, 11] usually, but let's be safe
        asst_seq = tokenizer.encode("<|im_start|>assistant").ids
        user_seq = tokenizer.encode("<|im_start|>user").ids
        eot_seq = tokenizer.encode("<|im_end|>").ids
        
        for cfg in corpora_configs:
            path = cfg['path']
            weight = cfg.get('weight', 1)
            if not Path(path).exists():
                continue
            print(f"Loading {path} (weight: {weight})...")
            
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            tokens = tokenizer.encode(text).ids
            
            # Generate role mask
            mask = [0] * len(tokens)
            is_assistant = False
            i = 0
            while i < len(tokens):
                # Check for assistant start
                if tokens[i:i+len(asst_seq)] == asst_seq:
                    is_assistant = True
                    i += len(asst_seq)
                    continue
                # Check for user start (or EOT)
                if tokens[i:i+len(user_seq)] == user_seq or tokens[i:i+len(eot_seq)] == eot_seq:
                    is_assistant = False
                    # Keep EOT masked as 0 so we don't overtrain on stopping either (or set to 1 to train stopping)
                    i += 1
                    continue
                
                if is_assistant:
                    mask[i] = 1
                i += 1
            
            # Oversample chunks
            # Shift by 1 because y is chunks[1:] and mask must align with y
            for _ in range(weight):
                for i in range(0, len(tokens) - max_len, max_len // 2):
                    self.chunks.append(tokens[i:i+max_len])
                    self.masks.append(mask[i:i+max_len])
        
        print(f"Total dataset chunks: {len(self.chunks)}")
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        mask = self.masks[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        m = torch.tensor(mask[1:], dtype=torch.float32) # Align mask with y
        return x, y, m

def compute_ngram_penalty(logits, n=3):
    """Heavy penalty for repeating n-gram patterns."""
    batch_size, seq_len, vocab_size = logits.shape
    if seq_len < n * 2:
        return torch.tensor(0.0, device=logits.device)
    
    # Get predicted tokens
    preds = logits.argmax(dim=-1) # [B, L]
    
    penalty = 0
    # Check for repeating n-grams: pattern [i:i+n] == [i+n : i+2n]
    for i in range(seq_len - 2*n):
        match = (preds[:, i:i+n] == preds[:, i+n:i+2*n]).all(dim=-1).float()
        penalty += match.mean()
    
    return penalty / (seq_len - 2*n)

def masked_loss(logits, targets, mask, smoothing=0.05):
    """Calculate loss only where mask=1 (assistant response)."""
    vocab_size = logits.size(-1)
    logits = logits.view(-1, vocab_size)
    targets = targets.view(-1)
    mask = mask.view(-1)
    
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    
    # Softmax and log
    log_probs = F.log_softmax(logits, dim=-1)
    
    # NLL Loss
    nll = -log_probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
    
    # Smoothed component
    smooth = -log_probs.mean(dim=-1)
    
    # Combine and Mask
    loss = (1 - smoothing) * nll + smoothing * smooth
    masked_loss = (loss * mask).sum() / mask.sum()
    
    return masked_loss

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training Vernex Pro (380M) with ROLE MASKING")
    
    tokenizer_path = MODEL_DIR / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    cfg = VernexConfig(
        dim=1024, n_layers=24, n_heads=16, n_kv_heads=8,
        hidden_dim=4096, vocab_size=tokenizer.get_vocab_size(), max_seq_len=512
    )
    
    print(f"DEBUG: Config dim={cfg.dim}, layers={cfg.n_layers}, hidden={cfg.hidden_dim}")
    model = VernexForCausalLM(cfg).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"CRITICAL: Model Parameters: {param_count/1e6:.2f}M")
    
    if param_count < 300_000_000:
        raise RuntimeError(f"MODEL SIZE ERROR: Expected ~380M, got {param_count/1e6:.2f}M. Check VernexConfig defaults!")

    latest_ckpt = MODEL_DIR / "vernex_pro_latest.pt"
    if latest_ckpt.exists():
        print(f"✅ Loading {latest_ckpt} ({os.path.getsize(latest_ckpt)/1e6:.1f} MB)")
        try:
            model.load_state_dict(torch.load(str(latest_ckpt), map_location=device))
            print("✓ Checkpoint loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            print("Tip: If size mismatch, check if you are trying to load Nano weights into Pro model.")
            raise e
    
    # Data Mix - High Priority on Knowledge, but strictly masked
    corpora = [
        {'path': DATA_DIR / "reasoning_corpus.txt", 'weight': 20},
        {'path': DATA_DIR / "cpp_juce_skia_corpus.txt", 'weight': 15},
        {'path': DATA_DIR / "search_behavior_corpus.txt", 'weight': 10},
        {'path': DATA_DIR / "terminal_logic_corpus.txt", 'weight': 10},
        {'path': DATA_DIR / "honesty_corpus.txt", 'weight': 5},
        {'path': DATA_DIR / "conversation_corpus.txt", 'weight': 5}, # Lowered (greetings often confuse)
        {'path': DATA_DIR / "combined_corpus.txt", 'weight': 2},
    ]
    
    dataset = MultiCorpusDataset(corpora, tokenizer, max_len=cfg.max_seq_len)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.1) # Lower LR for stability
    
    GRAD_ACCUM = 32
    print("Training Started. Only Assistant tokens contribute to loss.")
    
    model.train()
    for epoch in range(10):
        optimizer.zero_grad()
        for batch_idx, (x, y, m) in enumerate(loader):
            x, y, m = x.to(device), y.to(device), m.to(device)
            
            logits, _ = model(x)
            
            # 1. Masked Context Loss
            loss_val = masked_loss(logits, y, m, LABEL_SMOOTHING)
            
            # 2. Repetition Penalty (Applied to logits to influence distribution)
            rep_pen = compute_ngram_penalty(logits, REPETITION_PENALTY_NDIM)
            
            total_loss = loss_val + (REPETITION_PENALTY_WEIGHT * rep_pen)
            
            if torch.isnan(total_loss):
                continue
                
            (total_loss / GRAD_ACCUM).backward()
            
            if (batch_idx + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                optimizer.zero_grad()
            
            if batch_idx % 10 == 0:
                m_perc = (m.sum() / m.numel()) * 100
                print(f"B:{batch_idx} | Loss:{loss_val.item():.3f} | Rep:{rep_pen.item():.4f} | Mask:{m_perc:.1f}%")
            
            # ATOMIC SAVE every 500 batches
            if batch_idx > 0 and batch_idx % 500 == 0:
                try:
                    final_path = MODEL_DIR / "vernex_pro_latest.pt"
                    tmp_path = str(final_path) + ".tmp"
                    torch.save(model.state_dict(), tmp_path)
                    if os.path.exists(tmp_path):
                        os.replace(tmp_path, str(final_path))
                        print(f"✅ Atomic Checkpoint saved at batch {batch_idx}")
                except Exception as e:
                    print(f"⚠️ Warning: Checkpoint save failed at batch {batch_idx}: {e}")
                    print("Continuing training... will retry next 500 batches.")

if __name__ == "__main__":
    train()
