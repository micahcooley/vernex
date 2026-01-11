"""
Vernex Model with KV Cache for fast incremental decoding.
Target: 20+ TPS on CPU with 8GB RAM.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class VernexConfig:
    vocab_size: int = 4096
    dim: int = 256
    n_layers: int = 8       # Must match train.py
    n_heads: int = 8
    n_kv_heads: int = 4
    hidden_dim: int = 512
    max_seq_len: int = 512
    norm_eps: float = 1e-6
    rope_theta: float = 10000.0

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

def precompute_rope_freqs(dim: int, max_len: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_len).float()
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)

def apply_rope(xq, xk, freqs_cis, start_pos: int = 0):
    seq_len = xq.shape[1]
    freqs = freqs_cis[start_pos : start_pos + seq_len]
    freqs = freqs.view(1, seq_len, 1, -1)
    
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    xq_out = torch.view_as_real(xq_ * freqs).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class VernexAttention(nn.Module):
    def __init__(self, cfg: VernexConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.repeats = self.n_heads // self.n_kv_heads
        
        self.wq = nn.Linear(cfg.dim, cfg.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.head_dim, cfg.dim, bias=False)
        
        # KV Cache
        self.cache_k = None
        self.cache_v = None

    def forward(self, x, freqs_cis, start_pos: int = 0, use_cache: bool = False):
        B, L, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        
        xq = xq.view(B, L, self.n_heads, self.head_dim)
        xk = xk.view(B, L, self.n_kv_heads, self.head_dim)
        xv = xv.view(B, L, self.n_kv_heads, self.head_dim)
        
        xq, xk = apply_rope(xq, xk, freqs_cis, start_pos)
        
        if use_cache:
            if self.cache_k is None:
                self.cache_k = xk
                self.cache_v = xv
            else:
                self.cache_k = torch.cat([self.cache_k, xk], dim=1)
                self.cache_v = torch.cat([self.cache_v, xv], dim=1)
            xk, xv = self.cache_k, self.cache_v
        
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        
        if self.repeats > 1:
            xk = xk.repeat_interleave(self.repeats, dim=1)
            xv = xv.repeat_interleave(self.repeats, dim=1)
        
        scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Causal mask
        if L > 1:
            mask = torch.triu(torch.full((L, xk.shape[2]), float("-inf")), diagonal=1)
            scores = scores + mask.to(x.device)
        
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, xv)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.wo(out)
    
    def clear_cache(self):
        self.cache_k = None
        self.cache_v = None

class VernexFeedForward(nn.Module):
    def __init__(self, cfg: VernexConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.dim, cfg.hidden_dim, bias=False)
        self.w2 = nn.Linear(cfg.hidden_dim, cfg.dim, bias=False)
        self.w3 = nn.Linear(cfg.dim, cfg.hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class VernexBlock(nn.Module):
    def __init__(self, cfg: VernexConfig):
        super().__init__()
        self.attn = VernexAttention(cfg)
        self.ffn = VernexFeedForward(cfg)
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)

    def forward(self, x, freqs_cis, start_pos=0, use_cache=False):
        h = x + self.attn(self.attn_norm(x), freqs_cis, start_pos, use_cache)
        return h + self.ffn(self.ffn_norm(h))

class VernexModel(nn.Module):
    def __init__(self, cfg: VernexConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList([VernexBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.freqs_cis = precompute_rope_freqs(cfg.dim // cfg.n_heads, cfg.max_seq_len * 2)

    def forward(self, tokens, start_pos=0, use_cache=False):
        h = self.embed(tokens)
        freqs = self.freqs_cis.to(h.device)
        for layer in self.layers:
            h = layer(h, freqs, start_pos, use_cache)
        return self.norm(h)
    
    def clear_cache(self):
        for layer in self.layers:
            layer.attn.clear_cache()

class VernexForCausalLM(nn.Module):
    def __init__(self, cfg: VernexConfig):
        super().__init__()
        self.model = VernexModel(cfg)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.model.embed.weight  # Weight tying

    def forward(self, tokens, targets=None, start_pos=0, use_cache=False):
        h = self.model(tokens, start_pos, use_cache)
        logits = self.lm_head(h)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    
    def clear_cache(self):
        self.model.clear_cache()

    @torch.inference_mode()
    def generate(self, input_ids, max_new_tokens=50, tokenizer=None, temperature=0.7, top_p=0.9):
        """Fast generation with KV cache and sampling. Yields tokens."""
        self.clear_cache()
        
        # Prefill
        logits, _ = self(input_ids, start_pos=0, use_cache=True)
        next_token = self.sample(logits[:, -1, :], temperature, top_p)
        
        token_id = next_token.item()
        yield token_id
        
        pos = input_ids.shape[1]
        
        for _ in range(max_new_tokens - 1):
            logits, _ = self(next_token, start_pos=pos, use_cache=True)
            next_token = self.sample(logits[:, -1, :], temperature, top_p)
            token_id = next_token.item()
            yield token_id
            pos += 1
            
            if tokenizer:
                text = tokenizer.decode([token_id])
                if "<|im_end|>" in text:
                    break

    def sample(self, logits, temperature, top_p):
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        return next_token
