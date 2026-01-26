"""
Vernex Dual-Agent System - DEFAULT MODE
========================================
Agent-1 (Main): Full 500M model - generates responses, codes, explains
Agent-2 (Critic): Lighter config - watches Agent-1, catches errors, verifies

Usage:
    python dual_agent.py          # Interactive (default)
    python dual_agent.py --test   # Run test interactions
    python dual_agent.py --single # Single agent mode (no critic)
"""

import torch
import sys
# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

import torch
import re
from pathlib import Path
from typing import Generator, Tuple, Optional

print("[DEBUG] Imports complete. Starting setup...")

# Setup paths
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from model import VernexConfig, VernexForCausalLM
from tokenizers import Tokenizer
from search import search_and_fetch

MODEL_DIR = ROOT / "model"

# === SYSTEM PROMPTS ===

AGENT1_SYSTEM = """You are Vernex Agent-1, a senior C++/JUCE/Skia engineer.
- Direct answers, no fluff
- Show code, explain WHY
- Admit uncertainty: use [SEARCH: query] when unsure
- Respectful roasting of bad practices
"""

AGENT2_SYSTEM = """You are Vernex Agent-2, the Total Skeptic & Design Critic. 
Your job: Question EVERYTHING Agent-1 says, especially Performance and UX.
- Is it efficient? "Wait, but won't that cause performance issues?"
- Is it the ONLY way? "Is there a more modern way to do this in JUCE 8?"
- **UX/Design?** "This UI looks ugly/dated. Users will expect a smoother transition here." or "That layout is confusing."
- Is it safe? "What happens if this object is null/thread-contended?"
- If it's perfect, still ask: "Could this be even simpler?"
Be brief, be sharp, be annoying if you have to. NEVER just say 'Verified' without double-checking the code AND the user experience.
"""

class DualAgentSystem:
    def __init__(self, device: str = None, enable_critic: bool = True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.enable_critic = enable_critic
        
        if "--mobile" in sys.argv:
            print("[INIT] Mobile Mode: CPU only, utilizing quantized model.")
            self.device = "cpu"

        print(f"[INIT] Initializing models on {self.device}...")
        # Load tokenizer
        tokenizer_path = MODEL_DIR / "tokenizer.json"
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
        print(f"[INIT] Loading tokenizer from {tokenizer_path}...")
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        
        vocab_size = self.tokenizer.get_vocab_size()
        
        if "--mobile" in sys.argv:
            print("[INIT] Mobile Mode: CPU only, utilizing quantized model.")
            self.device = "cpu"

        # Agent-1: Setup Placeholder (will be loaded or replaced)
        # We need the config only if we are creating a fresh model from state_dict
        # If loading a full quantized object, this is overwritten.
        cfg1 = VernexConfig(
            dim=768, n_layers=12, n_heads=12, n_kv_heads=4,
            hidden_dim=3072, vocab_size=vocab_size, max_seq_len=512
        )
        self.model1 = VernexForCausalLM(cfg1).to(self.device)
        
        # Agent-2: Lighter config (~100M) - still capable but faster
        if enable_critic:
            cfg2 = VernexConfig(
                dim=512, n_layers=12, n_heads=8, n_kv_heads=4,
                hidden_dim=2048, vocab_size=vocab_size, max_seq_len=256
            )
            self.model2 = VernexForCausalLM(cfg2).to(self.device)
            print(f"[INIT] Agent-2 (Critic): {sum(p.numel() for p in self.model2.parameters())/1e6:.1f}M params")
        else:
            self.model2 = None
        
        # Load weights for both models
        self._load_weights()
        
        # Print actual size (handling quantized vs float)
        def get_size(m):
            return sum(p.numel() for p in m.parameters())/1e6 if hasattr(m, "parameters") else "Quantized"

        print(f"[INIT] Agent-1 (Main): {get_size(self.model1)} params")
        print(f"[INIT] Ready on {self.device}. Critic: {'ON' if enable_critic else 'OFF'}")
    
    def _load_weights(self):
        """Load weights - supports State Dict (Standard) and Full Object (Quantized)."""
        import glob
        import os
        
        # Special handling for Mobile Mode
        if "--mobile" in sys.argv:
            mobile_path = MODEL_DIR / "vernex_mobile.pt"
            if mobile_path.exists():
                print(f"[INIT] Loading QUANTIZED model from {mobile_path}")
                # Load full object
                self.model1 = torch.load(str(mobile_path), map_location="cpu")
                return
            else:
                print(f"[INIT] vernex_mobile.pt not found! Falling back to standard search.")
        
        # Standard search: tooltuned > base > nano
        # Check specific priority list
        priorities = ["vernex_tooltuned_*.pt", "vernex_base_*.pt", "vernex_nano_*.pt"]
        weights = []
        for p in priorities:
            found = glob.glob(str(MODEL_DIR / p))
            if found:
                weights = found
                break # Found higher priority class
        
        if not weights:
            # Fallback to any pt
            weights = glob.glob(str(MODEL_DIR / "*.pt"))

        if not weights:
            print("[INIT] No weights found, using random initialization")
            return
            
        latest = max(weights, key=os.path.getctime)
        print(f"[INIT] Loading weights from {Path(latest).name}")
        
        state_dict = torch.load(latest, map_location=self.device)
        
        # Load Agent-1 (should match perfectly or close)
        try:
            self.model1.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"[INIT] Agent-1 partial load: {e}")
            # If architecture mismatch (e.g. loading 500M into 120M config), we should ideally rebuild model1
            # But for now assuming Nano unless configured otherwise.
        
        # Agent-2 has different architecture, load what matches
        if self.model2:
            try:
                # Only load embedding and lm_head which should match
                partial_dict = {k: v for k, v in state_dict.items() 
                                if 'embed' in k or 'lm_head' in k}
                self.model2.load_state_dict(partial_dict, strict=False)
            except:
                pass  # Agent-2 will use random weights for layers
        
        self.model1.eval()
        if self.model2:
            self.model2.eval()
    
    def _generate(self, model, prompt: str, max_tokens: int = 200, temp: float = 0.7) -> str:
        """Generate text from prompt using specified model."""
        max_context = 400 if model == self.model1 else 200
        ids = self.tokenizer.encode(prompt).ids[-max_context:]
        input_ids = torch.tensor([ids]).to(self.device)
        
        model.clear_cache()
        output = ""
        
        with torch.inference_mode():
            # RX 5700 XT Fix: Disable AMP (FP16) to prevent kernel hangs
            # with torch.amp.autocast(self.device, enabled=(self.device == "cuda")):
            if True:
                for token_id in model.generate(
                    input_ids, 
                    max_new_tokens=max_tokens, 
                    tokenizer=self.tokenizer,
                    temperature=temp
                ):
                    text = self.tokenizer.decode([token_id])
                    if "<|im_end|>" in text:
                        break
                    output += text
        
        return output.strip()
    
    def _execute_search(self, text: str) -> Tuple[str, Optional[str]]:
        """Check for [SEARCH: ...] and execute if found."""
        match = re.search(r'\[SEARCH:\s*(.+?)\]', text)
        if match:
            query = match.group(1)
            print(f"\n[SEARCH] {query}")
            result = search_and_fetch(query, num_results=2)
            return text, f"\n<search_result>\n{result[:800]}\n</search_result>\n"
        return text, None
    
    def agent1_respond(self, user_message: str) -> str:
        """Agent-1 generates response."""
        prompt = f"<|im_start|>system\n{AGENT1_SYSTEM}<|im_end|>\n"
        prompt += f"<|im_start|>user\n{user_message}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        
        response = self._generate(self.model1, prompt, max_tokens=300)
        
        # Check for search request
        response, search_result = self._execute_search(response)
        if search_result:
            prompt += response + search_result + "\nBased on search:\n"
            continuation = self._generate(self.model1, prompt, max_tokens=150)
            response = response + search_result + continuation
        
        return response
    
    def agent2_critique(self, user_message: str, agent1_response: str) -> str:
        """Agent-2 critiques Agent-1's response. Lightweight and fast."""
        if not self.model2:
            return ""
        
        # Shorter prompt for faster critique
        prompt = f"<|im_start|>system\n{AGENT2_SYSTEM}<|im_end|>\n"
        prompt += f"<|im_start|>user\n{user_message}<|im_end|>\n"
        prompt += f"<|im_start|>agent1\n{agent1_response[:500]}<|im_end|>\n"  # Truncate for speed
        prompt += "<|im_start|>critic\n"
        
        # Lower temp, fewer tokens for focused critique
        critique = self._generate(self.model2, prompt, max_tokens=100, temp=0.4)
        
        # Check if Agent-2 wants to search
        critique, search_result = self._execute_search(critique)
        if search_result:
            prompt += critique + search_result + "\nConclusion:\n"
            continuation = self._generate(self.model2, prompt, max_tokens=50, temp=0.3)
            critique = critique + search_result + continuation
        
        return critique.strip()
    
    def run(self, user_message: str, show_process: bool = True) -> dict:
        """
        Full dual-agent flow:
        1. Agent-1 responds
        2. Agent-2 critiques (if enabled)
        """
        if show_process:
            print("\n" + "="*50)
            print(f"USER: {user_message}")
            print("="*50)
        
        # Agent-1 generates
        if show_process:
            print("\n[AGENT-1]")
        
        agent1_response = self.agent1_respond(user_message)
        
        if show_process:
            print(agent1_response)
        
        # Agent-2 critiques
        critique = ""
        if self.enable_critic:
            if show_process:
                print("\n[CRITIC]")
            
            critique = self.agent2_critique(user_message, agent1_response)
            
            if show_process:
                print(critique)
        
        if show_process:
            print("="*50 + "\n")
        
        return {
            "user": user_message,
            "agent1": agent1_response,
            "critique": critique,
            "verified": "verified" in critique.lower() or "correct" in critique.lower() if critique else True
        }


def run_test():
    """Run test interactions."""
    system = DualAgentSystem()
    
    test_queries = [
        "How do I clear a JUCE AudioBuffer?",
        "Write a basic delay line in C++",
    ]
    
    for q in test_queries:
        result = system.run(q)
        print(f"Verified: {result['verified']}\n")


def run_interactive():
    """Interactive chat mode - DEFAULT."""
    single_mode = "--single" in sys.argv
    system = DualAgentSystem(enable_critic=not single_mode)
    
    mode_str = "Single Agent" if single_mode else "Dual Agent (with Critic)"
    print(f"\nVernex {mode_str} Mode")
    print("Type 'quit' to exit\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ["quit", "exit", "q"]:
                break
            if not user_input:
                continue
            
            system.run(user_input)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break


if __name__ == "__main__":
    if "--test" in sys.argv:
        run_test()
    else:
        run_interactive()
