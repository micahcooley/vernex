import torch
from model import VernexConfig, VernexForCausalLM
from tokenizers import Tokenizer
import sys, os, glob, re
from pathlib import Path
from search import search_and_fetch
from tools import read_file, write_file, edit_file, run_command as exec_cmd, grep_search

# Resolve paths
ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "model"

SESSION_EDITS = []

def execute_tool(tag_name, content):
    """Route tool calls to actual implementations."""
    if tag_name == "SEARCH":
        return search_and_fetch(content)
    elif tag_name == "READ":
        return read_file(content)
    elif tag_name == "RUN":
        return exec_cmd(content)
    elif tag_name == "GREP":
        return grep_search(content)
    elif tag_name == "WRITE":
        # Content format: path\ncontent\n[END_WRITE]
        lines = content.split("\n", 1)
        if len(lines) == 2:
            path = lines[0].strip()
            file_content = lines[1].replace("[END_WRITE]", "").strip()
            return write_file(path, file_content)
    elif tag_name == "EDIT":
        # Content: path\nold\n[REPLACE]\nnew\n[END_EDIT]
        parts = content.split("[REPLACE]")
        if len(parts) == 2:
            header, replacement = parts
            header_lines = header.strip().split("\n", 1)
            path = header_lines[0].strip()
            search = header_lines[1].strip() if len(header_lines) > 1 else ""
            replacement = replacement.replace("[END_EDIT]", "").strip()
            return edit_file(path, search, replacement)
    return "Unknown tool."

def generate(prompt, model_path=None, max_tokens=300):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    print(f"DEBUG: Tokenizer vocab size: {tokenizer.get_vocab_size()}")
    # Vernex Nano Config
    cfg = VernexConfig(dim=768, n_layers=12, n_heads=12, n_kv_heads=4, hidden_dim=3072, vocab_size=tokenizer.get_vocab_size(), max_seq_len=512)
    model = VernexForCausalLM(cfg).to(device)
    
    if model_path is None:
        model_path = MODEL_DIR / "vernex_nano_latest.pt"
    
    if not model_path.exists():
        paths = glob.glob(str(MODEL_DIR / "vernex_nano_epoch_*.pt"))
        model_path = Path(max(paths, key=os.path.getctime)) if paths else None
    
    if model_path:
        try:
            model.load_state_dict(torch.load(str(model_path), map_location=device))
            print(f"Loaded: {model_path.name}")
        except Exception as e:
            print(f"DEBUG: Caught exception: {type(e).__name__}: {e}")
            if "size mismatch" in str(e):
                print("DEBUG: Vocab mismatch detected. Retrying with vocab_size=50257...")
                cfg = VernexConfig(dim=768, n_layers=12, n_heads=12, n_kv_heads=4, hidden_dim=3072, vocab_size=50257, max_seq_len=512)
                model = VernexForCausalLM(cfg).to(device)
                model.load_state_dict(torch.load(str(model_path), map_location=device))
                print(f"Loaded with adjusted vocab: {model_path.name}")
            else:
                raise e
    else:
        print("No weights found.")
    
    model.eval()
    
    sys_prompt = ""
    system_prompt_path = ROOT / "SYSTEM_PROMPT.txt"
    if system_prompt_path.exists():
        with open(system_prompt_path, "r") as f: 
            sys_prompt = f.read()
            
    chat = f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    print(f"\nUser: {prompt}")
    print("Vernex: ", end="", flush=True)

    for _ in range(5):  # Max tool loops
        ids = tokenizer.encode(chat).ids
        tensor = torch.tensor([ids[-400:]], device=device)  # Keep context manageable
        
        gen = ""
        for _ in range(max_tokens):
            with torch.no_grad():
                logits, _ = model(tensor)
                next_tok = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
                tensor = torch.cat([tensor, next_tok], dim=1)
                tok = tokenizer.decode([next_tok.item()])
                gen += tok
                print(tok, end="", flush=True)
                if "<|im_end|>" in gen: break
        
        chat += gen
        
        # Multi-tool detection
        tool_patterns = [
            (r"\[SEARCH: (.*?)\]", "SEARCH"),
            (r"\[READ: (.*?)\]", "READ"),
            (r"\[RUN: (.*?)\]", "RUN"),
            (r"\[GREP: (.*?)\]", "GREP"),
            (r"\[WRITE: (.*?)\[END_WRITE\]", "WRITE"),
            (r"\[EDIT: (.*?)\[END_EDIT\]", "EDIT"),
        ]
        
        tool_found = False
        for pattern, name in tool_patterns:
            match = re.search(pattern, gen, re.DOTALL)
            if match:
                result = execute_tool(name, match.group(1))
                chat += f"\n<|tool_result|>\n{result}<|im_end|>\n<|im_start|>assistant\n"
                print(f"\n[Tool: {name}] ", end="", flush=True)
                tool_found = True
                break
        
        if not tool_found:
            break

    print("\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="?", default="What files are in the src folder?")
    parser.add_argument("--model", help="Path to model checkpoint", default=None)
    args = parser.parse_args()
    
    model_path = Path(args.model) if args.model else None
    generate(args.prompt, model_path=model_path)
