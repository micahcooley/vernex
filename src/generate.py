import torch
from model import VernexConfig, VernexForCausalLM
from tokenizers import Tokenizer
import sys, os, glob, re
from search import search_and_fetch
from tools import read_file, write_file, edit_file, run_command as exec_cmd, grep_search

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

def generate(prompt, model_path="c:/vernex/model/vernex_nano_latest.pt", max_tokens=300):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = Tokenizer.from_file("c:/vernex/model/tokenizer.json")
    cfg = VernexConfig(dim=256, n_layers=8, n_heads=8, n_kv_heads=4, vocab_size=tokenizer.get_vocab_size(), max_seq_len=512)
    model = VernexForCausalLM(cfg).to(device)
    
    if not os.path.exists(model_path):
        paths = glob.glob("c:/vernex/model/vernex_nano_epoch_*.pt")
        model_path = max(paths, key=os.path.getctime) if paths else None
    
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded: {model_path}")
    else:
        print("No weights found.")
    
    model.eval()
    
    sys_prompt = ""
    if os.path.exists("c:/vernex/SYSTEM_PROMPT.txt"):
        with open("c:/vernex/SYSTEM_PROMPT.txt", "r") as f: sys_prompt = f.read()
            
    chat = f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    print(f"\nUser: {prompt}")
    print("Vernex: ", end="", flush=True)

    for _ in range(5):  # Max tool loops
        ids = tokenizer.encode(chat).ids
        tensor = torch.tensor([ids], device=device)
        
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
    prompt = sys.argv[1] if len(sys.argv) > 1 else "What files are in the src folder?"
    generate(prompt)
