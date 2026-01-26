from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import torch, json, glob, os, re
from pathlib import Path
from model import VernexConfig, VernexForCausalLM
from tokenizers import Tokenizer
from search import search_and_fetch
from tools import read_file, write_file, edit_file, run_command as exec_cmd, grep_search

# Resolve paths relative to project root
ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "model"

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Model configurations - New family structure
MODEL_CONFIGS = {
    "pro": {"pattern": "vernex_pro_*.pt", "params": "500M", "dim": 1024, "layers": 24, "heads": 16, "kv_heads": 8, "hidden": 4096},
    "base": {"pattern": "vernex_*.pt", "params": "105M", "dim": 768, "layers": 12, "heads": 12, "kv_heads": 4, "hidden": 3072},
    "micro": {"pattern": "vernex_micro_*.pt", "params": "30M", "dim": 384, "layers": 6, "heads": 6, "kv_heads": 2, "hidden": 1536},
}

print("Loading Vernex...")
TOK = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Global model state
current_model_name = "base"
MODEL = None
CFG = None

def load_model(model_name: str):
    """Load a specific model variant."""
    global MODEL, CFG, current_model_name
    
    config = MODEL_CONFIGS.get(model_name, MODEL_CONFIGS["base"])
    pattern = config["pattern"]
    
    # Find checkpoint
    paths = glob.glob(str(MODEL_DIR / pattern))
    # Filter out micro AND pro if we are looking for base
    if model_name == "base":
        paths = [p for p in paths if "micro" not in os.path.basename(p) and "pro" not in os.path.basename(p)]
    
    if not paths:
        print(f"No checkpoints found for {model_name}, falling back to base")
        model_name = "base"
        config = MODEL_CONFIGS["base"]
        paths = glob.glob(str(MODEL_DIR / config["pattern"]))
    
    if not paths:
        raise RuntimeError("No model checkpoints found!")
    
    checkpoint_path = max(paths, key=os.path.getctime)
    print(f"Loading {model_name} from {checkpoint_path}...")
    
    # Copy to temp to avoid lock contention with training
    temp_path = MODEL_DIR / f"server_load_temp_{model_name}.pt"
    import shutil
    try:
        shutil.copy2(checkpoint_path, temp_path)
        load_path = temp_path
    except Exception as e:
        print(f"Warning: Could not copy to temp, trying direct load: {e}")
        load_path = checkpoint_path
    
    # Use config-driven architecture params
    CFG = VernexConfig(
        vocab_size=TOK.get_vocab_size(),
        dim=config.get("dim", 768),
        n_layers=config.get("layers", 12),
        n_heads=config.get("heads", 12),
        n_kv_heads=config.get("kv_heads", 4),
        hidden_dim=config.get("hidden", 3072),
    )
    
    # Create and load model
    MODEL = VernexForCausalLM(CFG).to(DEVICE)
    
    try:
        loaded = torch.load(load_path, map_location=DEVICE, weights_only=False)
        print(f"DEBUG: Loaded checkpoint type: {type(loaded)}")
        # If the checkpoint is the full model object (like our quantized version)
        if isinstance(loaded, torch.nn.Module):
            MODEL = loaded.to(DEVICE)
        else:
            # Standard state_dict loading with strict=False to handle any mismatch
            MODEL.load_state_dict(loaded, strict=False)
        print(f"✓ {model_name} loaded to {DEVICE}! vocab_size={CFG.vocab_size}")
        # Cleanup temp
        if load_path == temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        if load_path == temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
    
    MODEL.eval()
    current_model_name = model_name
    return config["params"]

# Load default model - Pro has reached ultra-low loss (~0.02)
default_model = "pro" if glob.glob(str(MODEL_DIR / "vernex_pro_*.pt")) else "base"
params = load_model(default_model)
print(f"Model ready on {DEVICE}! Params: {params}")

def execute_tool(text):
    """Detect and execute tool calls, return result."""
    # Search
    match = re.search(r'\[SEARCH:\s*(.+?)\]', text)
    if match:
        query = match.group(1)
        print(f"[TOOL] Executing search: {query}")
        result = search_and_fetch(query)
        return f"\n<|tool_result|>\n{result}\n<|end_tool|>\n"
    
    # Read file
    match = re.search(r'\[READ:\s*(.+?)\]', text)
    if match:
        path = match.group(1)
        print(f"[TOOL] Reading: {path}")
        return f"\n<|tool_result|>\n{read_file(path)}\n<|end_tool|>\n"
    
    # Grep
    match = re.search(r'\[GREP:\s*(.+?)\]', text)
    if match:
        pattern = match.group(1)
        print(f"[TOOL] Grep: {pattern}")
        return f"\n<|tool_result|>\n{grep_search(pattern)}\n<|end_tool|>\n"
    
    return None

@app.get("/v1/models")
async def list_models():
    available = []
    for name, cfg in MODEL_CONFIGS.items():
        paths = glob.glob(str(MODEL_DIR / cfg["pattern"]))
        available.append({
            "id": name,
            "params": cfg["params"],
            "ready": len(paths) > 0
        })
    return {"data": available, "current": current_model_name}

@app.post("/v1/switch")
async def switch_model(request: Request):
    data = await request.json()
    model_name = data.get("model", "base")
    
    if model_name not in MODEL_CONFIGS:
        return {"error": f"Unknown model: {model_name}"}
    
    params = load_model(model_name)
    return {"model": model_name, "params": params, "device": DEVICE}

@app.post("/v1/reload")
async def reload_model():
    global current_model_name
    params = load_model(current_model_name)
    return {"status": "success", "model": current_model_name, "params": params}

@app.post("/v1/chat/completions")
async def chat(request: Request):
    global MODEL
    data = await request.json()
    msgs = data.get("messages", [])
    
    # Switch model if requested
    req_model = data.get("model")
    if req_model and req_model != current_model_name:
        load_model(req_model)
    
    # Load System Prompt
    system_prompt = ""
    try:
        with open(ROOT / "SYSTEM_PROMPT.txt", "r", encoding="utf-8") as f:
            system_prompt = f.read()
    except:
        system_prompt = "You are Vernex, an AI assistant."

    prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    for m in msgs[-5:]:
        prompt += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    
    ids = TOK.encode(prompt).ids[-200:]
    input_ids = torch.tensor([ids])
    
    temp = data.get("temperature", 0.7)
    top_p = data.get("top_p", 0.9)
    
    def stream():
        nonlocal input_ids
        full_text = ""
        
        for _ in range(3):  # Max 3 tool loops
            MODEL.clear_cache()
            
            # Autocast for faster generation
            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                generator = MODEL.generate(input_ids.to(DEVICE), max_new_tokens=150, tokenizer=TOK, temperature=temp, top_p=top_p)
                
                chunk = ""
                for token_id in generator:
                    text = TOK.decode([token_id])
                    # Stop if we see any participant markers or end tokens
                    if "<|im_end|>" in text or "<|im_start|>" in text:
                        break
                    chunk += text
                    full_text += text
                    yield f"data: {json.dumps({'choices': [{'delta': {'content': text}}]})}\n\n"
            
            # Check for tool call after accumulating
            tool_result = execute_tool(full_text)
            if tool_result:
                yield f"data: {json.dumps({'choices': [{'delta': {'content': tool_result}}]})}\n\n"
                full_text += tool_result
                # Continue with tool result in context
                new_prompt = prompt + full_text
                input_ids = torch.tensor([TOK.encode(new_prompt).ids[-200:]])
            else:
                # No tool call, we're done
                break
        
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(stream(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
