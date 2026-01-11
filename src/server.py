from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import torch, json, glob, os, re
from model import VernexConfig, VernexForCausalLM
from tokenizers import Tokenizer
from search import search_and_fetch
from tools import read_file, write_file, edit_file, run_command as exec_cmd, grep_search

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

print("Loading Vernex...")
TOK = Tokenizer.from_file("c:/vernex/model/tokenizer.json")
CFG = VernexConfig(vocab_size=TOK.get_vocab_size())
MODEL = VernexForCausalLM(CFG)

paths = glob.glob("c:/vernex/model/vernex_nano_*.pt")
if paths:
    try:
        MODEL.load_state_dict(torch.load(max(paths, key=os.path.getctime), map_location="cpu"))
        print("Weights loaded!")
    except Exception as e:
        print(f"Using random weights: {e}")

MODEL.eval()
print(f"Model ready! Params: {sum(p.numel() for p in MODEL.parameters())/1e6:.1f}M")

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
    return {"data": [{"id": "vernex-nano", "object": "model"}]}

@app.post("/v1/chat/completions")
async def chat(request: Request):
    data = await request.json()
    msgs = data.get("messages", [])
    
    prompt = ""
    for m in msgs[-3:]:
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
            generator = MODEL.generate(input_ids, max_new_tokens=150, tokenizer=TOK, temperature=temp, top_p=top_p)
            
            chunk = ""
            for token_id in generator:
                text = TOK.decode([token_id])
                if "<|im_end|>" in text: break
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
                    break
            else:
                # No tool call, we're done
                break
        
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(stream(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
