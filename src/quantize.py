
import torch
from pathlib import Path
from model import VernexConfig, VernexForCausalLM
from tokenizers import Tokenizer
import os

ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "model"

def quantize_model():
    print("Optimization: Starting Dynamic Quantization (INT8)...")
    
    device = "cpu" # Quantization is processed on CPU
    
    # Load Tokenizer
    tokenizer = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))

    # Config must match 500M Base
    cfg = VernexConfig(
        dim=1024, n_layers=24, n_heads=16, n_kv_heads=8,
        hidden_dim=4096, vocab_size=tokenizer.get_vocab_size(), 
        max_seq_len=512
    )
    
    model = VernexForCausalLM(cfg).to(device)
    
    # Load the FINAL Tool-Tuned model
    # We look for the latest checkpoint from the tool tuning
    import glob
    tool_weights = glob.glob(str(MODEL_DIR / "vernex_tooltuned_*.pt"))
    
    # Fallback to base weights if tool weights don't exist yet (for testing)
    if not tool_weights:
        print("Warning: Tool-tuned weights not found. Looking for base weights...")
        tool_weights = glob.glob(str(MODEL_DIR / "vernex_base_*.pt"))
        
    if not tool_weights:
        print("Error: No weights found to quantize.")
        return

    latest_weight = max(tool_weights, key=os.path.getctime)
    print(f"Loading Model for Quantization: {latest_weight}")
    model.load_state_dict(torch.load(latest_weight, map_location=device))
    model.eval()
    
    # Apply Dynamic Quantization
    # We only quantize Linear and LSTM layers (we utilize Linear heavily)
    print("Quantizing Linear layers to INT8...")
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {torch.nn.Linear}, 
        dtype=torch.qint8
    )
    
    # Save
    output_path = MODEL_DIR / "vernex_mobile.pt"
    # We save the state dict, but to save the full quantized structure properly for easy loading
    # typically usually requires saving the model object or careful handling.
    # For simplicity in this project, we'll save the state dict. 
    # NOTE: Loading a quantized state_dict requires preparing the model structure first.
    # To make it easier for the user, we will save the ENTIRE model object for the mobile version.
    print(f"Saving quantized model to {output_path}...")
    torch.save(quantized_model, str(output_path))
    
    print(f"✅ Quantization Complete!")
    print(f"Original Size: {os.path.getsize(latest_weight) / 1024/1024:.2f} MB")
    print(f"Quantized Size: {os.path.getsize(output_path) / 1024/1024:.2f} MB")

if __name__ == "__main__":
    quantize_model()
