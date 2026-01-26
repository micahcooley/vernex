
import torch
import sys
import time

print(f"PyTorch Version: {torch.__version__}")
print("Checking availability...")

if not torch.cuda.is_available():
    print("FAIL: No CUDA/ROCm device detected.")
    sys.exit(1)

print(f"Device count: {torch.cuda.device_count()}")
print(f"Device Name: {torch.cuda.get_device_name(0)}")

print("Attempting small tensor allocation...")
try:
    x = torch.ones(1024, 1024, device="cuda")
    print("Allocation success.")
    
    print("Attempting multiplication...")
    start = time.time()
    y = x * x
    torch.cuda.synchronize() # Force wait for GPU
    print(f"Operation success in {time.time() - start:.4f}s")
    print("GPU IS WORKING.")
except Exception as e:
    print(f"CRASH: {e}")
