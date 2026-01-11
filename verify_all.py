import subprocess
import sys

def run(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Error running {cmd}")
        sys.exit(1)

def main():
    print("--- 1. Training Tokenizer ---")
    run(f"{sys.executable} src/tokenizer.py")
    
    print("\n--- 2. Training Model (Nano) ---")
    run(f"{sys.executable} src/train.py")
    
    print("\n--- 3. Testing Inference ---")
    run(f"{sys.executable} src/generate.py")
    
    print("\nSUCCESS! Vernex is working.")

if __name__ == "__main__":
    main()
