
#!/bin/bash
# run_queue.sh - Queues 500M Training -> Tool Tuning -> Quantization

# --- STEP 1: Wait for 120M Run ---
echo "[QUEUE] Step 1: Watching for 'src/train.py' (120M run) to finish..."
while pgrep -f "src/train.py" > /dev/null; do
    sleep 60
done
echo "[QUEUE] 120M training finished."

# --- STEP 2: Run 500M Base Training ---
echo "[QUEUE] Step 2: Starting Optimized 500M CPU Training (Log: training_500m.log)..."
source .venv/bin/activate
export HIP_VISIBLE_DEVICES=-1
export CUDA_VISIBLE_DEVICES=-1
python3 src/train_500m.py > training_500m.log 2>&1

echo "[QUEUE] 500M Training Complete."

# --- STEP 3: Run Tool-Use Fine-Tuning ---
echo "[QUEUE] Step 3: Starting Tool-Use Fine-Tuning (Log: training_tools.log)..."
export HIP_VISIBLE_DEVICES=-1
export CUDA_VISIBLE_DEVICES=-1
python3 src/train_tools.py > training_tools.log 2>&1

echo "[QUEUE] Tool Fine-Tuning Complete."

# --- STEP 4: Quantization (INT8) ---
echo "[QUEUE] Step 4: Quantizing Model for Mobile/Laptop (Log: quantization.log)..."
export HIP_VISIBLE_DEVICES=-1
export CUDA_VISIBLE_DEVICES=-1
python3 src/quantize.py > quantization.log 2>&1

echo "[QUEUE] ALL OPERATIONS COMPLETE."
echo "Generated Models:"
echo "- Nano (120M): model/vernex_nano_latest.pt"
echo "- Base (500M): model/vernex_base_latest.pt"
echo "- Smart (Tool): model/vernex_tooltuned_epoch_1.pt"
echo "- Mobile (FAST): model/vernex_mobile.pt"
