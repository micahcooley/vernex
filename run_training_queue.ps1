# Vernex Training Queue Runner
# Runs all training phases in sequence after base training completes

$env:HF_HOME = "\\MICAHS-PC\vernex\cache\huggingface"
$env:TORCH_HOME = "\\MICAHS-PC\vernex\cache\torch"
$env:PYTHONPYCACHEPREFIX = "\\MICAHS-PC\vernex\cache\pycache"
$env:PIP_CACHE_DIR = "\\MICAHS-PC\vernex\cache\pip"
$env:TEMP = "\\MICAHS-PC\vernex\cache\temp"
$env:TMP = "\\MICAHS-PC\vernex\cache\temp"

Set-Location "\\MICAHS-PC\vernex"
$python = "C:\Users\jmael\AppData\Local\Programs\Python\Python312\python.exe"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "VERNEX TRAINING QUEUE" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Phase 1: Chat, Tools, Web Search
Write-Host "`n[1/4] Starting Chat/Tools/Search training..." -ForegroundColor Yellow
& $python src/train_chat_tools.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Phase 1 failed!" -ForegroundColor Red
    exit 1
}
Write-Host "[1/4] Chat/Tools/Search COMPLETE" -ForegroundColor Green

# Phase 2: Critic
Write-Host "`n[2/4] Starting Critic training..." -ForegroundColor Yellow
& $python src/train_critic.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Phase 2 failed!" -ForegroundColor Red
    exit 1
}
Write-Host "[2/4] Critic COMPLETE" -ForegroundColor Green

# Phase 3: Honesty
Write-Host "`n[3/4] Starting Honesty training..." -ForegroundColor Yellow
& $python src/train_honesty.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Phase 3 failed!" -ForegroundColor Red
    exit 1
}
Write-Host "[3/4] Honesty COMPLETE" -ForegroundColor Green

# Phase 4: Reasoning
Write-Host "`n[4/4] Starting Reasoning training..." -ForegroundColor Yellow
& $python src/train_reasoning.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Phase 4 failed!" -ForegroundColor Red
    exit 1
}
Write-Host "[4/4] Reasoning COMPLETE" -ForegroundColor Green

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "ALL TRAINING PHASES COMPLETE!" -ForegroundColor Cyan
Write-Host "Final model: vernex_nano_latest.pt" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
