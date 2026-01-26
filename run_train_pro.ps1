# Vernex Pro (500M) Training Runner
$env:HF_HOME = "\\MICAHS-PC\vernex\cache\huggingface"
$env:TORCH_HOME = "\\MICAHS-PC\vernex\cache\torch"
$env:PYTHONPYCACHEPREFIX = "\\MICAHS-PC\vernex\cache\pycache"
$env:PIP_CACHE_DIR = "\\MICAHS-PC\vernex\cache\pip"
$env:TEMP = "\\MICAHS-PC\vernex\cache\temp"
$env:TMP = "\\MICAHS-PC\vernex\cache\temp"

Set-Location "\\MICAHS-PC\vernex"
$python = "C:\Users\jmael\AppData\Local\Programs\Python\Python312\python.exe"

Write-Host "Starting Vernex Pro (500M) Training..." -ForegroundColor Cyan
& $python src/train_pro.py
