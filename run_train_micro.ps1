# Set environment variables to redirect caches and temp files
$env:HF_HOME = "\\MICAHS-PC\vernex\cache\huggingface"
$env:TORCH_HOME = "\\MICAHS-PC\vernex\cache\torch"
$env:PYTHONPYCACHEPREFIX = "\\MICAHS-PC\vernex\cache\pycache"
$env:PIP_CACHE_DIR = "\\MICAHS-PC\vernex\cache\pip"
$env:TEMP = "\\MICAHS-PC\vernex\cache\temp"
$env:TMP = "\\MICAHS-PC\vernex\cache\temp"

# Ensure directories exist
if (!(Test-Path $env:HF_HOME)) { New-Item -ItemType Directory -Path $env:HF_HOME -Force }
if (!(Test-Path $env:TORCH_HOME)) { New-Item -ItemType Directory -Path $env:TORCH_HOME -Force }
if (!(Test-Path $env:PYTHONPYCACHEPREFIX)) { New-Item -ItemType Directory -Path $env:PYTHONPYCACHEPREFIX -Force }
if (!(Test-Path $env:PIP_CACHE_DIR)) { New-Item -ItemType Directory -Path $env:PIP_CACHE_DIR -Force }
if (!(Test-Path $env:TEMP)) { New-Item -ItemType Directory -Path $env:TEMP -Force }

# Change to the network directory
Set-Location "\\MICAHS-PC\vernex"

# Run the training script using the full path to python 3.12
& "C:\Users\jmael\AppData\Local\Programs\Python\Python312\python.exe" src/train_micro.py
