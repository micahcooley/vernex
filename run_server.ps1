# Set environment variables to redirect caches and temp files
$env:HF_HOME = "\\MICAHS-PC\vernex\cache\huggingface"
$env:TORCH_HOME = "\\MICAHS-PC\vernex\cache\torch"
$env:PYTHONPYCACHEPREFIX = "\\MICAHS-PC\vernex\cache\pycache"
$env:PIP_CACHE_DIR = "\\MICAHS-PC\vernex\cache\pip"
$env:TEMP = "\\MICAHS-PC\vernex\cache\temp"
$env:TMP = "\\MICAHS-PC\vernex\cache\temp"

# Change to the network directory
Set-Location "\\MICAHS-PC\vernex"

# Run the server using the full path to python 3.12
& "C:\Users\jmael\AppData\Local\Programs\Python\Python312\python.exe" src/server.py
