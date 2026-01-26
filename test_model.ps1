param (
    [string]$Prompt = "Write a C++ function to add two numbers"
)

# Set environment variables
$env:HF_HOME = "\\MICAHS-PC\vernex\cache\huggingface"
$env:TORCH_HOME = "\\MICAHS-PC\vernex\cache\torch"
$env:PYTHONPYCACHEPREFIX = "\\MICAHS-PC\vernex\cache\pycache"
$env:PIP_CACHE_DIR = "\\MICAHS-PC\vernex\cache\pip"
$env:TEMP = "\\MICAHS-PC\vernex\cache\temp"
$env:TMP = "\\MICAHS-PC\vernex\cache\temp"

# Change to network directory
Set-Location "\\MICAHS-PC\vernex"

# Prefer the latest checkpoint from current training
$checkpoint = "model\vernex_nano_latest.pt"
if (!(Test-Path $checkpoint)) {
    # Fallback to epoch checkpoints if latest doesn't exist
    $latest = Get-ChildItem "model\vernex_nano_epoch_*.pt" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($latest) {
        $checkpoint = $latest.FullName
    } else {
        Write-Error "No checkpoints found!"
        exit 1
    }
}

Write-Host "Testing checkpoint: $checkpoint"

# Copy to temp file with retry logic
$tempCheckpoint = "model\vernex_test_temp.pt"
$maxRetries = 5
$retryDelay = 2 # seconds

for ($i = 0; $i -lt $maxRetries; $i++) {
    try {
        Copy-Item $checkpoint -Destination $tempCheckpoint -Force -ErrorAction Stop
        Write-Host "Checkpoint copied successfully."
        break
    } catch {
        Write-Warning "Copy failed (attempt $($i+1)/$maxRetries): $_"
        Start-Sleep -Seconds $retryDelay
    }
}

if (!(Test-Path $tempCheckpoint)) {
    Write-Error "Failed to copy checkpoint after $maxRetries attempts."
    exit 1
}

# Run generation
& "C:\Users\jmael\AppData\Local\Programs\Python\Python312\python.exe" src/generate.py "$Prompt" --model "$tempCheckpoint"

# Cleanup
Remove-Item $tempCheckpoint -Force
