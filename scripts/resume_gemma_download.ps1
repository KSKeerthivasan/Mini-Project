$target = "C:\Users\ksvin\OneDrive\Desktop\fmini\models\gemma\gemma-3-4b-it-Q4_K_M.gguf"
$url = "https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q4_K_M.gguf"
$expectedSize = 2489757856

while ($true) {
    if (Test-Path $target) {
        $currSize = (Get-Item $target).Length
        Write-Host "Current Gemma size: $([math]::Round($currSize / 1MB, 2)) MB / $([math]::Round($expectedSize / 1MB, 2)) MB ($([math]::Round($currSize / $expectedSize * 100, 2))%)"
        if ($currSize -ge $expectedSize) {
            Write-Host "Gemma Download complete!"
            break
        }
    }
    Write-Host "Resuming Gemma download with curl -C - ..."
    & curl.exe -L -C - --progress-bar -o $target $url
    Start-Sleep -Seconds 2
}

