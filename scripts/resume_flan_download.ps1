$target = "C:\Users\ksvin\OneDrive\Desktop\fmini\temp\flan_t5_raw\pytorch_model.bin"
$url = "https://huggingface.co/alykassem/FLAN-T5-Paraphraser/resolve/main/pytorch_model.bin"
$expectedSize = 3132680487

while ($true) {
    if (Test-Path $target) {
        $currSize = (Get-Item $target).Length
        Write-Host "Current size: $([math]::Round($currSize / 1MB, 2)) MB / $([math]::Round($expectedSize / 1MB, 2)) MB ($([math]::Round($currSize / $expectedSize * 100, 2))%)"
        if ($currSize -ge $expectedSize) {
            Write-Host "Download complete!"
            break
        }
    }
    Write-Host "Resuming download with curl -C - ..."
    & curl.exe -L -C - --progress-bar -o $target $url
    Start-Sleep -Seconds 2
}

