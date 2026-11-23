import os
import sys
import time
import urllib.request

GEMMA_URL = "https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q4_K_M.gguf"
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "gemma"))
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "gemma-3-4b-it-Q4_K_M.gguf")

def download_with_progress(url, dest_path):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    temp_path = dest_path + ".download"
    
    if os.path.exists(dest_path):
        size = os.path.getsize(dest_path)
        if size > 1_000_000_000:
            print(f"[Gemma] Model already downloaded: {dest_path} ({size / (1024*1024):.2f} MB)")
            return

    print(f"[Gemma] Starting download from: {url}")
    print(f"[Gemma] Saving to: {dest_path}")
    
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(temp_path, "wb") as out_file:
        total_size = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 8 * 1024 * 1024 # 8 MB chunks
        start_time = time.time()
        last_log = start_time
        
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            now = time.time()
            if now - last_log > 3:
                elapsed = now - start_time
                speed = (downloaded / (1024 * 1024)) / (elapsed if elapsed > 0 else 1)
                pct = (downloaded / total_size * 100) if total_size > 0 else 0
                print(f"[Gemma] Downloaded {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({pct:.1f}%) at {speed:.2f} MB/s")
                last_log = now

    if os.path.exists(dest_path):
        os.remove(dest_path)
    os.rename(temp_path, dest_path)
    print(f"[Gemma] Download completed successfully: {dest_path} ({os.path.getsize(dest_path) / (1024*1024):.2f} MB)")

if __name__ == "__main__":
    download_with_progress(GEMMA_URL, OUTPUT_FILE)

