import os
import sys
from huggingface_hub import hf_hub_download

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(ROOT_DIR, "temp", "flan_t5_raw")
os.makedirs(RAW_DIR, exist_ok=True)

print("[FLAN-T5] Downloading pytorch_model.bin using huggingface_hub...")
file_path = hf_hub_download(
    repo_id="alykassem/FLAN-T5-Paraphraser",
    filename="pytorch_model.bin",
    local_dir=RAW_DIR,
    local_dir_use_symlinks=False
)
print(f"[FLAN-T5] Download completed: {file_path} ({os.path.getsize(file_path)/(1024*1024):.2f} MB)")

