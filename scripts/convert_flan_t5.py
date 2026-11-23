import os
import sys
import subprocess
import urllib.request
import json
import shutil

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEMP_DIR = os.path.join(ROOT_DIR, "temp")
RAW_MODEL_DIR = os.path.join(TEMP_DIR, "flan_t5_raw")
MODELS_DIR = os.path.join(ROOT_DIR, "models", "flan-t5")
RUNTIME_DIR = os.path.join(ROOT_DIR, "runtime", "llama.cpp")
F16_GGUF = os.path.join(TEMP_DIR, "flan-t5-paraphraser-f16.gguf")
FINAL_Q4_GGUF = os.path.join(MODELS_DIR, "flan-t5-large-paraphraser-Q4_K_M.gguf")
FINAL_F16_GGUF = os.path.join(MODELS_DIR, "flan-t5-large-paraphraser-f16.gguf")

HF_REPO_URL = "https://huggingface.co/alykassem/FLAN-T5-Paraphraser/resolve/main"
FILES_TO_DOWNLOAD = [
    "config.json",
    "pytorch_model.bin",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json"
]
SPIECE_URL = "https://huggingface.co/google/flan-t5-large/resolve/main/spiece.model"

def download_file(url, target_path):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    if os.path.exists(target_path):
        size = os.path.getsize(target_path)
        if size > 1000:
            print(f"[FLAN-T5] Already exists: {os.path.basename(target_path)} ({size / (1024*1024):.2f} MB)")
            return
    print(f"[FLAN-T5] Downloading: {url} -> {target_path}")
    # Using curl for robust download with resume & progress
    ret = subprocess.run(["curl.exe", "-L", "--progress-bar", "-o", target_path, url])
    if ret.returncode != 0:
        raise RuntimeError(f"Failed to download {url}")

def setup_flan_t5():
    os.makedirs(RAW_MODEL_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("=== Step 1: Downloading alykassem/FLAN-T5-Paraphraser checkpoint ===")
    for fname in FILES_TO_DOWNLOAD:
        url = f"{HF_REPO_URL}/{fname}"
        target = os.path.join(RAW_MODEL_DIR, fname)
        download_file(url, target)

    print("=== Step 2: Downloading SentencePiece tokenizer (spiece.model) from google/flan-t5-large ===")
    spiece_target = os.path.join(RAW_MODEL_DIR, "spiece.model")
    download_file(SPIECE_URL, spiece_target)

    print("=== Step 3: Converting to GGUF using llama.cpp converter ===")
    converter_script = os.path.join(TEMP_DIR, "llama_repo", "convert_hf_to_gguf.py")
    python_exe = sys.executable

    convert_cmd = [
        python_exe,
        converter_script,
        RAW_MODEL_DIR,
        "--outfile", F16_GGUF,
        "--outtype", "f16"
    ]
    print(f"Running conversion: {' '.join(convert_cmd)}")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(TEMP_DIR, "llama_repo", "gguf-py") + os.pathsep + env.get("PYTHONPATH", "")
    
    ret = subprocess.run(convert_cmd, env=env)
    if ret.returncode != 0:
        raise RuntimeError("GGUF conversion failed!")

    print(f"Successfully converted to f16 GGUF: {F16_GGUF} ({os.path.getsize(F16_GGUF) / (1024*1024):.2f} MB)")

    print("=== Step 4: Quantizing to Q4_K_M using llama-quantize ===")
    quantize_exe = os.path.join(RUNTIME_DIR, "llama-quantize.exe")
    quantize_cmd = [
        quantize_exe,
        F16_GGUF,
        FINAL_Q4_GGUF,
        "Q4_K_M"
    ]
    print(f"Running quantize: {' '.join(quantize_cmd)}")
    q_ret = subprocess.run(quantize_cmd)
    
    # Check if quantize succeeded or if Q4 is compatible with T5
    if q_ret.returncode == 0 and os.path.exists(FINAL_Q4_GGUF) and os.path.getsize(FINAL_Q4_GGUF) > 100_000_000:
        print(f"Quantization succeeded: {FINAL_Q4_GGUF} ({os.path.getsize(FINAL_Q4_GGUF) / (1024*1024):.2f} MB)")
        target_model_to_test = FINAL_Q4_GGUF
    else:
        print("Quantize failed or produced invalid file, keeping f16 GGUF as fallback.")
        shutil.copyfile(F16_GGUF, FINAL_F16_GGUF)
        target_model_to_test = FINAL_F16_GGUF

    print("=== Step 5: Testing llama-cli inference on FLAN-T5 Paraphraser ===")
    cli_exe = os.path.join(RUNTIME_DIR, "llama-cli.exe")
    test_cmd = [
        cli_exe,
        "-m", target_model_to_test,
        "-p", "paraphrase: Artificial intelligence is transforming the technological landscape across the entire world.",
        "-n", "64",
        "--temp", "0.7",
        "-ngl", "0"
    ]
    print(f"Running test: {' '.join(test_cmd)}")
    test_ret = subprocess.run(test_cmd, capture_output=True, text=True, timeout=120)
    print("Test Return Code:", test_ret.returncode)
    print("Test Stdout:\n", test_ret.stdout)
    if test_ret.stderr:
        print("Test Stderr:\n", test_ret.stderr[-500:])

    print("=== Step 6: Cleaning up temporary raw weights ===")
    if os.path.exists(RAW_MODEL_DIR):
        shutil.rmtree(RAW_MODEL_DIR, ignore_errors=True)
    if os.path.exists(F16_GGUF) and target_model_to_test != F16_GGUF:
        try:
            os.remove(F16_GGUF)
        except Exception:
            pass
    print("FLAN-T5 Paraphraser setup and verification completed!")

if __name__ == "__main__":
    setup_flan_t5()

