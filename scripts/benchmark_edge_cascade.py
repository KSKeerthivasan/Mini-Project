import asyncio
import os
import sys
import time
import csv
import threading
import subprocess
from datetime import datetime

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from backend.services.llama_manager import llama_manager
from backend.services.gemma_service import gemma_service
from backend.services.flan_t5_service import flan_t5_service
from backend.services.humanizer_service import humanizer_service

LOGS_DIR = os.path.join(ROOT_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
BENCHMARK_CSV = os.path.join(LOGS_DIR, "edge_cascade_telemetry.csv")
VRAM_LOG_CSV = os.path.join(LOGS_DIR, "vram_timeline_samples.csv")

class HighFreqVRAMSampler:
    def __init__(self, interval_sec=0.1):
        self.interval = interval_sec
        self.running = False
        self.thread = None
        self.samples = []

    def _sample_loop(self):
        while self.running:
            try:
                cmd = ["nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"]
                out = subprocess.check_output(cmd, encoding="utf-8").strip()
                used, free, util = [float(x.strip()) for x in out.split(",")]
                self.samples.append((time.time(), used, free, util))
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        self.running = True
        self.samples = []
        self.thread = threading.Thread(target=self._sample_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        return self.samples

WORKLOAD_PROMPTS = [
    {
        "length_category": "Short",
        "target_tokens": 128,
        "prompt": "Explain the concept of quantum superposition and how qubits operate in simple physical terms."
    },
    {
        "length_category": "Medium",
        "target_tokens": 256,
        "prompt": "Discuss the primary bottlenecks of on-device large language model inference on consumer GPUs, focusing on memory bandwidth, VRAM constraints, and quantization trade-offs."
    },
    {
        "length_category": "Standard",
        "target_tokens": 512,
        "prompt": "Provide a comprehensive technical overview of modern deep learning compiler architectures, examining computational graph representations, kernel fusion, and hardware-specific code generation for edge processors."
    }
]

async def run_benchmark_trial(trial_id: int, workload: dict, csv_writer, file_handle):
    cat = workload["length_category"]
    prompt = workload["prompt"]
    print(f"\n>>> [Trial {trial_id}] Workload: {cat} (Target ~{workload['target_tokens']} tokens) >>>")

    sampler = HighFreqVRAMSampler(interval_sec=0.1)
    sampler.start()

    t_trial_start = time.time()

    # Stage 0: Initial Idle State
    hw_idle = llama_manager.get_hardware_info()
    v_idle = hw_idle["vram"]["used_mb"]

    # --- STAGE 1: GEMMA 3 4B ---
    t_s1_start = time.time()
    gemma_res = await gemma_service.generate(prompt=prompt, content_type="Article", content_length=cat)
    t_s1_end = time.time()
    t_s1_total = t_s1_end - t_s1_start
    hw_s1 = llama_manager.get_hardware_info()
    v_peak_s1 = hw_s1["vram"]["used_mb"]

    # Unload S1 & Measure Valley
    t_unload_s1_start = time.time()
    await llama_manager.unload()
    t_unload_s1 = time.time() - t_unload_s1_start
    hw_valley_1 = llama_manager.get_hardware_info()
    v_valley_1 = hw_valley_1["vram"]["used_mb"]

    # --- STAGE 2: FLAN-T5 ---
    t_s2_start = time.time()
    flan_res = await flan_t5_service.restructure(text=gemma_res["content"])
    t_s2_end = time.time()
    t_s2_total = t_s2_end - t_s2_start
    hw_s2 = llama_manager.get_hardware_info()
    v_peak_s2 = hw_s2["vram"]["used_mb"]

    # Unload S2 & Measure Valley
    t_unload_s2_start = time.time()
    await llama_manager.unload()
    t_unload_s2 = time.time() - t_unload_s2_start
    hw_valley_2 = llama_manager.get_hardware_info()
    v_valley_2 = hw_valley_2["vram"]["used_mb"]

    # --- STAGE 3: HUMANIZER 12B ---
    t_s3_start = time.time()
    humanizer_res = await humanizer_service.humanize(draft_text=flan_res["content"])
    t_s3_end = time.time()
    t_s3_total = t_s3_end - t_s3_start
    hw_s3 = llama_manager.get_hardware_info()
    v_peak_s3 = hw_s3["vram"]["used_mb"]

    # Unload S3 & Measure Final Valley
    t_unload_s3_start = time.time()
    await llama_manager.unload()
    t_unload_s3 = time.time() - t_unload_s3_start
    hw_final = llama_manager.get_hardware_info()
    v_valley_final = hw_final["vram"]["used_mb"]

    t_trial_end = time.time()
    total_latency = t_trial_end - t_trial_start

    samples = sampler.stop()
    v_max_overall = max([s[1] for s in samples]) if samples else max(v_peak_s1, v_peak_s3)

    row = [
        trial_id,
        cat,
        round(v_idle, 1),
        round(v_peak_s1, 1),
        round(v_valley_1, 1),
        round(v_peak_s2, 1),
        round(v_valley_2, 1),
        round(v_peak_s3, 1),
        round(v_valley_final, 1),
        round(v_max_overall, 1),
        round(t_s1_total, 2),
        round(t_unload_s1, 2),
        round(t_s2_total, 2),
        round(t_unload_s2, 2),
        round(t_s3_total, 2),
        round(t_unload_s3, 2),
        round(total_latency, 2),
        "SUCCESS"
    ]
    csv_writer.writerow(row)
    file_handle.flush()

    print(f"Trial {trial_id} Finished: Max VRAM = {v_max_overall:.1f} MB, Total Time = {total_latency:.2f}s, Final Valley = {v_valley_final:.1f} MB")
    return samples

async def main():
    print("=====================================================================")
    print("       EDGECASCADE EMPIRICAL TELEMETRY BENCHMARK HARNESS            ")
    print("  Host: NVIDIA RTX 3050 Laptop GPU (6,144 MiB VRAM), Windows 11      ")
    print("=====================================================================")

    # Initialize CSV files
    file_exists = os.path.isfile(BENCHMARK_CSV)
    f_bench = open(BENCHMARK_CSV, "w", newline="", encoding="utf-8")
    writer = csv.writer(f_bench)
    headers = [
        "trial_id", "workload", "v_idle_mb", 
        "v_peak_s1_mb", "v_valley_s1_mb", 
        "v_peak_s2_mb", "v_valley_s2_mb", 
        "v_peak_s3_mb", "v_valley_final_mb", 
        "v_max_overall_mb", 
        "t_s1_sec", "t_unload_s1_sec", 
        "t_s2_sec", "t_unload_s2_sec", 
        "t_s3_sec", "t_unload_s3_sec", 
        "total_latency_sec", "status"
    ]
    writer.writerow(headers)
    f_bench.flush()

    # Pre-clean any lingering processes
    await llama_manager.unload()

    all_vram_timeline = []
    trial_count = 1
    # Run 2 passes across Short, Medium, Standard (Total 6 full cascaded runs to establish empirical variance)
    for repetition in range(1, 3):
        print(f"\n=== Commencing Pass {repetition}/2 across workloads ===")
        for wl in WORKLOAD_PROMPTS:
            samples = await run_benchmark_trial(trial_count, wl, writer, f_bench)
            for s in samples:
                all_vram_timeline.append((trial_count, s[0], s[1], s[2], s[3]))
            trial_count += 1
            # Brief cool-off between trials
            time.sleep(2.0)

    f_bench.close()

    # Save VRAM timeline samples for plotting
    with open(VRAM_LOG_CSV, "w", newline="", encoding="utf-8") as f_samples:
        s_writer = csv.writer(f_samples)
        s_writer.writerow(["trial_id", "timestamp", "vram_used_mb", "vram_free_mb", "gpu_util_pct"])
        for entry in all_vram_timeline:
            s_writer.writerow(entry)

    print("\n=====================================================================")
    print(f"BENCHMARK COMPLETED: Telemetry saved to {BENCHMARK_CSV}")
    print(f"Timeline samples saved to {VRAM_LOG_CSV}")
    print("=====================================================================")

if __name__ == "__main__":
    asyncio.run(main())
