# -*- coding: utf-8 -*-
"""Section V: Experimental Methodology & Testbed"""

def get_sec5():
    md = r"""## V. Experimental Methodology & Hardware Testbed

To rigorously evaluate EdgeCascade under realistic operating constraints, we deployed the framework on a physical commodity mobile workstation configured with an NVIDIA GeForce RTX 3050 Laptop GPU. The complete hardware, driver, and software environment is cataloged in Table II.

```
+--------------------------------------------------------------------------------------------------------------------+
|                               TABLE II: EXPERIMENTAL WORKSTATION HARDWARE & SOFTWARE SPECIFICATIONS                |
+------------------------------+-------------------------------------------------------------------------------------+
| Component / Subsystem        | Hardware & Operating Configuration Specifications                                  |
+------------------------------+-------------------------------------------------------------------------------------+
| Host Processor (CPU)         | AMD Ryzen 7 5800H (8 Physical Cores, 16 Execution Threads @ 3.20 GHz, Boost 4.4 GHz)|
| Host System Memory (RAM)     | 16,384 MiB (16 GB) DDR4 Dual-Channel @ 3200 MT/s (Peak Bandwidth: 51.2 GB/s)        |
| Dedicated Graphics (GPU)     | NVIDIA GeForce RTX 3050 Laptop GPU (GA107 Silicon, 2,048 CUDA Cores, 64 Tensor C.) |
| Physical Video Memory (VRAM) | 6,144 MiB (6 GB) GDDR6, 96-bit Memory Interface (Peak Bandwidth: 168.0 GB/s)       |
| Interconnect Bus Interface   | PCI Express 3.0 x4 (Theoretical Bidirectional Bandwidth: 3.94 GB/s)                 |
| Operating System & Kernel    | Microsoft Windows 11 Home 64-bit (Build 22631.4317)                                |
| Graphics Driver Subsystem    | Windows Display Driver Model (WDDM 3.1), NVIDIA Game Ready Driver 551.86            |
| Deep Learning Runtimes       | llama.cpp CUDA Runtime (Build b2840, CUDA 12.4, cuBLAS), PyTorch 2.4.0+cpu, HF Trans.|
| Telemetry & Monitoring Tools | NVIDIA Management Library (NVML v12), nvidia-smi 100ms continuous sampling daemon  |
+------------------------------+-------------------------------------------------------------------------------------+
```

### A. Evaluated Model Stack & Quantization Profiles
The evaluated compound AI ensemble comprises three heterogeneous foundation models spanning diverse parameter scales, attention mechanisms, and architectural designs, detailed in Table III.
- **Stage 1 (Gemma 3 4B-it)**: An autoregressive decoder-only transformer developed by Google DeepMind [15]. It incorporates 36 layers, a hidden dimension of 2,560, and 8 query-key-value attention heads with rotary position embeddings (RoPE). The model is quantized to GGUF `Q4_K_M` format (static weight size: 2,490 MiB).
- **Stage 2 (FLAN-T5 Large)**: A sequence-to-sequence encoder-decoder model comprising 24 encoder layers and 24 decoder layers with relative position representations [16]. Quantized to 8-bit integer precision via PyTorch dynamic quantization, it occupies 1,620 MiB of system memory and runs entirely on the host CPU.
- **Stage 3 (Humanizer 12B)**: A 12-billion parameter autoregressive language model fine-tuned specifically for syntactic de-biasing, lexical diversification, and cadence alignment [17]. Quantized under GGUF `Q4_K_M` (static weight size: 7,320 MiB), it represents the primary memory challenge on a 6 GB card.

```
+--------------------------------------------------------------------------------------------------------------------+
|                               TABLE III: HETEROGENEOUS MODEL STACK & ALLOCATION PARAMETERS                         |
+----------------------+--------------------+--------------------+--------------------+--------------------+---------+
| Pipeline Phase       | Model Architecture | Parameter Volume   | Quantization Type  | Runtime Execution  | Active  |
|                      | & Base Checkpoint  | & Component Layers | & File Size        | Target & Threading | Layers  |
+----------------------+--------------------+--------------------+--------------------+--------------------+---------+
| Stage 1: Drafting    | Gemma 3 4B-it [15] | 4.22 Billion (36L) | GGUF Q4_K_M (2.4GB)| Dedicated GPU      | ngl=99  |
| Stage 2: Verification| FLAN-T5 Large [16] | 770 Million (48L)  | PyTorch INT8(1.6GB)| Host CPU (16 Thr.) | ngl=0   |
| Stage 3: Refinement  | Humanizer 12B [17] | 12.18 Billion (40L)| GGUF Q4_K_M (7.3GB)| Hybrid (GPU + CPU) | ngl=20  |
+----------------------+--------------------+--------------------+--------------------+--------------------+---------+
```

### B. Workload Benchmarking Protocol & Telemetry Instrumentation
To evaluate performance across varying operational scales, we established three standardized evaluation workloads:
1. *Short Generation Workload*: 64 input prompt tokens, generating 128 output tokens ($N_{\text{ctx}} = 192$). Representative of conversational queries and brief summaries.
2. *Medium Generation Workload*: 128 input prompt tokens, generating 256 output tokens ($N_{\text{ctx}} = 384$). Representative of multi-paragraph technical abstracts and explanations.
3. *Standard Generation Workload*: 256 input prompt tokens, generating 512 output tokens ($N_{\text{ctx}} = 768$). Representative of full-length research section drafting and detailed scientific argumentation.

Throughout all benchmarking runs, dedicated hardware telemetry was recorded via a dedicated background asynchronous daemon querying NVML at a continuous sampling frequency of 10.0 Hz (every 100 milliseconds). The telemetry engine captures:
- `fb_used`: Physical framebuffer (VRAM) occupied on the RTX 3050 Laptop GPU in megabytes.
- `fb_free`: Remaining physical memory headroom.
- `shared_used`: Windows shared system memory committed across the PCIe bus by WDDM.
- `gpu_util`: Tensor core and graphics engine utilization percentage.
- `pcie_rx_tx`: Direct Memory Access transfer throughput across the PCIe 3.0 x4 bus.
- `cpu_util`: Host AMD Ryzen 7 5800H processor utilization across all 16 execution threads."""
    return md

if __name__ == '__main__':
    print(f"Sec 5 word count: {len(get_sec5().split())}")
