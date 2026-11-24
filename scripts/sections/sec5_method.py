# -*- coding: utf-8 -*-
"""Section V: Experimental Methodology & Hardware Testbed - Fully Humanized"""

def get_sec5():
    md = r"""## V. Experimental Methodology & Hardware Testbed

To test EdgeCascade under real-world operating conditions, we set up a physical testbed on a commodity mobile laptop with an NVIDIA GeForce RTX 3050 Laptop GPU. The full hardware, driver, and software stack is detailed in Table II.

```
TABLE II: EXPERIMENTAL WORKSTATION HARDWARE & SOFTWARE SPECIFICATIONS
```

### A. Evaluated Model Stack & Quantization Profiles
Our compound AI pipeline combines three distinct foundation models across different parameter sizes, attention mechanisms, and architectural designs, summarized in Table III.
- **Stage 1 (Gemma 3 4B-it)**: An autoregressive decoder-only transformer developed by Google DeepMind [15]. It features 36 layers, a hidden dimension of 2,560, and 8 query-key-value attention heads with rotary position embeddings (RoPE). We quantize the model to GGUF Q4_K_M format (weight file size: 2,490 MiB).
- **Stage 2 (FLAN-T5 Large)**: A sequence-to-sequence encoder-decoder model with 24 encoder layers and 24 decoder layers using relative position encodings [16]. Quantized to 8-bit dynamic integer precision with PyTorch, it sits entirely in system memory (1,620 MiB footprint) and executes on the host CPU.
- **Stage 3 (Humanizer 12B)**: A 12-billion parameter autoregressive language model tuned for lexical diversification, cadence balancing, and de-biasing [17]. Quantized under GGUF Q4_K_M (7,320 MiB weight file), it poses the primary memory challenge on a 6 GB card.

```
TABLE III: HETEROGENEOUS MODEL STACK & ALLOCATION PARAMETERS
```

### B. Workload Benchmarking Protocol & Telemetry Instrumentation
We evaluated performance across three standardized benchmark workloads:
1. *Short Generation Workload*: 64 input prompt tokens, producing 128 output tokens ($N_{\text{ctx}} = 192$). This mirrors brief summaries and quick technical responses.
2. *Medium Generation Workload*: 128 input prompt tokens, producing 256 output tokens ($N_{\text{ctx}} = 384$). This represents multi-paragraph technical explanations.
3. *Standard Generation Workload*: 256 input prompt tokens, producing 512 output tokens ($N_{\text{ctx}} = 768$). This mirrors full research section drafting and rigorous argumentation.

Throughout all runs, a background telemetry daemon polled the NVIDIA Management Library (NVML) at 10.0 Hz (every 100 milliseconds). The logger tracked:
- `fb_used`: Physical framebuffer (VRAM) occupied on the RTX 3050 Laptop GPU in megabytes.
- `fb_free`: Remaining physical memory headroom.
- `shared_used`: Windows shared system memory committed across the PCIe bus by WDDM.
- `gpu_util`: Graphics engine and tensor core utilization percentage.
- `pcie_rx_tx`: Direct memory access throughput across the PCIe 3.0 x4 bus.
- `cpu_util`: Host AMD Ryzen 7 5800H processor utilization across all 16 execution threads."""
    return md

if __name__ == '__main__':
    print(f"Sec 5 words: {len(get_sec5().split())}")
