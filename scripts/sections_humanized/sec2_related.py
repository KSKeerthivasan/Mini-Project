# -*- coding: utf-8 -*-
"""Section II: Related Work & Taxonomic Foundation - Humanized Voice"""

def get_sec2():
    md = r"""## II. Related Work & Taxonomic Foundation

Serving multi-billion parameter models on edge devices spans four distinct technical fields: weight quantization, memory offloading, compound AI pipelines, and operating system graphics virtualization. We contrast existing approaches in Table I.

### A. Post-Training Quantization & Compressed Representations
Quantizing model weights is the first defense against tight memory constraints. Early systems like LLM.int8() [26] showed that 16-bit floating-point weights (FP16/BF16) could be packed into 8-bit integers (INT8) with almost zero perplexity loss, provided that extreme outlier values are kept in high precision. Modern methods push deeper into sub-4-bit representations. GPTQ [27] uses second-order Hessian approximations to quantize layers sequentially while minimizing layer-wise error. AWQ (Activation-aware Weight Quantization) [28] proved that protecting the top 1% of salient weight channels based on activation magnitude preserves model behavior without retraining.

On consumer devices, the GGUF binary format from llama.cpp [18] has become the practical standard. GGUF introduces structured block quantization (k-quants), mixing different bit-widths across attention projections and feed-forward networks. We use 4-bit medium quantization (Q4_K_M), which keeps attention weights and feed-forward blocks in 4-bit integers while retaining 5-bit or 6-bit scales for sensitive normalization layers. 

Even so, 4-bit quantization only gets you so far. It reduces static weight size by roughly 75%, but a 17.2B three-model ensemble still occupies 11.5 GB. That is almost double the 6,144 MiB memory capacity of an RTX 3050 Laptop GPU. Quantization is mandatory, but on its own, it cannot solve the edge co-residency problem.

### B. Memory Offloading & Virtualized Inference Frameworks
To work around hardware limits, researchers have built tiered memory hierarchies across GPU VRAM, host DDR RAM, and NVMe SSDs. DeepSpeed ZeRO-Offload [20] introduced host offloading for distributed training, shifting optimizer states and FP16 gradients to system memory. For inference, FlexGen [22] introduced high-throughput offloading, using linear programming to schedule tensor movement across GPU, CPU, and disk. FlexGen aggregates tokens into large batches to hide the slow transfer rate of the PCIe bus behind matrix multiplications.

FlexGen falls short on interactive edge devices where batch size is strictly one ($b=1$). In single-query generation, the bottleneck is memory bandwidth, not raw compute. Every newly generated token forces the entire model weight tensor to stream across the PCIe bus. On a four-lane PCIe 3.0 interface (throughput capped at 3.94 GB/s), generation speed stalls between 0.5 and 1.5 tokens per second. Separately, vLLM [25] introduced PagedAttention, which splits key-value caches into non-contiguous physical pages to stop memory fragmentation. PagedAttention is great for multi-user servers where models stay loaded in high-bandwidth memory. But it offers no mechanism to run ensembles whose combined weights exceed physical VRAM.

### C. Compound AI Systems & Hierarchical Model Cascades
Recent work shows that splitting complex tasks into specialized models produces better accuracy and lower latency than relying on a single general-purpose model [13]. FrugalGPT [23] introduced cascade routing: cheap, lightweight models handle simple queries first, escalating to larger models only when output confidence drops below a set threshold. RouteLLM [29] formalized this with preference-trained routing classifiers.

Similarly, speculative decoding (Leviathan et al. [30], Chen et al. [31], Medusa [32]) pairs a fast draft model with a large target model. The small model drafts candidate tokens, and the large model verifies them in a single forward pass. Standard speculative decoding requires both the draft model and target model to stay loaded in GPU memory at the same time. On a 6 GB edge laptop, holding both models in VRAM is physically impossible. Without memory virtualization, classical speculative decoding cannot run on consumer silicon.

### D. Operating System Memory Virtualization & WDDM Paging Dynamics
A major practical hurdle on edge machines is the operating system's graphics driver architecture [33]. On Linux, the Direct Rendering Manager (DRM) and Kernel Mode Setting (KMS) give user applications direct, low-level control over memory allocations. But consumer laptops run overwhelmingly on Microsoft Windows, where all GPU tasks go through the Windows Display Driver Model (WDDM 3.x) [11], [21].

Under WDDM, the Video Memory Manager (VidMm) virtualizes GPU memory, treating physical VRAM as a dynamic cache backed by host system paging files. When an application asks for more memory than the card physically has, WDDM does not return an allocation error. Instead, VidMm silently evicts existing pages across the PCIe bus into host RAM. When the GPU next touches those evicted pages, a hard hardware page fault occurs, stalling execution until the pages are hauled back into VRAM. Under continuous inference, this creates a memory thrashing storm: GPU load hits 100%, but generation throughput drops by 95%. If a compute kernel stalls waiting on pages for more than 2.0 seconds, the Windows watchdog triggers a Timeout Detection and Recovery (TDR) event, terminating the application. EdgeCascade prevents this failure mode by keeping physical allocations strictly below the WDDM threshold.

```
TABLE I: ARCHITECTURAL COMPARISON OF EDGE INFERENCE SYSTEMS
```
"""
    return md
