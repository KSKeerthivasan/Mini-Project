# -*- coding: utf-8 -*-
"""Section II: Related Work & Literature Review"""

def get_sec2():
    md = r"""## II. Related Work & Taxonomic Foundation

The challenge of executing large language models under stringent hardware constraints intersects several foundational domains: model compression, memory offloading, compound AI routing, and operating system GPU virtualization. We synthesize prior art across these four pillars and contrast existing paradigms in Table I.

### A. Post-Training Quantization & Compressed Representations
Weight quantization represents the primary mechanism for compressing neural network memory footprints. Early post-training quantization (PTQ) techniques, such as 8-bit integer quantization (LLM.int8() [26]), demonstrated that vector representations could be reduced from 16-bit floating-point (FP16/BF16) to INT8 with negligible perplexity degradation, preserving outlier features through isolated high-precision matrix multiplication. Subsequent advancements pushed weight representations into the sub-4-bit regime. GPTQ [27] leveraged approximate second-order Hessian information to quantize transformer layers sequentially while minimizing layer-wise output reconstruction error. AWQ (Activation-aware Weight Quantization) [28] observed that protection of salient weight channels based on activation magnitudes preserves generation fidelity without requiring expensive backpropagation or fine-tuning.

In local edge environments, the GGUF binary format introduced by the GGML / llama.cpp project [18] has become the dominant runtime abstraction. GGUF implements structured block quantization (k-quants), combining different quantization bit-widths across attention projections and feed-forward networks (FFNs). In our work, we utilize 4-bit medium block quantization (`Q4_K_M`), which maintains 4-bit representations for attention weights and critical MLP projections while selectively retaining 5-bit or 6-bit precision for high-variance normalization scales. Although 4-bit quantization reduces static parameter memory by approximately $75\%$, deploying a 17.2B aggregate ensemble still requires 11.5 GB of storage, exceeding a 6 GB VRAM envelope by over $91\%$. Consequently, quantization alone is necessary but strictly insufficient for multi-model edge orchestration.

### B. Memory Offloading & Virtualized Inference Frameworks
To overcome physical VRAM limitations, systems researchers have explored tiered memory hierarchies spanning GPU VRAM, host DDR RAM, and non-volatile SSD storage. DeepSpeed ZeRO-Offload [20] introduced memory offloading for distributed training, partitioning optimizer states and FP16 gradients to host memory. For inference workloads, FlexGen [22] pioneered throughput-oriented offloading, utilizing linear programming to formulate optimal tensor placement schedules across GPU, CPU, and NVMe disks. FlexGen aggregates tokens into large batches to amortize the high latency of PCIe weight transfers over multiple tokens.

However, FlexGen's computational model exhibits catastrophic latency degradation in single-query, latency-sensitive edge scenarios. In interactive edge environments (batch size $b=1$), the arithmetic intensity of autoregressive generation is bounded by memory bandwidth rather than compute throughput. Every newly generated token requires the full weight tensor of the model to be streamed from host memory over the PCIe bus, capping generation speed at 0.5 to 1.5 tokens per second on PCIe 3.0 x4 interfaces (bandwidth $\approx 3.94\text{ GB/s}$). Furthermore, vLLM [25] introduced PagedAttention, which virtualizes the key-value (KV) cache into non-contiguous physical memory blocks to eliminate internal fragmentation. While PagedAttention maximizes GPU memory utilization in multi-tenant cloud environments with large concurrent batches, it presupposes that the model weights reside permanently in high-bandwidth GPU memory, providing no architectural mechanism for running ensembles whose aggregate weights exceed physical VRAM.

### C. Compound AI Systems & Hierarchical Model Cascades
Recent architectural literature demonstrates that decomposing complex cognitive workflows into specialized compound AI systems yields Pareto-optimal trade-offs between computational cost, inference latency, and task accuracy [13]. FrugalGPT [23] established the paradigm of LLM cascading, wherein incoming user queries are initially routed to low-cost, lightweight foundation models; only queries whose generation confidence falls below an empirical threshold are escalated to expensive frontier models. RouteLLM [29] formalized this routing mechanism using preference-trained classifiers and matrix factorization.

Concurrently, speculative decoding architectures (e.g., Leviathan et al. [30], Chen et al. [31], and Medusa [32]) pair a small draft model with a large target model. The draft model rapidly generates speculative token sequences which are subsequently validated in a single forward pass by the target model. While highly effective, traditional speculative decoding mandates that both draft and target models remain concurrently loaded in GPU memory to support rapid token verification handshakes. In consumer edge environments with 6 GB VRAM, concurrent co-residency of draft and target models is physically impossible, rendering classical speculative decoding inapplicable without specialized memory virtualization.

### D. Operating System Memory Virtualization & WDDM Paging Dynamics
A critical yet frequently overlooked dimension in edge deep learning systems is the interaction between the runtime inference engine and the host operating system's graphics kernel [33]. In Linux environments, the Direct Rendering Manager (DRM) and Kernel Mode Setting (KMS) provide granular, low-level control over memory allocations via unified memory buffers and IOCTL calls. In contrast, the consumer personal computing ecosystem is overwhelmingly dominated by Microsoft Windows, where GPU access is mediated by the Windows Display Driver Model (WDDM 3.x) [11], [21].

Under WDDM, the Video Memory Manager (`VidMm`) virtualizes GPU memory, treating physical VRAM as a dynamic cache backed by host system paging files. When a user process requests a GPU memory allocation that exceeds available physical VRAM, WDDM does not immediately return an out-of-memory error. Instead, `VidMm` evicts existing memory allocations to system RAM via asynchronous direct memory access (DMA) transfers across the PCIe bus. When the evicted memory pages are subsequently referenced by a GPU execution packet, a hard GPU page fault occurs, stalling the graphics execution engine until the pages are swapped back into VRAM. Under continuous inference workloads, this dynamic paging induces catastrophic memory thrashing, driving GPU utilization to 100% while execution throughput collapses by $95\%$. Furthermore, if an execution packet is stalled by page fault servicing for longer than the operating system timeout threshold (typically 2.0 seconds), the WDDM watchdog mechanism triggers a Timeout Detection and Recovery (TDR) event, forcibly resetting the GPU driver and terminating the calling process. EdgeCascade addresses this vulnerability directly through proactive, deterministic driver synchronization.

```
+------------------------------------------------------------------------------------------------------------------------+
|                                  TABLE I: ARCHITECTURAL COMPARISON OF EDGE INFERENCE PARADIGMS                         |
+----------------------+--------------------+--------------------+--------------------+--------------------+-------------+
| System Architecture  | Target Hardware    | Max Concurrent     | Memory Allocation  | Multi-Model        | OOM Safety  |
|                      | Memory Tier        | Parameter Capacity | Strategy           | Co-residency       | Guarantee   |
+----------------------+--------------------+--------------------+--------------------+--------------------+-------------+
| vLLM [25]            | Server GPU (>=24GB)| Static VRAM Bound  | PagedAttention     | Co-resident Only   | None (Hard) |
| FlexGen [22]         | Server / Cloud Node| 30B - 175B Params  | Batch Weight Stream| Co-resident Paged  | Thrashing   |
| DeepSpeed-Offload [20| Server Node (HPC)  | 13B - 70B Params   | Static NVMe / Host | Co-resident Split  | Paging Lag  |
| HuggingFace Acc. [19]| Edge / Workstation | 7B - 13B Params    | Virtual Host Swap  | Unified Paging     | TDR Crash   |
| Standard Ollama [24] | Edge Workstation   | Single 7B Model    | Monolithic Load    | Sequential Naive   | Driver Race |
| EdgeCascade (Ours)   | Consumer GPU (6GB) | 17.2B Aggregate    | Mutex Multiplexing | Orthogonal Paged   | Zero OOM    |
+----------------------+--------------------+--------------------+--------------------+--------------------+-------------+
```"""
    return md

if __name__ == '__main__':
    print(f"Sec 2 word count: {len(get_sec2().split())}")
