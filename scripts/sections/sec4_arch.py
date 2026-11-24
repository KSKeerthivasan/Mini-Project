# -*- coding: utf-8 -*-
"""Section IV: EdgeCascade System Architecture & Low-Level Mechanics - Fully Humanized"""

def get_sec4():
    md = r"""## IV. EdgeCascade System Architecture & Low-Level Mechanics

EdgeCascade replaces static daemon-based model co-residency with a dynamic, phase-aware process orchestration pipeline. The architecture is structured into three discrete execution tiers operating under the governance of a centralized mutex and memory pacing coordinator.

```
FIGURE 1: EDGECASCADE ARCHITECTURE & DATAFLOW
```

### A. Tri-Stage Heterogeneous Cascade Pipeline
The division of labor across EdgeCascade's three stages exploits the physical strengths of heterogeneous hardware (GPU tensor cores versus multi-core CPU SIMD vector units):

1. *Stage 1 (Speculative Drafting)*: We instantiate Google Gemma 3 4B-it quantized at 4-bit block precision. Because the 4B parameter footprint fits comfortably within the 6 GB physical boundary ($3,105.0\text{ MiB}$), we assign an offload parameter of $ngl=99$. This maps all 36 transformer layers, attention heads, and positional embeddings directly into dedicated GPU memory. Gemma 3 delivers high autoregressive throughput ($\sim 21.5\text{ tok/s}$), writing a structured initial draft from the user's prompt and constraint set.

2. *Stage 2 (Deterministic Verification)*: Rather than dedicating scarce GPU memory to verification, we assign Google FLAN-T5 Large (770M parameters) exclusively to the host CPU ($ngl=0$). FLAN-T5 Large is a sequence-to-sequence encoder-decoder model with high empirical sensitivity to factual discrepancies, logical contradictions, and hallucinated claims. Running on 16 threads of the AMD Ryzen 7 5800H host processor with AVX2 SIMD vector extensions, FLAN-T5 audits the Stage 1 draft against ground-truth constraints without occupying a single megabyte of GPU VRAM ($0.0\text{ MiB}$). This allows Stage 2 execution to proceed concurrently with the WDDM driver unmapping of Stage 1.

3. *Stage 3 (High-Capacity Syntactic Refinement)*: The final stage deploys a specialized 12-billion parameter discourse refinement model quantized at Q4_K_M. A 12B model under 4-bit precision occupies approximately 7.2 GB of static weight memory, which exceeds physical 6 GB VRAM. To resolve this, EdgeCascade implements asymmetric layer partitioning: exactly 20 transformer layers ($ngl=20$) are offloaded to the RTX 3050 Laptop GPU, consuming $4,287.0\text{ MiB}$ of VRAM. The remaining 20 layers reside in host DDR4 memory. During inference, token embeddings and early layer activations run directly in GPU VRAM with full Tensor Core acceleration; intermediate activations are then streamed via PCIe 3.0 DMA to host CPU memory for the final layer projections. Stage 3 ingests the raw draft from Stage 1 along with the structural verification critique from Stage 2, performing an exhaustive linguistic rewrite that elevates syntactic variance, introduces sentence-length burstiness, and eliminates the stereotypical n-gram probability patterns characteristic of machine-generated text.

### B. Mutex-Guarded Lifecycle Management & IPC Protocol
In monolithic serving frameworks, background processes maintain persistent memory handles. When multiple requests trigger different models, memory allocations overlap, causing instantaneous OOM crashes. EdgeCascade eliminates this vulnerability through an operating-system-level mutual exclusion lock ($\mathcal{X}_{\text{VRAM}}$) combined with a disciplined child process lifecycle.

```
ALGORITHM 1: Mutex-Guarded Dynamic Memory-Multiplexed Inference Protocol
```

### C. Deterministic WDDM Paging & Driver Cleanup Engine
Under Microsoft Windows, terminating a process handle via `TerminateProcess()` or standard Python process closure does not guarantee immediate physical VRAM reclamation. The WDDM kernel driver (`dxgkrnl.sys`) performs deferred cleanup: memory pages allocated by the terminated process are placed on a deferred free list and unmapped asynchronously across several scheduling ticks. If a subsequent process immediately requests a large GPU allocation while pages are still queued on the deferred free list, the driver's memory budgeter registers an oversubscription condition, precipitating catastrophic page migration or outright allocation rejection.

EdgeCascade implements a proactive driver synchronization protocol, formalized in Algorithm 2. Following process termination, the coordinator interrogates the hardware counters via NVML (NVIDIA Management Library). The coordinator enforces an empirical settling window $\tau_{\text{settle}} = 1.2\text{ s}$, continuously polling the physical dedicated memory counter until resident allocation returns to the baseline resting state ($V_{\text{residual}} = 0.0\text{ MiB}$). This deterministic flush guarantees that every subsequent pipeline stage encounters an unfragmented, fully reclaimed 6,144 MiB physical memory space.

```
ALGORITHM 2: Deterministic WDDM Paging & Driver Cleanup Engine
```"""
    return md

if __name__ == '__main__':
    print(f"Sec 4 words: {len(get_sec4().split())}")
