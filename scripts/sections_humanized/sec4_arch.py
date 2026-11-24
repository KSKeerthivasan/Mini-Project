# -*- coding: utf-8 -*-
"""Section IV: EdgeCascade System Architecture & Low-Level Mechanics - Humanized Voice"""

def get_sec4():
    md = r"""## IV. EdgeCascade System Architecture & Low-Level Mechanics

EdgeCascade replaces static background daemons with a phase-aware execution pipeline. We coordinate three distinct model stages through an operating-system mutex lock and an active memory-pacing engine.

```
FIGURE 1: EDGECASCADE ARCHITECTURE & DATAFLOW
```

### A. Tri-Stage Heterogeneous Cascade Pipeline
Our three stages divide computational labor between GPU tensor cores and multi-core CPU SIMD vector units:

1. *Stage 1 (Speculative Drafting)*: We run Google Gemma 3 4B-it quantized at 4-bit block precision. Because the 4B parameter footprint fits easily within 6 GB VRAM ($3,105.0\text{ MiB}$), we assign an offload parameter of $ngl=99$. This maps all 36 transformer layers directly into dedicated GPU memory. Gemma 3 delivers fast autoregressive throughput ($\sim 21.5\text{ tok/s}$), producing a structured initial draft from the user's prompt.

2. *Stage 2 (Deterministic Verification)*: Rather than consuming scarce GPU memory on verification, we run Google FLAN-T5 Large (770M parameters) exclusively on the host CPU ($ngl=0$). FLAN-T5 Large is an encoder-decoder model with high empirical sensitivity to factual errors, contradictions, and unsupported claims. Executing across 16 threads of the AMD Ryzen 7 5800H CPU using AVX2 vector extensions, FLAN-T5 audits the Stage 1 draft without taking up a single megabyte of GPU VRAM ($0.0\text{ MiB}$). This allows Stage 2 verification to run concurrently with the driver's unmapping of Stage 1.

3. *Stage 3 (High-Capacity Syntactic Refinement)*: The final stage runs a specialized 12-billion parameter refinement model quantized at Q4_K_M. A 12B model under 4-bit precision occupies roughly 7.2 GB of static weight memory, which exceeds physical 6 GB VRAM. To solve this, EdgeCascade splits layers unevenly: exactly 20 transformer layers ($ngl=20$) run on the RTX 3050 Laptop GPU, taking up $4,287.0\text{ MiB}$ of VRAM. The remaining 20 layers execute in host DDR4 memory. Intermediate activations move via PCIe 3.0 DMA to host memory for final layer projections. Stage 3 takes the draft from Stage 1 along with the critique from Stage 2, performing an exhaustive linguistic rewrite that elevates syntactic variety and eliminates predictable machine-generated phrasing.

### B. Mutex-Guarded Lifecycle Management & IPC Protocol
In monolithic serving systems, background processes hold persistent memory handles. When multiple requests invoke different models, allocations collide, causing immediate OOM crashes. EdgeCascade prevents this failure through an operating-system mutual exclusion lock ($\mathcal{X}_{\text{VRAM}}$) paired with strict child process lifecycles.

```
ALGORITHM 1: Mutex-Guarded Dynamic Memory-Multiplexed Inference Protocol
```

### C. Deterministic WDDM Paging & Driver Cleanup Engine
On Windows, terminating a process handle via standard Python calls does not guarantee immediate physical VRAM reclamation. The WDDM kernel driver (`dxgkrnl.sys`) relies on deferred cleanup: pages from the closed process are placed on a deferred free list and unmapped asynchronously across scheduling ticks. If a new process requests memory while old pages sit on this list, the driver's memory budgeter registers an oversubscription, causing page migration or allocation failure.

EdgeCascade resolves this with an active driver synchronization protocol, shown in Algorithm 2. Following process termination, our coordinator polls hardware counters through NVML. We enforce an empirical settling window ($\tau_{\text{settle}} = 1.2\text{ s}$), checking memory until resident allocation returns to the baseline resting state ($V_{\text{residual}} = 0.0\text{ MiB}$). This clean flush guarantees that each subsequent stage begins with an unfragmented 6,144 MiB physical memory space.

```
ALGORITHM 2: Deterministic WDDM Paging & Driver Cleanup Engine
```
"""
    return md
