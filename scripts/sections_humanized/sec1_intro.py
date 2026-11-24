# -*- coding: utf-8 -*-
"""Section I: Introduction - Fully Humanized Systems Engineering Voice"""

def get_sec1():
    md = r"""## I. Introduction

Deploying large language models on personal computers reveals a stark gap between model size and physical video memory [1]–[3]. Today's foundation models span from 7 billion to hundreds of billions of parameters [4]–[7]. Cloud data centers serve these models across multi-GPU servers using high-bandwidth memory (HBM3) links that deliver terabytes per second of interconnect speed [8]. 

Client workstations tell a very different story [9]. A standard developer laptop—such as one with an NVIDIA GeForce RTX 3050 Laptop GPU—features strictly 6,144 MiB (6 GB) of GDDR6 memory on a 96-bit bus capped at 168 GB/s [10]. Mobile laptops also restrict these processors to tight power limits between 35W and 60W. On Windows 11, the graphics driver (WDDM 3.1) manages all memory access, virtualizing VRAM across both display rendering and compute tasks [11].

Recent studies show that single monolithic models struggle when handling complex multi-stage work [12]. Pipelining specialized models—dividing work into separate drafting, verification, and stylistic rewriting stages—consistently beats single models in factual accuracy while cutting down hallucinations [13], [14]. A practical long-form generation pipeline uses three distinct architectures:
1. *Drafting Engine*: A fast autoregressive model (Google Gemma 3 4B-it [15]) that drafts the initial argument and context.
2. *Deterministic Verifier*: A bidirectional sequence-to-sequence model (Google FLAN-T5 Large 770M [16]) that audits factual claims, flags contradictions, and enforces structural rules.
3. *Refinement Engine*: A high-capacity model (a specialized 12-billion parameter humanization model [17]) that rewrites the verified draft into polished, publication-grade academic text.

Together, these three models total 17.2 billion parameters. Quantizing them down to 4-bit block formats (GGUF Q4_K_M) [18] shrinks the weight files to 11.5 GB. But 11.5 GB still overwhelms a 6 GB graphics card. Once you add runtime KV-cache allocations, CUDA context overhead, and temporary scratch tensors, holding all three models in memory at once requires roughly 13.8 GB of VRAM. Attempting to run them simultaneously on a consumer laptop triggers immediate failure:
- *Co-Residency Out-Of-Memory (OOM)*: Standard serving frameworks like HuggingFace Accelerate [19] or DeepSpeed-Offload [20] allocate all process weights at startup. When allocations pass 6,144 MiB, the CUDA driver aborts with an unrecoverable out-of-memory error.
- *WDDM Driver Paging Thrashing*: When an application oversubscribes memory, the Windows kernel driver (VidMm) evicts active pages across the PCIe bus into host RAM. Under continuous inference, this triggers severe page thrashing. GPU compute utilization collapses, and token generation slows by up to 95%. Worse, if a GPU compute kernel stalls waiting for memory for longer than 2.0 seconds, Windows triggers a Timeout Detection and Recovery (TDR) reset and kills the process outright [21].
- *Layer-by-Layer Streaming*: Offloading engines like FlexGen [22] and FrugalGPT [23] stream weights layer by layer. While this approach works well on Linux servers with wide PCIe slots, on a 4-lane mobile interface it stalls tensor cores, pushing inference past 1,200 seconds for a single document.
- *Background Daemon Engines*: Standard runtimes like Ollama [24] and vLLM [25] run as persistent background daemons. Calling sequential models through them triggers driver race conditions: the second model requests memory before the OS driver finishes releasing buffers from the first.

To break through this wall, we designed **EdgeCascade**. Instead of fighting 6 GB physical limits through continuous memory swapping, EdgeCascade coordinates execution with a predictable memory-multiplexing protocol. We enforce strict lifecycle boundaries across pipeline stages, matching each task to the right hardware engine.

We make five concrete engineering contributions:
1. **Dynamic Memory-Multiplexed Execution**: We formalize an OS-level mutex protocol that guarantees mutual exclusivity of GPU allocations across pipeline phases. We prove that peak VRAM draw is bounded by the single largest stage footprint, $\max_i(V_i)$, rather than the catastrophic sum of all models.
2. **Asymmetric Workload Mapping**: We match model architectures to physical execution targets based on computational behavior: Gemma 3 4B runs entirely on GPU ($ngl=99$), FLAN-T5 Large runs on host CPU vector units ($ngl=0$), and Humanizer 12B runs in a calibrated hybrid split ($ngl=20$ on GPU, 20 on CPU).
3. **Deterministic WDDM Pacing Engine**: We build an active synchronization mechanism with a 1.2-second settling window that absorbs WDDM's asynchronous driver unmapping delays, wiping residual VRAM to zero between stages and preventing driver allocation rejections.
4. **Physical Hardware Benchmarks**: Across tests on an NVIDIA RTX 3050 Laptop GPU (6,144 MiB), EdgeCascade caps peak VRAM at 4,287.0 MiB—leaving an unfragmented 1,857.0 MiB (30.2%) buffer beneath the hardware ceiling—with zero OOM crashes across all trials.
5. **Empirical Quality and Detector Resilience**: We show that three-stage verification and refinement raises syntactic burstiness and lexical diversity, dropping machine-generation detector scores from 98.4% to 1.8% while preserving factual accuracy.
"""
    return md
