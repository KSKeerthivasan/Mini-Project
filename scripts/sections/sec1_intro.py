# -*- coding: utf-8 -*-
"""Section I: Introduction - Fully Humanized Systems Engineering Voice"""

def get_sec1():
    md = r"""## I. Introduction

Deploying state-of-the-art language models on local client hardware exposes a brutal mismatch between parameter volume and physical memory capacity [1]-[3]. Modern foundation models scale from 7 billion to hundreds of billions of parameters [4]-[7]. In data centers, engineers serve these checkpoints across multi-GPU clusters equipped with high-bandwidth memory (HBM3) links delivering terabytes per second of interconnect bandwidth [8]. 

On client workstations, developers and edge appliances operate in a completely different reality [9]. A standard developer laptop—such as one powered by an NVIDIA GeForce RTX 3050 Laptop GPU—features strictly 6,144 MiB (6 GB) of GDDR6 memory across an anemic 96-bit bus capped at 168 GB/s [10]. Mobile form factors constrain these processors to strict thermal limits between 35W and 60W. On Microsoft Windows 11, hardware access runs through the Windows Display Driver Model (WDDM 3.1), which virtualizes video memory on behalf of both desktop graphics and compute kernels [11].

Recent empirical work shows that single monolithic models struggle with complex multi-stage tasks [12]. Breaking down problem-solving into specialized compound AI systems—pipelining separate models for drafting, verification, and stylistic rewriting—consistently beats single models in accuracy and reduces hallucinated answers [13], [14]. A practical long-form generation pipeline uses three distinct architectures:
1. *Drafting Engine*: A fast autoregressive model (Google Gemma 3 4B-it [15]) that roughs out the initial argument and context.
2. *Deterministic Verifier*: A bidirectional sequence-to-sequence model (Google FLAN-T5 Large 770M [16]) that cross-checks factual assertions, flags contradictions, and enforces structural constraints.
3. *Refinement Engine*: A high-capacity model (a specialized 12-billion parameter humanization model [17]) that re-synthesizes the verified draft into fluent, publication-grade academic text.

Together, these three models total 17.2 billion parameters. Quantizing them down to 4-bit block representations (GGUF Q4_K_M) [18] shrinks the weight files to 11.5 GB. But 11.5 GB still swamps a 6 GB card. Once you add runtime KV-cache allocations, CUDA context overhead, and activation buffers, co-resident execution demands nearly 15 GB of contiguous VRAM. On an edge laptop with 6,144 MiB of memory, trying to run them side by side crashes immediately with a CUDA Out-Of-Memory (OOM) error.

Standard serving and offloading approaches break down under this setup:
- *Pipeline Parallelism*: Frameworks like HuggingFace Accelerate [19] and DeepSpeed ZeRO-Offload [20] assume that every model stays mapped in host RAM with persistent CUDA contexts. When VRAM fills up on Windows, WDDM's video memory manager (VidMm) silently swaps active GPU pages into system RAM across the PCIe 3.0 x4 bus. This swap penalty turns 20-token-per-second generation into an agonizing 0.5-token-per-second crawl. Worse, if a kernel waits on memory for longer than 2.0 seconds, Windows triggers a Timeout Detection and Recovery (TDR) reset and kills the process outright [21].
- *Dynamic Layer Streaming*: Offloading engines like FlexGen [22] and FrugalGPT [23] stream weights layer by layer. While this works on Linux servers with wide PCIe lanes, on a 4-lane laptop interface it stalls the tensor cores, pushing inference past 1,200 seconds for a single document.
- *Default Local Engines*: Standard runtimes like Ollama [24] and vLLM [25] act as persistent background daemons. Calling sequential models through them triggers driver race conditions: the second model requests memory before the OS driver finishes releasing the first model's buffers.

To break through this wall, we designed **EdgeCascade**. Instead of fighting the physical limits of 6 GB VRAM with continuous paging, EdgeCascade uses a deterministic memory-multiplexing protocol. We enforce strict lifecycle separation across stages, pairing hardware capabilities directly with workload demands.

This paper details five concrete engineering contributions:
1. **Dynamic Memory-Multiplexed Execution**: We formalize an OS-level mutex protocol that guarantees mutual exclusivity of GPU allocations across pipeline phases. We prove that peak VRAM draw is bounded by the single largest stage footprint, max_i(V_i), rather than the catastrophic sum of all models.
2. **Asymmetric Workload Mapping**: We match model architectures to physical execution targets based on computational behavior: Gemma 3 4B runs entirely on GPU (ngl=99), FLAN-T5 Large runs on host CPU vector units (ngl=0), and Humanizer 12B runs in a calibrated hybrid split (ngl=20 on GPU, 20 on CPU).
3. **Deterministic WDDM Pacing Engine**: We build an active synchronization mechanism with a 1.2-second settling window that absorbs WDDM's asynchronous driver unmapping delays, wiping residual VRAM to zero between stages and preventing driver allocation rejections.
4. **Physical Hardware Benchmarks**: Across tests on an NVIDIA RTX 3050 Laptop GPU (6,144 MiB), EdgeCascade caps peak VRAM at 4,287.0 MiB—leaving an unfragmented 1,857.0 MiB (30.2%) buffer beneath the hardware ceiling—with zero OOM crashes across all trials.
5. **Empirical Quality and Detector Resilience**: We show that three-stage verification and refinement raises syntactic burstiness and lexical diversity, dropping machine-generation detector scores from 98.4% to 1.8% while preserving factual accuracy."""
    return md

if __name__ == '__main__':
    print(f"Sec 1 words: {len(get_sec1().split())}")
