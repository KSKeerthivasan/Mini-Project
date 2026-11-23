# -*- coding: utf-8 -*-
"""Section IV: EdgeCascade System Architecture & Low-Level Mechanics"""

def get_sec4():
    md = r"""## IV. EdgeCascade System Architecture & Low-Level Mechanics

EdgeCascade replaces static daemon-based model co-residency with a dynamic, phase-aware process orchestration pipeline. The architecture is structured into three discrete execution tiers operating under the governance of a centralized mutex and memory pacing coordinator.

```
+--------------------------------------------------------------------------------------------------------------------+
|                                      FIGURE 1: EDGECASCADE ARCHITECTURE & DATAFLOW                                 |
+--------------------------------------------------------------------------------------------------------------------+
|                                                                                                                    |
|   +------------------------------------------------------------------------------------------------------------+   |
|   |                                          USER PROMPT INGESTION & PIPELINE CONFIG                           |   |
|   +------------------------------------------------------------------------------------------------------------+   |
|                                                         |                                                          |
|                                                         v                                                          |
|   +------------------------------------------------------------------------------------------------------------+   |
|   | STAGE 1: Fast Speculative Drafting Engine                                                                  |   |
|   | Model: Google Gemma 3 4B-it (GGUF Q4_K_M) | Execution Target: GPU Dedicated (ngl=99)                      |   |
|   | VRAM Footprint: 3,105.0 MiB               | Host RAM Footprint: Negligible (Weights fully on GPU)          |   |
|   | Role: Rapid context hypothesis generation, logical structure drafting, and preliminary argument layout.   |   |
|   +------------------------------------------------------------------------------------------------------------+   |
|                                                         |                                                          |
|                                                         v                                                          |
|                       [WDDM Teardown Protocol: Terminate PID -> tau_settle = 1.2s -> VRAM = 0.0 MiB]               |
|                                                         |                                                          |
|                                                         v                                                          |
|   +------------------------------------------------------------------------------------------------------------+   |
|   | STAGE 2: Deterministic Structural Verification & Logical Audit Engine                                      |   |
|   | Model: Google FLAN-T5 Large 770M (Seq2Seq)| Execution Target: Host CPU (ngl=0, 16 AVX-512 Threads)         |   |
|   | VRAM Footprint: 0.0 MiB                   | Host RAM Footprint: 1,620.0 MiB                                |   |
|   | Role: Factual correctness auditing, contradiction detection, citation validation, and constraint checking.  |   |
|   +------------------------------------------------------------------------------------------------------------+   |
|                                                         |                                                          |
|                                                         v                                                          |
|                       [Intermediate Payload Assembly: Structured JSON Payload with Verification Critiques]        |
|                                                         |                                                          |
|                                                         v                                                          |
|   +------------------------------------------------------------------------------------------------------------+   |
|   | STAGE 3: High-Capacity Syntactic Refinement & Humanization Engine                                          |   |
|   | Model: Humanizer 12B (GGUF Q4_K_M)        | Execution Target: Hybrid GPU/CPU (ngl=20 GPU, 20 CPU Layers)   |   |
|   | VRAM Footprint: 4,287.0 MiB (Peak)        | Host RAM Footprint: 4,850.0 MiB                                |   |
|   | Role: Re-synthesis of verified content into natural, bursty, publication-grade academic discourse.         |   |
|   +------------------------------------------------------------------------------------------------------------+   |
|                                                         |                                                          |
|                                                         v                                                          |
|   +------------------------------------------------------------------------------------------------------------+   |
|   | FINAL DELIVERABLE: Publication-Grade IEEE Manuscript (Factual, Fluent, AI-Detector Evasive)                |   |
|   +------------------------------------------------------------------------------------------------------------+   |
+--------------------------------------------------------------------------------------------------------------------+
```

### A. Tri-Stage Heterogeneous Cascade Pipeline
The cognitive division of labor across EdgeCascade's three stages is customized to exploit the hardware strengths of heterogeneous compute units (GPU tensor cores vs. multi-core CPU SIMD vectors):

1. *Stage 1 (Speculative Drafting)*: We instantiate Google Gemma 3 4B-it quantized at 4-bit block precision. Because the 4B parameter footprint fits comfortably within the 6 GB physical boundary ($3,105.0\text{ MiB}$), we assign an offload parameter of $ngl=99$. This forces all 36 transformer layers, attention heads, and positional embeddings to reside in dedicated GPU memory. Gemma 3 operates with high autoregressive throughput ($\sim 21.5\text{ tok/s}$), synthesizing a comprehensive draft response from the user's initial prompt and contextual constraints.

2. *Stage 2 (Deterministic Verification)*: Rather than dedicating scarce GPU memory to verification, we assign Google FLAN-T5 Large (770M parameters) exclusively to the host CPU ($ngl=0$). FLAN-T5 Large represents a mature, sequence-to-sequence encoder-decoder architecture that demonstrates high empirical sensitivity to factual discrepancies, logical contradictions, and hallucinated assertions. Executing on 16 threads of the AMD Ryzen 7 5800H host processor utilizing AVX2 SIMD vector extensions, FLAN-T5 evaluates the Stage 1 draft against ground-truth constraints without occupying a single megabyte of GPU VRAM ($0.0\text{ MiB}$). This allows Stage 2 execution to proceed concurrently with the WDDM driver unmapping of Stage 1.

3. *Stage 3 (High-Capacity Syntactic Refinement)*: The final stage deploys a specialized 12-billion parameter humanization and discourse refinement model quantized at `Q4_K_M`. A 12B model under 4-bit precision occupies approximately 7.2 GB of static weight memory, which exceeds physical 6 GB VRAM. To resolve this, EdgeCascade implements asymmetric layer partitioning: exactly 20 transformer layers ($ngl=20$) are offloaded to the RTX 3050 Laptop GPU, consuming $4,287.0\text{ MiB}$ of VRAM. The remaining 20 layers reside in host DDR4 memory. During inference, token embeddings and early layer activations are processed directly in GPU VRAM with full Tensor Core acceleration; intermediate activations are then streamed via PCIe 3.0 DMA to host CPU memory for the final layer projections. Stage 3 ingests the raw draft from Stage 1 along with the structural verification critique from Stage 2, performing an exhaustive linguistic rewrite that elevates syntactic variance, introduces sentence-length burstiness, and eliminates the stereotypical n-gram probability artifacts characteristic of machine-generated text.

### B. Mutex-Guarded Lifecycle Management & IPC Protocol
In monolithic serving frameworks, background processes maintain persistent memory handles. When multiple requests trigger different models, memory allocations overlap, causing instantaneous OOM crashes. EdgeCascade eliminates this vulnerability through an operating-system-level mutual exclusion lock ($\mathcal{X}_{\text{VRAM}}$) combined with a disciplined child process lifecycle.

```
+--------------------------------------------------------------------------------------------------------------------+
| ALGORITHM 1: Mutex-Guarded Dynamic Memory-Multiplexed Inference Protocol                                           |
+--------------------------------------------------------------------------------------------------------------------+
|  1: Input: User Query Q, Pipeline Configuration C = {(m_1, ngl_1), (m_2, ngl_2), (m_3, ngl_3)}                    |
|  2: Output: Final Publication-Grade Artifact A_final                                                              |
|  3: Initialize Global Named Mutex X_VRAM <- AcquireMutexHandle("Global\\EdgeCascade_VRAM_Lock")                    |
|  4: Initialize IPC Shared Memory Buffer S_shared <- AllocateSharedMemory(Capacity = 64 MiB)                       |
|  5:                                                                                                                |
|  6: // --- STAGE 1: SPECULATIVE DRAFTING ---                                                                       |
|  7: WaitAndAcquire(X_VRAM)                                                                                         |
|  8: Telemetry_Log("Stage 1 Pre-Allocation", QueryNVMLMemory())                                                     |
|  9: PID_1 <- SpawnChildProcess("llama-cli", Model = m_1, Layers = ngl_1, Prompt = Q, Output = S_shared)            |
| 10: WaitForProcessExit(PID_1, Timeout = 120.0s)                                                                    |
| 11: Draft_Payload <- ReadSharedMemory(S_shared)                                                                    |
| 12: Execute_Deterministic_Teardown(PID_1, tau_settle = 1.2s)                                                       |
| 13: ReleaseMutex(X_VRAM)                                                                                           |
| 14:                                                                                                                |
| 15: // --- STAGE 2: CPU DETERMINISTIC VERIFICATION ---                                                             |
| 16: // Pinned to CPU: Mutex X_VRAM is NOT required                                                                 |
| 17: Telemetry_Log("Stage 2 CPU Execution", QueryNVMLMemory())                                                      |
| 18: PID_2 <- SpawnChildProcess("flan-t5-engine", Model = m_2, Device = "CPU", Input = Draft_Payload)               |
| 19: WaitForProcessExit(PID_2, Timeout = 180.0s)                                                                    |
| 20: Verified_Payload <- ReadProcessOutput(PID_2)                                                                   |
| 21: TerminateProcess(PID_2)                                                                                        |
| 22:                                                                                                                |
| 23: // --- STAGE 3: HYBRID SYNTACTIC REFINEMENT ---                                                                |
| 24: WaitAndAcquire(X_VRAM)                                                                                         |
| 25: Telemetry_Log("Stage 3 Pre-Allocation", QueryNVMLMemory())                                                     |
| 26: Combined_Prompt <- AssembleRefinementPrompt(Draft_Payload, Verified_Payload)                                   |
| 27: PID_3 <- SpawnChildProcess("llama-cli", Model = m_3, Layers = ngl_3, Prompt = Combined_Prompt)                 |
| 28: WaitForProcessExit(PID_3, Timeout = 300.0s)                                                                    |
| 29: A_final <- ReadProcessOutput(PID_3)                                                                            |
| 30: Execute_Deterministic_Teardown(PID_3, tau_settle = 1.2s)                                                       |
| 31: ReleaseMutex(X_VRAM)                                                                                           |
| 32: Return A_final                                                                                                 |
+--------------------------------------------------------------------------------------------------------------------+
```

### C. Deterministic WDDM Paging & Driver Cleanup Engine
Under Microsoft Windows, terminating a process handle via `TerminateProcess()` or standard Python process closure does not guarantee immediate physical VRAM reclamation. The WDDM kernel driver (`dxgkrnl.sys`) performs deferred cleanup: memory pages allocated by the terminated process are placed on a deferred free list and unmapped asynchronously across several scheduling ticks. If a subsequent process immediately requests a large GPU allocation while pages are still queued on the deferred free list, the driver's memory budgeter registers an oversubscription condition, precipitating catastrophic page migration or outright allocation rejection.

EdgeCascade implements a proactive driver synchronization protocol, formalized in Algorithm 2. Following process termination, the coordinator interrogates the hardware counters via NVML (NVIDIA Management Library). The coordinator enforces an empirical settling window $\tau_{\text{settle}} = 1.2\text{ s}$, continuously polling the physical dedicated memory counter until resident allocation returns to the baseline resting state ($V_{\text{residual}} = 0.0\text{ MiB}$). This deterministic flush guarantees that every subsequent pipeline stage encounters an unfragmented, fully reclaimed 6,144 MiB physical memory space.

```
+--------------------------------------------------------------------------------------------------------------------+
| ALGORITHM 2: Deterministic WDDM Paging & Driver Cleanup Engine                                                     |
+--------------------------------------------------------------------------------------------------------------------+
|  1: Input: Process ID PID, Target Idle VRAM V_idle_target = 0.0 MiB, Settling Window tau_settle = 1.2s             |
|  2: Output: Boolean Success_Status                                                                                 |
|  3: Function Execute_Deterministic_Teardown(PID, tau_settle):                                                      |
|  4:    hProcess <- OpenProcess(PROCESS_TERMINATE | SYNCHRONIZE, FALSE, PID)                                        |
|  5:    If hProcess != NULL Then:                                                                                   |
|  6:       TerminateProcess(hProcess, 0)                                                                            |
|  7:       WaitForSingleObject(hProcess, INFINITE)                                                                  |
|  8:       CloseHandle(hProcess)                                                                                    |
|  9:    End If                                                                                                      |
| 10:    t_start <- GetSystemTimePrecise()                                                                           |
| 11:    While (GetSystemTimePrecise() - t_start) < tau_settle Do:                                                   |
| 12:       Current_VRAM <- NVML_QueryDeviceMemory(DeviceIndex = 0).used                                            |
| 13:       If Current_VRAM <= V_idle_target Then:                                                                   |
| 14:          Break // Memory successfully unmapped by WDDM driver                                                  |
| 15:       End If                                                                                                   |
| 16:       Sleep(50 Milliseconds) // Yield CPU scheduling quantum                                                   |
| 17:    End While                                                                                                   |
| 18:    Final_VRAM <- NVML_QueryDeviceMemory(DeviceIndex = 0).used                                                  |
| 19:    If Final_VRAM > V_idle_target Then:                                                                         |
| 20:       InvokeWin32API_EmptyWorkingSet(GetCurrentProcess())                                                      |
| 21:       Sleep(200 Milliseconds) // Enforce hard settling barrier                                                 |
| 22:    End If                                                                                                      |
| 23:    Return (NVML_QueryDeviceMemory(DeviceIndex = 0).used <= V_idle_target)                                       |
+--------------------------------------------------------------------------------------------------------------------+
```"""
    return md

if __name__ == '__main__':
    print(f"Sec 4 word count: {len(get_sec4().split())}")
