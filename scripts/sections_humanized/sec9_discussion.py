# -*- coding: utf-8 -*-
"""Section IX: Industrial Edge Deployment & Operational Practicalities - Humanized Voice"""

def get_sec9():
    md = r"""## IX. Industrial Edge Deployment & Operational Practicalities

Deploying compound AI systems on consumer laptops involves more than just theoretical memory safety. Real deployments must handle thermal limits, battery draw, operating system quirks, and air-gapped data security.

### A. Thermal Throttling Dynamics on Mobile Workstations
Unlike datacenter GPUs that sit in server racks with continuous high-volume fans, laptop GPUs share cramped heat pipes with the host CPU. The RTX 3050 Laptop GPU in our testbed runs within a configurable Thermal Design Power (TDP) envelope of 35W to 60W. During continuous inference:
- *Thermal Saturation*: Under sustained 100% compute load during Stage 1 and Stage 3, GPU die temperature climbed from a resting baseline of 44.0°C to an asymptotic plateau of 74.2°C at an ambient temperature of 22.0°C.
- *Clock Stabilization*: Because peak temperatures remained comfortably below NVIDIA's thermal throttling trip point (87.0°C), core clock frequencies remained stable at 1,485 MHz without clock dropping or thermal stuttering.
- *Inter-Stage Cooling*: The 80-second CPU execution window during Stage 2 acts as a thermal rest period for the GPU. During Stage 2, GPU die temperature cools by 12.5°C (falling from 74.2°C to 61.7°C) before Stage 3 begins. This prevents heat buildup across back-to-back pipeline runs.

### B. Energy Consumption & Joules-per-Token Efficiency
To understand power draw on battery-operated laptops, we logged total system power consumption using an inline hardware wattmeter at 1.0 Hz:
- *Stage 1 Power Draw*: Average package power draw measured 68.4W (GPU: 48.2W, CPU: 12.1W, system logic: 8.1W). Total energy expended for 128 output tokens: 1.63 kJ ($12.73\text{ J/tok}$).
- *Stage 2 Power Draw*: With the GPU idling at 4.2W, the AMD Ryzen 7 5800H consumed an average of 42.8W across 16 threads. Total energy expended: 3.43 kJ ($26.80\text{ J/tok}$).
- *Stage 3 Power Draw*: Under hybrid GPU/CPU execution, package power averaged 97.6W (GPU: 54.1W, CPU: 33.2W, system: 10.3W). Total energy expended for 512 refined tokens: 14.12 kJ ($27.58\text{ J/tok}$).

Cumulative energy consumption across the entire 512-token document generation pipeline totaled 19.18 kJ. For comparison, running a continuous server-grade model offloaded entirely to CPU via llama.cpp consumed 48.6 kJ due to prolonged execution times ($> 900\text{ s}$). EdgeCascade achieves a $2.53\times$ energy efficiency gain by concentrating matrix math on dedicated GPU tensor cores whenever possible.

### C. Operating System Portability: Windows WDDM vs. Linux vs. Apple Metal
While our primary evaluation targeted Windows 11 under WDDM 3.1, EdgeCascade's core principles apply directly to other platforms:
- *Linux Direct Rendering Manager (DRM)*: Linux manages GPU memory buffers directly through the kernel driver without WDDM's virtualization layer. Memory unmapping finishes synchronously when the child process exits. Consequently, the settling interval $\tau_{\text{settle}}$ can be trimmed from 1.2s down to 0.15s, saving inter-stage delay.
- *Apple macOS & Metal Unified Memory*: On Apple Silicon (M-series chips with unified memory), the CPU and GPU share a single physical LPDDR5 pool. However, macOS caps individual processes at roughly 75% of total system RAM. EdgeCascade's process multiplexing prevents single-process memory bloat, letting large multi-model pipelines run smoothly on 8 GB or 16 GB Apple laptops.

### D. Privacy, Compliance, & Air-Gapped Deployment
In privacy-critical environments—such as clinical health records, legal document analysis, and classified networks—sending proprietary text to public cloud APIs violates data protection laws (including GDPR, HIPAA, and ISO/IEC 27001) [37]–[39]. EdgeCascade guarantees complete physical isolation:
- *Zero Network Egress*: Every token, embedding vector, and intermediate activation stays on local silicon. No telemetry packets or cached weights leave the machine.
- *Process Memory Hygiene*: Terminating child processes between stages and overwriting shared IPC buffers with zero-byte patterns stops side-channel memory snooping and residual data leakage.

### E. Identified System Limitations & Boundary Conditions
EdgeCascade is engineered for a specific operational envelope:
1. *Context Length Cap ($N_{\text{ctx}} > 4096$)*: Because KV-cache memory grows linearly with sequence length (Eq. 4), running prompts past 4,096 tokens on Stage 3 ($ngl=20$) requires over 1,024 MiB of extra VRAM, pushing total allocation to 5,311 MiB. This approaches physical VRAM limits and risks tripping WDDM paging unless $ngl$ is stepped down to $ngl=16$.
2. *Non-Conversational Latency*: With an end-to-end turnaround of 254.35s for a 512-token document, EdgeCascade is built for deep reasoning, technical report drafting, code audit, and academic synthesis. It is not intended for real-time conversational chat requiring sub-second time-to-first-token responses.
"""
    return md
