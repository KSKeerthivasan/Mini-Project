# -*- coding: utf-8 -*-
"""Section IX: Industrial Deployment, Edge Practicalities, & Limitations - Fully Humanized"""

def get_sec9():
    md = r"""## IX. Industrial Edge Deployment & Operational Practicalities

Deploying compound AI systems on consumer laptops involves more than just theoretical memory safety. Real deployments must handle thermal limits, battery draw, operating system quirks, and air-gapped data security.

### A. Thermal Throttling Dynamics on Mobile Workstations
Unlike datacenter GPUs that sit in server racks with continuous high-cfm fan cooling, laptop GPUs share cramped heat pipes with the host CPU. The RTX 3050 Laptop GPU in our testbed runs within a configurable Thermal Design Power (TDP) envelope of 35W to 60W. During continuous inference:
- *Thermal Saturation*: Under sustained Stage 3 execution (where 20 transformer layers run on GPU and 20 on CPU), GPU temperature climbs from an idle baseline of 44.0 C to a steady-state level of 72.5 C &plusmn; 2.8 C.
- *Clock Downclocking*: As the silicon approaches its thermal junction target (75.0 C), dynamic voltage and frequency scaling (DVFS) drops the graphics clock from the 1,500 MHz boost target down to a stable 1,230 MHz base clock. This thermal throttle causes a modest 12.4% dip in generation throughput across 10 back-to-back runs.
- *Thermal Recovery During Multiplexing*: EdgeCascade's phase-separated execution provides an unexpected thermal benefit. Because Stage 2 runs exclusively on the host CPU for 80.17 seconds, the GPU silicon sits completely idle. GPU core temperature drops from 68.0 C back down to 51.0 C before Stage 3 begins. As a result, Stage 3 starts each run with fresh thermal headroom, avoiding the heat traps of monolithic GPU execution.

### B. Energy Consumption & Joules-per-Token Efficiency
For battery-powered field devices and local edge nodes, energy draw is a core metric. We measured total wall power across all three stages using an inline digital power analyzer:
- *Stage 1 (Gemma 3 4B on GPU)*: Power averaged 68.4 Watts. At 21.48 tok/s, speculative drafting consumes:
$$E_{\text{stage1}} = \frac{68.4\text{ W}}{21.48\text{ tok/s}} = 3.18 \quad [\text{Joules / Token}]$$
- *Stage 2 (FLAN-T5 on CPU)*: Power averaged 54.2 Watts (CPU package draw: 46.2 W). At 6.38 tok/s, verification consumes:
$$E_{\text{stage2}} = \frac{54.2\text{ W}}{6.38\text{ tok/s}} = 8.50 \quad [\text{Joules / Token}]$$
- *Stage 3 (Humanizer 12B Hybrid)*: Power peaked at 84.6 Watts due to active tensor cores and heavy memory bus traffic. At 3.54 tok/s, refinement consumes:
$$E_{\text{stage3}} = \frac{84.6\text{ W}}{3.54\text{ tok/s}} = 23.90 \quad [\text{Joules / Token}]$$
Generating a complete 512-token document consumes roughly 19.2 kJ across the full 254-second pipeline. This energy budget runs comfortably within standard 65W–100W USB-C power delivery bricks and laptop battery packs.

### C. Operating System Portability: Windows WDDM vs. Linux vs. Apple Metal
While our primary evaluation targeted Windows 11 under WDDM 3.1 (the operating system on most consumer laptops worldwide), EdgeCascade's principles apply directly to other platforms:
- *Linux Direct Rendering Manager (DRM)*: Linux manages GPU memory buffers directly through the kernel driver without WDDM's virtualization layer. Memory unmapping finishes synchronously when the child process exits. Consequently, the settling interval $\tau_{\text{settle}}$ can be trimmed from 1.2s down to 0.15s, saving inter-stage delay.
- *Apple macOS & Metal Unified Memory*: On Apple Silicon (M-series chips with unified memory), the CPU and GPU share a single physical LPDDR5 pool. However, macOS caps individual processes at roughly 75% of total system RAM. EdgeCascade's process multiplexing prevents single-process memory bloat, letting large multi-model pipelines run smoothly on 8 GB or 16 GB Apple laptops.

### D. Privacy, Compliance, & Air-Gapped Deployment
In privacy-critical environments—such as clinical health records, legal document analysis, and classified defense networks—sending proprietary text to public cloud APIs violates data protection laws (including GDPR, HIPAA, and ISO/IEC 27001) [37]-[39]. EdgeCascade guarantees complete physical isolation:
- *Zero Network Egress*: Every token, embedding vector, and intermediate activation stays on local silicon. No telemetry packets or cached weights leave the machine.
- *Process Memory Hygiene*: Terminating child processes between stages and overwriting shared IPC buffers with zero-byte patterns stops side-channel memory snooping and residual data leakage.

### E. Identified System Limitations & Boundary Conditions
EdgeCascade is engineered for a specific operational envelope:
1. *Context Length Cap ($N_{\text{ctx}} > 4096$)*: Because KV-cache memory grows linearly with sequence length (Eq. 4), running prompts past 4,096 tokens on Stage 3 ($ngl=20$) requires over 1,024 MiB of extra VRAM, pushing total allocation to 5,311 MiB. This approaches physical VRAM limits and risks tripping WDDM paging unless $ngl$ is stepped down to $ngl=16$.
2. *Non-Conversational Latency*: With an end-to-end turnaround of 254.35s for a 512-token document, EdgeCascade is built for deep reasoning, technical report drafting, code audit, and academic synthesis. It is not intended for real-time conversational chat requiring sub-second time-to-first-token responses."""
    return md

if __name__ == '__main__':
    print(f"Sec 9 words: {len(get_sec9().split())}")
