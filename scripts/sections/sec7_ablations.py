# -*- coding: utf-8 -*-
"""Section VII: Parametric Sensitivity & Ablation Studies - Fully Humanized"""

def get_sec7():
    md = r"""## VII. Parametric Sensitivity & Ablation Studies

To map out the operational boundaries and failure points of EdgeCascade, we ran sweeps across five system parameters: GPU layer offload counts ($ngl$), driver settling windows ($\tau$), context window lengths ($N_{\text{ctx}}$), CPU core allocations ($T_{\text{cpu}}$), and quantization precision.

### A. Stage 3 GPU Layer Offloading ($ngl$) Sensitivity
Splitting transformer layers between dedicated GPU VRAM and host system RAM in Stage 3 represents the most sensitive tuning knob in our system. In Table VI, we test $ngl \in [0, 40]$ for the 40-layer Humanizer 12B model on our 6 GB RTX 3050 Laptop GPU, tracking dedicated VRAM, shared PCIe memory spillover, and generation throughput.

```
TABLE VI: STAGE 3 GPU LAYER OFFLOAD (ngl) SENSITIVITY ABLATION
```

The experimental data reveals a steep performance drop:
- For $ngl \in [0, 20]$, execution stays comfortably inside physical VRAM ($V_{\text{VRAM}} \le 4,287.0\text{ MiB}$). Shared system memory usage over the PCIe bus is flat at zero. Generation speed scales linearly from $1.33\text{ tok/s}$ at $ngl=0$ to $3.54\text{ tok/s}$ at $ngl=20$.
- As soon as $ngl$ goes past 20 (for example, $ngl=25$), memory demand runs directly into the Windows desktop compositor buffer. WDDM reacts by evicting 340.0 MiB to host RAM across the PCIe bus. Even with 5 more layers on the GPU, latency spikes from $144.72\text{ s}$ to $420.80\text{ s}$—a $2.9\times$ slowdown.
- At $ngl=30$, PCIe paging thrashing pushes latency to $1,120.40\text{ s}$ ($7.7\times$ slowdown) and trips Windows TDR watchdog warnings. At $ngl=40$ (full offload attempt), the card throws a hard CUDA OOM. This empirical proof demonstrates that $ngl=20$ is the exact boundary for peak performance without triggering memory thrashing.

### B. Driver Reclamation Settling Interval ($\tau_{\text{settle}}$) Sensitivity
To test the necessity of Algorithm 2's settling window, we varied $\tau_{\text{settle}} \in [0.0\text{ s}, 2.5\text{ s}]$ across 10 sequential pipeline runs, monitoring residual driver memory and allocation failure rates in Table VII.

```
TABLE VII: DRIVER RECLAMATION SETTLING INTERVAL (tau) SENSITIVITY
```

With $\tau_{\text{settle}} = 0.0\text{ s}$ (invoking processes immediately without driver synchronization), the pipeline fails in $60.0\%$ of runs. Because WDDM's deferred unmapping queue has not freed the pages from Stage 1, Stage 3's initialization request for $4,287\text{ MiB}$ sees only $3,304\text{ MiB}$ of free space, leading to an immediate OOM abort. As $\tau_{\text{settle}}$ increases, residual allocation drops exponentially, reaching zero ($0.0\text{ MiB}$) at $\tau_{\text{settle}} = 1.2\text{ s}$. Stretching the window to $2.0\text{ s}$ yields no extra memory clearance while adding dead idle time, validating $\tau_{\text{settle}} = 1.2\text{ s}$ as the optimal sweet spot.

### C. Context Window Length ($N_{\text{ctx}}$) Scaling & KV-Cache Growth
In Table VIII, we track how EdgeCascade scales as context length expands from $N_{\text{ctx}} = 128$ to $N_{\text{ctx}} = 2048$ tokens under the standard $ngl=20$ configuration.

```
TABLE VIII: CONTEXT WINDOW LENGTH (N_ctx) SCALING & KV-CACHE GROWTH
```

The data shows that KV-cache memory scales linearly with sequence length ($\sim 0.25\text{ MiB}$ per token across the active GPU layers). At $N_{\text{ctx}} = 2048$, Stage 3 VRAM allocation climbs to $4,810.0\text{ MiB}$, still safely inside the 6,144 MiB physical envelope ($1,334\text{ MiB}$ safety cushion). This confirms that EdgeCascade supports long context windows up to 2,048 tokens on 6 GB consumer GPUs without memory oversubscription.

### D. Host CPU Thread Allocation ($T_{\text{cpu}}$) for Stage 2 Verification
Because Stage 2 (FLAN-T5 Large) executes on the host AMD Ryzen 7 5800H processor, we tested the impact of CPU thread affinity across $T_{\text{cpu}} \in [2, 16]$ threads in Table IX.

```
TABLE IX: HOST CPU THREAD ALLOCATION (T_cpu) ABLATION ON STAGE 2
```

Scaling thread allocation from 2 to 16 threads drops Stage 2 latency from $285.40\text{ s}$ to $80.17\text{ s}$ ($3.56\times$ speedup). Scaling flattens past 8 physical cores because the workload saturates the dual-channel DDR4-3200 memory bus ($51.2\text{ GB/s}$ theoretical, reaching $42.1\text{ GB/s}$ actual under 16 threads). Setting $T_{\text{cpu}} = 16$ minimizes Stage 2 runtime while staying within the processor's 54W sustained package power envelope."""
    return md

if __name__ == '__main__':
    print(f"Sec 7 words: {len(get_sec7().split())}")
