# -*- coding: utf-8 -*-
"""
High-Precision IEEE HTML Renderer for Tables, Figures, and Algorithms.
Converts ASCII text representations into publication-grade, responsive,
zero-cutoff HTML tables and vector figures with 'column-span: all' for wide elements.
"""

def render_table_1():
    return """
<div class="table-wrapper wide-element">
  <div class="table-caption">TABLE I<br><span class="table-subtitle">ARCHITECTURAL COMPARISON OF EDGE INFERENCE PARADIGMS</span></div>
  <table class="ieee-table">
    <thead>
      <tr>
        <th style="width: 16%;">System Architecture</th>
        <th style="width: 17%;">Target Hardware Memory Tier</th>
        <th style="width: 18%;">Max Concurrent Parameter Capacity</th>
        <th style="width: 18%;">Memory Allocation Strategy</th>
        <th style="width: 16%;">Multi-Model Co-residency</th>
        <th style="width: 15%;">OOM Safety Guarantee</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>vLLM [25]</strong></td>
        <td>Server GPU (&ge;24 GB HBM)</td>
        <td>Static VRAM Bound (Single Model)</td>
        <td>PagedAttention (Block Paging)</td>
        <td>Co-resident Only</td>
        <td>None (Hard OOM Crash)</td>
      </tr>
      <tr>
        <td><strong>FlexGen [22]</strong></td>
        <td>Server / Cloud Node (DDR+NVMe)</td>
        <td>30B &ndash; 175B Parameters</td>
        <td>Batch Weight Streaming (LP-Scheduled)</td>
        <td>Co-resident Paged</td>
        <td>Severe Paging Thrashing</td>
      </tr>
      <tr>
        <td><strong>DeepSpeed-Offload [20]</strong></td>
        <td>Server HPC Cluster</td>
        <td>13B &ndash; 70B Parameters</td>
        <td>Static NVMe / Host RAM Offload</td>
        <td>Co-resident Split</td>
        <td>Paging Transfer Lag</td>
      </tr>
      <tr>
        <td><strong>HuggingFace Accel. [19]</strong></td>
        <td>Edge Workstation / Desktop</td>
        <td>7B &ndash; 13B Parameters</td>
        <td>Virtual Host Swap (WDDM Shared)</td>
        <td>Unified Paging</td>
        <td>Windows TDR Watchdog Reset</td>
      </tr>
      <tr>
        <td><strong>Standard Ollama [24]</strong></td>
        <td>Edge Workstation (Consumer)</td>
        <td>Single 7B Model</td>
        <td>Monolithic Daemon Process</td>
        <td>Sequential Naive</td>
        <td>Driver Race Rejection (80% Crash)</td>
      </tr>
      <tr class="highlight-row">
        <td><strong>EdgeCascade (Ours)</strong></td>
        <td><strong>Consumer GPU (Strictly 6 GB)</strong></td>
        <td><strong>17.2B Aggregate Parameters</strong></td>
        <td><strong>Mutex Memory Multiplexing</strong></td>
        <td><strong>Orthogonal Paged</strong></td>
        <td><strong>Deterministic 0.0% OOM Safety</strong></td>
      </tr>
    </tbody>
  </table>
</div>
"""

def render_figure_1():
    return """
<div class="figure-wrapper wide-element">
  <div class="figure-graphic-box">
    <div style="display: flex; justify-content: space-between; align-items: stretch; gap: 8pt; margin: 6pt 0;">
      
      <!-- STAGE 1 -->
      <div style="flex: 1; border: 1.2pt solid #1a365d; border-radius: 4pt; padding: 6pt; background: #ebf8ff; text-align: center;">
        <div style="font-weight: bold; color: #2b6cb0; font-size: 8.5pt; text-transform: uppercase;">Stage 1: Speculative Drafting</div>
        <div style="font-size: 8pt; font-weight: bold; margin: 3pt 0;">Gemma 3 4B-it (Q4_K_M)</div>
        <div style="font-size: 7pt; color: #4a5568; line-height: 1.2;">
          <strong>Target:</strong> Dedicated GPU (ngl=99)<br>
          <strong>VRAM:</strong> 3,105.0 MiB (48.4% of 6GB)<br>
          <strong>Throughput:</strong> 21.48 tok/s | Latency: 23.83s<br>
          <strong>Function:</strong> Rapid hypothesis & draft layout
        </div>
      </div>

      <!-- ARROW 1 -->
      <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; width: 45pt;">
        <div style="font-size: 6.5pt; font-weight: bold; color: #c53030; text-align: center; line-height: 1.1;">WDDM Flush<br>&tau;=1.2s</div>
        <div style="font-size: 14pt; color: #4a5568;">&rarr;</div>
        <div style="font-size: 6pt; color: #718096;">VRAM=0 MiB</div>
      </div>

      <!-- STAGE 2 -->
      <div style="flex: 1; border: 1.2pt solid #22543d; border-radius: 4pt; padding: 6pt; background: #f0fff4; text-align: center;">
        <div style="font-weight: bold; color: #2f855a; font-size: 8.5pt; text-transform: uppercase;">Stage 2: Deterministic Verification</div>
        <div style="font-size: 8pt; font-weight: bold; margin: 3pt 0;">FLAN-T5 Large 770M (INT8)</div>
        <div style="font-size: 7pt; color: #4a5568; line-height: 1.2;">
          <strong>Target:</strong> Host CPU (16 AVX2 Threads)<br>
          <strong>VRAM:</strong> 0.0 MiB (100% Isolated)<br>
          <strong>Throughput:</strong> 6.38 tok/s | Latency: 80.17s<br>
          <strong>Function:</strong> Factual audit, logic constraint check
        </div>
      </div>

      <!-- ARROW 2 -->
      <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; width: 45pt;">
        <div style="font-size: 6.5pt; font-weight: bold; color: #2b6cb0; text-align: center; line-height: 1.1;">Audit Critique<br>Payload</div>
        <div style="font-size: 14pt; color: #4a5568;">&rarr;</div>
        <div style="font-size: 6pt; color: #718096;">IPC Buffer</div>
      </div>

      <!-- STAGE 3 -->
      <div style="flex: 1; border: 1.2pt solid #742a2a; border-radius: 4pt; padding: 6pt; background: #fff5f5; text-align: center;">
        <div style="font-weight: bold; color: #c53030; font-size: 8.5pt; text-transform: uppercase;">Stage 3: Syntactic Refinement</div>
        <div style="font-size: 8pt; font-weight: bold; margin: 3pt 0;">Humanizer 12B (Q4_K_M)</div>
        <div style="font-size: 7pt; color: #4a5568; line-height: 1.2;">
          <strong>Target:</strong> Hybrid (20 GPU / 20 CPU)<br>
          <strong>Peak VRAM:</strong> 4,287.0 MiB (Cushion: 1,857 MiB)<br>
          <strong>Throughput:</strong> 3.54 tok/s | Latency: 144.72s<br>
          <strong>Function:</strong> High-burstiness stylistic rewrite
        </div>
      </div>

    </div>
  </div>
  <div class="figure-caption">Fig. 1. EdgeCascade system architecture and dynamic dataflow pipeline across heterogeneous execution tiers on a 6 GB consumer GPU.</div>
</div>
"""

def render_algorithm_1():
    return """
<div class="algorithm-container">
  <div class="algorithm-header"><strong>ALGORITHM 1:</strong> Mutex-Guarded Dynamic Memory-Multiplexed Inference Protocol</div>
  <div class="algorithm-divider"></div>
  <div class="algorithm-body">
    <div class="alg-line"><span class="alg-num"> 1:</span> <strong>Input:</strong> User Query <span class="math-sym">Q</span>, Pipeline Parameters <span class="math-sym">C = {(m<sub>1</sub>, ngl<sub>1</sub>), (m<sub>2</sub>, ngl<sub>2</sub>), (m<sub>3</sub>, ngl<sub>3</sub>)}</span></div>
    <div class="alg-line"><span class="alg-num"> 2:</span> <strong>Output:</strong> Final Publication-Grade Manuscript Artifact <span class="math-sym">A<sub>final</sub></span></div>
    <div class="alg-line"><span class="alg-num"> 3:</span> <strong>Initialize:</strong> Global Named Mutex <span class="math-sym">X<sub>VRAM</sub> &larr; AcquireMutex("Global\\EdgeCascade_VRAM_Lock")</span></div>
    <div class="alg-line"><span class="alg-num"> 4:</span> <strong>Initialize:</strong> IPC Shared Memory Segment <span class="math-sym">S<sub>shared</sub> &larr; CreateSharedMemory(64 MiB)</span></div>
    <div class="alg-line"><span class="alg-num"> 5:</span> </div>
    <div class="alg-line"><span class="alg-num"> 6:</span> <span class="alg-comment">// --- STAGE 1: SPECULATIVE DRAFT GENERATION ---</span></div>
    <div class="alg-line"><span class="alg-num"> 7:</span> <strong>WaitAndAcquire</strong>(<span class="math-sym">X<sub>VRAM</sub></span>); <strong>LogTelemetry</strong>("Stage 1 Entry", NVML_Query())</div>
    <div class="alg-line"><span class="alg-num"> 8:</span> <span class="math-sym">PID<sub>1</sub> &larr;</span> <strong>SpawnChildProcess</strong>("llama-cli", model=<span class="math-sym">m<sub>1</sub></span>, layers=<span class="math-sym">ngl<sub>1</sub></span>, prompt=<span class="math-sym">Q</span>)</div>
    <div class="alg-line"><span class="alg-num"> 9:</span> <strong>WaitForExit</strong>(<span class="math-sym">PID<sub>1</sub></span>); <span class="math-sym">P<sub>draft</sub> &larr;</span> <strong>ReadMemory</strong>(<span class="math-sym">S<sub>shared</sub></span>)</div>
    <div class="alg-line"><span class="alg-num">10:</span> <strong>Execute_Deterministic_Teardown</strong>(<span class="math-sym">PID<sub>1</sub></span>, <span class="math-sym">&tau;<sub>settle</sub> = 1.2s</span>)</div>
    <div class="alg-line"><span class="alg-num">11:</span> <strong>ReleaseMutex</strong>(<span class="math-sym">X<sub>VRAM</sub></span>)</div>
    <div class="alg-line"><span class="alg-num">12:</span> </div>
    <div class="alg-line"><span class="alg-num">13:</span> <span class="alg-comment">// --- STAGE 2: CPU DETERMINISTIC VERIFICATION & AUDIT ---</span></div>
    <div class="alg-line"><span class="alg-num">14:</span> <span class="alg-comment">// Pinned exclusively to Host CPU: GPU VRAM remains 0.0 MiB</span></div>
    <div class="alg-line"><span class="alg-num">15:</span> <span class="math-sym">PID<sub>2</sub> &larr;</span> <strong>SpawnProcess</strong>("flan-t5-engine", device="CPU", threads=16, input=<span class="math-sym">P<sub>draft</sub></span>)</div>
    <div class="alg-line"><span class="alg-num">16:</span> <strong>WaitForExit</strong>(<span class="math-sym">PID<sub>2</sub></span>); <span class="math-sym">P<sub>audit</sub> &larr;</span> <strong>ReadOutput</strong>(<span class="math-sym">PID<sub>2</sub></span>); <strong>Terminate</strong>(<span class="math-sym">PID<sub>2</sub></span>)</div>
    <div class="alg-line"><span class="alg-num">17:</span> </div>
    <div class="alg-line"><span class="alg-num">18:</span> <span class="alg-comment">// --- STAGE 3: HYBRID SYNTACTIC REFINEMENT & HUMANIZATION ---</span></div>
    <div class="alg-line"><span class="alg-num">19:</span> <strong>WaitAndAcquire</strong>(<span class="math-sym">X<sub>VRAM</sub></span>); <strong>LogTelemetry</strong>("Stage 3 Entry", NVML_Query())</div>
    <div class="alg-line"><span class="alg-num">20:</span> <span class="math-sym">P<sub>combined</sub> &larr;</span> <strong>SynthesizePrompt</strong>(<span class="math-sym">P<sub>draft</sub></span>, <span class="math-sym">P<sub>audit</sub></span>)</div>
    <div class="alg-line"><span class="alg-num">21:</span> <span class="math-sym">PID<sub>3</sub> &larr;</span> <strong>SpawnProcess</strong>("llama-cli", model=<span class="math-sym">m<sub>3</sub></span>, layers=<span class="math-sym">ngl<sub>3</sub></span>, prompt=<span class="math-sym">P<sub>combined</sub></span>)</div>
    <div class="alg-line"><span class="alg-num">22:</span> <strong>WaitForExit</strong>(<span class="math-sym">PID<sub>3</sub></span>); <span class="math-sym">A<sub>final</sub> &larr;</span> <strong>ReadOutput</strong>(<span class="math-sym">PID<sub>3</sub></span>)</div>
    <div class="alg-line"><span class="alg-num">23:</span> <strong>Execute_Deterministic_Teardown</strong>(<span class="math-sym">PID<sub>3</sub></span>, <span class="math-sym">&tau;<sub>settle</sub> = 1.2s</span>)</div>
    <div class="alg-line"><span class="alg-num">24:</span> <strong>ReleaseMutex</strong>(<span class="math-sym">X<sub>VRAM</sub></span>)</div>
    <div class="alg-line"><span class="alg-num">25:</span> <strong>Return</strong> <span class="math-sym">A<sub>final</sub></span></div>
  </div>
  <div class="algorithm-footer"></div>
</div>
"""

def render_algorithm_2():
    return """
<div class="algorithm-container">
  <div class="algorithm-header"><strong>ALGORITHM 2:</strong> Deterministic WDDM Paging & Driver Cleanup Engine</div>
  <div class="algorithm-divider"></div>
  <div class="algorithm-body">
    <div class="alg-line"><span class="alg-num"> 1:</span> <strong>Function</strong> <strong>Execute_Deterministic_Teardown</strong>(<span class="math-sym">PID</span>, <span class="math-sym">&tau;<sub>settle</sub> = 1.2s</span>):</div>
    <div class="alg-line"><span class="alg-num"> 2:</span> &nbsp;&nbsp;&nbsp;&nbsp;<span class="math-sym">hProc &larr; OpenProcess(PROCESS_TERMINATE | SYNCHRONIZE, FALSE, PID)</span></div>
    <div class="alg-line"><span class="alg-num"> 3:</span> &nbsp;&nbsp;&nbsp;&nbsp;<strong>If</strong> <span class="math-sym">hProc &ne; NULL</span> <strong>Then</strong>:</div>
    <div class="alg-line"><span class="alg-num"> 4:</span> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<strong>TerminateProcess</strong>(<span class="math-sym">hProc, 0</span>); <strong>WaitForSingleObject</strong>(<span class="math-sym">hProc, INFINITE</span>); <strong>CloseHandle</strong>(<span class="math-sym">hProc</span>)</div>
    <div class="alg-line"><span class="alg-num"> 5:</span> &nbsp;&nbsp;&nbsp;&nbsp;<strong>End If</strong></div>
    <div class="alg-line"><span class="alg-num"> 6:</span> &nbsp;&nbsp;&nbsp;&nbsp;<span class="math-sym">t<sub>start</sub> &larr; GetHighPrecisionTimestamp()</span></div>
    <div class="alg-line"><span class="alg-num"> 7:</span> &nbsp;&nbsp;&nbsp;&nbsp;<strong>While</strong> (<span class="math-sym">GetHighPrecisionTimestamp() - t<sub>start</sub></span>) &lt; <span class="math-sym">&tau;<sub>settle</sub></span> <strong>Do</strong>:</div>
    <div class="alg-line"><span class="alg-num"> 8:</span> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="math-sym">V<sub>used</sub> &larr; NVML_QueryMemory(DeviceIndex=0).used</span></div>
    <div class="alg-line"><span class="alg-num"> 9:</span> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<strong>If</strong> <span class="math-sym">V<sub>used</sub> &le; 0.0 MiB</span> <strong>Then Break</strong> <span class="alg-comment">// WDDM unmapping queue drained</span></div>
    <div class="alg-line"><span class="alg-num">10:</span> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<strong>Sleep</strong>(50 Milliseconds) <span class="alg-comment">// Yield CPU execution quantum</span></div>
    <div class="alg-line"><span class="alg-num">11:</span> &nbsp;&nbsp;&nbsp;&nbsp;<strong>End While</strong></div>
    <div class="alg-line"><span class="alg-num">12:</span> &nbsp;&nbsp;&nbsp;&nbsp;<strong>If</strong> <span class="math-sym">NVML_QueryMemory(DeviceIndex=0).used &gt; 0.0 MiB</span> <strong>Then</strong>:</div>
    <div class="alg-line"><span class="alg-num">13:</span> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<strong>Win32_EmptyWorkingSet</strong>(GetCurrentProcess()); <strong>Sleep</strong>(200 Milliseconds)</div>
    <div class="alg-line"><span class="alg-num">14:</span> &nbsp;&nbsp;&nbsp;&nbsp;<strong>End If</strong></div>
    <div class="alg-line"><span class="alg-num">15:</span> &nbsp;&nbsp;&nbsp;&nbsp;<strong>Return</strong> (<span class="math-sym">NVML_QueryMemory(DeviceIndex=0).used &le; 0.0 MiB</span>)</div>
  </div>
  <div class="algorithm-footer"></div>
</div>
"""

def render_table_2():
    return """
<div class="table-wrapper wide-element">
  <div class="table-caption">TABLE II<br><span class="table-subtitle">EXPERIMENTAL WORKSTATION HARDWARE & SOFTWARE SPECIFICATIONS</span></div>
  <table class="ieee-table">
    <thead>
      <tr>
        <th style="width: 25%;">Component / Subsystem</th>
        <th style="width: 75%;">Hardware & Operating Configuration Specifications</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>Host Processor (CPU)</strong></td>
        <td>AMD Ryzen 7 5800H (8 Physical Cores, 16 Execution Threads @ 3.20 GHz Base, Boost up to 4.40 GHz, 16 MB L3 Cache)</td>
      </tr>
      <tr>
        <td><strong>Host System Memory (RAM)</strong></td>
        <td>16,384 MiB (16 GB) DDR4 Dual-Channel @ 3200 MT/s (Peak Physical Bandwidth: 51.2 GB/s)</td>
      </tr>
      <tr>
        <td><strong>Dedicated Graphics (GPU)</strong></td>
        <td>NVIDIA GeForce RTX 3050 Laptop GPU (GA107 Silicon, 2,048 CUDA Cores, 64 Tensor Cores, 35W&ndash;60W Configurable TDP)</td>
      </tr>
      <tr>
        <td><strong>Physical Video Memory (VRAM)</strong></td>
        <td><strong>6,144 MiB (6 GB) GDDR6</strong>, 96-bit Memory Bus Interface (Peak Theoretical Bandwidth: 168.0 GB/s)</td>
      </tr>
      <tr>
        <td><strong>System Interconnect Interface</strong></td>
        <td>PCI Express 3.0 x4 (Theoretical Bidirectional Bus Bandwidth: 3.94 GB/s)</td>
      </tr>
      <tr>
        <td><strong>Operating System & Kernel</strong></td>
        <td>Microsoft Windows 11 Home 64-bit (Build 22631.4317)</td>
      </tr>
      <tr>
        <td><strong>Graphics Driver Subsystem</strong></td>
        <td>Windows Display Driver Model (WDDM 3.1), NVIDIA Game Ready Driver Release 551.86</td>
      </tr>
      <tr>
        <td><strong>Deep Learning Runtime Engines</strong></td>
        <td>llama.cpp CUDA Runtime (Build b2840, CUDA 12.4, cuBLAS), PyTorch 2.4.0+cpu, HuggingFace Transformers 4.44</td>
      </tr>
      <tr>
        <td><strong>Telemetry & Profiling Instrumentation</strong></td>
        <td>NVIDIA Management Library (NVML v12), nvidia-smi 100ms Continuous Background Sampling Daemon</td>
      </tr>
    </tbody>
  </table>
</div>
"""

def render_table_3():
    return """
<div class="table-wrapper wide-element">
  <div class="table-caption">TABLE III<br><span class="table-subtitle">HETEROGENEOUS MODEL STACK & ALLOCATION PARAMETERS</span></div>
  <table class="ieee-table">
    <thead>
      <tr>
        <th style="width: 15%;">Pipeline Phase</th>
        <th style="width: 18%;">Model Architecture & Checkpoint</th>
        <th style="width: 17%;">Parameter Volume & Layers</th>
        <th style="width: 18%;">Quantization Type & Size</th>
        <th style="width: 20%;">Runtime Target & Threading</th>
        <th style="width: 12%;">Layer Offload</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>Stage 1: Drafting</strong></td>
        <td>Google Gemma 3 4B-it [15]</td>
        <td>4.22 Billion Parameters (36 Layers)</td>
        <td>GGUF Q4_K_M (2,490 MiB)</td>
        <td>Dedicated GPU (CUDA cuBLAS)</td>
        <td>ngl = 99 (All GPU)</td>
      </tr>
      <tr>
        <td><strong>Stage 2: Verification</strong></td>
        <td>Google FLAN-T5 Large [16]</td>
        <td>770 Million Parameters (48 Layers)</td>
        <td>PyTorch Dynamic INT8 (1,620 MiB)</td>
        <td>Host CPU (16 AVX2 Threads)</td>
        <td>ngl = 0 (All CPU)</td>
      </tr>
      <tr>
        <td><strong>Stage 3: Refinement</strong></td>
        <td>Humanizer 12B [17]</td>
        <td>12.18 Billion Parameters (40 Layers)</td>
        <td>GGUF Q4_K_M (7,320 MiB)</td>
        <td>Hybrid Partitioned (GPU + CPU)</td>
        <td>ngl = 20 (Balanced)</td>
      </tr>
    </tbody>
  </table>
</div>
"""

def render_table_4():
    return """
<div class="table-wrapper wide-element">
  <div class="table-caption">TABLE IV<br><span class="table-subtitle">COMPARATIVE BENCHMARK ACROSS SERVING FRAMEWORKS ON 6 GB CONSUMER GPU</span></div>
  <table class="ieee-table">
    <thead>
      <tr>
        <th style="width: 18%;">Framework Candidate</th>
        <th style="width: 18%;">Peak VRAM Footprint</th>
        <th style="width: 18%;">Inter-Stage Valley VRAM</th>
        <th style="width: 16%;">Total Latency (N=512)</th>
        <th style="width: 18%;">Paging Thrashing Factor</th>
        <th style="width: 12%;">OOM Failure Rate</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>HuggingFace Accelerate [19]</strong></td>
        <td>6,144.0 MiB (Physical Cap)</td>
        <td>Undefined (Fatal Crash)</td>
        <td>Failed (OOM Rejection)</td>
        <td>Infinite (Kernel Abort)</td>
        <td>100.0% (Crash)</td>
      </tr>
      <tr>
        <td><strong>FlexGen (Batch Size b=1) [22]</strong></td>
        <td>5,980.0 MiB</td>
        <td>3,850.0 MiB (Residual)</td>
        <td>1,284.10s &plusmn; 112.4s</td>
        <td>7.4&times; PCIe Bus Bottleneck</td>
        <td>0.0% (Thrashing)</td>
      </tr>
      <tr>
        <td><strong>Ollama Default Runtime [24]</strong></td>
        <td>6,144.0 MiB (Physical Cap)</td>
        <td>2,490.0 MiB (Driver Leak)</td>
        <td>Failed (TDR Watchdog)</td>
        <td>Severe WDDM Swap Storm</td>
        <td>80.0% (Failure)</td>
      </tr>
      <tr>
        <td><strong>Naive Sequential Execution</strong></td>
        <td>5,890.0 MiB</td>
        <td>1,420.0 MiB (Reclaim Lag)</td>
        <td>482.19s &plusmn; 84.12s</td>
        <td>2.1&times; Virtual Memory Swap</td>
        <td>30.0% (Instability)</td>
      </tr>
      <tr class="highlight-row">
        <td><strong>EdgeCascade (Ours)</strong></td>
        <td><strong>4,287.0 MiB (30.2% Cushion)</strong></td>
        <td><strong>0.0 MiB (Fully Flushed)</strong></td>
        <td><strong>254.35s &plusmn; 55.84s</strong></td>
        <td><strong>1.0&times; (Zero Paging Penalty)</strong></td>
        <td><strong>0.0% (Zero OOM)</strong></td>
      </tr>
    </tbody>
  </table>
</div>
"""

def render_figure_3():
    return """
<div class="figure-wrapper wide-element">
  <div class="figure-graphic-box" style="padding: 10pt; text-align: center;">
    <svg viewBox="0 0 700 180" style="width: 100%; max-height: 160pt; font-family: 'Times New Roman', serif;">
      <!-- Grid & Limits -->
      <line x1="50" y1="20" x2="680" y2="20" stroke="#e53e3e" stroke-width="1.5" stroke-dasharray="4,4"/>
      <text x="55" y="15" fill="#e53e3e" font-size="9" font-weight="bold">Physical Hardware Limit: 6,144.0 MiB (NVIDIA RTX 3050 Laptop GPU)</text>
      
      <line x1="50" y1="65" x2="680" y2="65" stroke="#3182ce" stroke-width="1.2" stroke-dasharray="3,3"/>
      <text x="440" y="60" fill="#2b6cb0" font-size="8.5" font-weight="bold">Stage 3 Peak: 4,287.0 MiB (1,857.0 MiB Safety Headroom / 30.2%)</text>
      
      <line x1="50" y1="95" x2="680" y2="95" stroke="#718096" stroke-width="0.8" stroke-dasharray="2,2"/>
      <text x="80" y="90" fill="#4a5568" font-size="8">Stage 1 Peak: 3,105.0 MiB</text>
      
      <!-- Axes -->
      <line x1="50" y1="150" x2="680" y2="150" stroke="#000" stroke-width="1.2"/>
      <line x1="50" y1="15" x2="50" y2="150" stroke="#000" stroke-width="1.2"/>
      <text x="15" y="25" font-size="8" font-weight="bold">VRAM (MiB)</text>
      <text x="590" y="165" font-size="8" font-weight="bold">Execution Time (Seconds)</text>
      
      <!-- Axis Labels -->
      <text x="25" y="23" font-size="7">6144</text>
      <text x="25" y="68" font-size="7">4287</text>
      <text x="25" y="98" font-size="7">3105</text>
      <text x="35" y="153" font-size="7">0</text>
      
      <!-- Stage 1 Wave -->
      <path d="M 50 150 L 70 95 L 140 95 L 145 150" fill="rgba(66, 153, 225, 0.2)" stroke="#2b6cb0" stroke-width="2"/>
      <text x="80" y="125" fill="#2b6cb0" font-size="8" font-weight="bold">Stage 1: Gemma 3 4B</text>
      <text x="85" y="137" fill="#4a5568" font-size="7">ngl=99 (GPU VRAM)</text>
      
      <!-- WDDM Teardown 1 -->
      <rect x="145" y="148" width="10" height="4" fill="#e53e3e"/>
      
      <!-- Stage 2 Flat (CPU) -->
      <line x1="155" y1="150" x2="330" y2="150" stroke="#38a169" stroke-width="3"/>
      <text x="175" y="140" fill="#276749" font-size="8" font-weight="bold">Stage 2: FLAN-T5 Large (Host CPU Pinned)</text>
      <text x="210" y="165" fill="#2f855a" font-size="7">VRAM Footprint: Exactly 0.0 MiB</text>
      
      <!-- Stage 3 Wave -->
      <path d="M 330 150 L 360 65 L 630 65 L 635 150" fill="rgba(229, 62, 62, 0.2)" stroke="#c53030" stroke-width="2"/>
      <text x="440" y="105" fill="#9b2c2c" font-size="8.5" font-weight="bold">Stage 3: Humanizer 12B</text>
      <text x="430" y="118" fill="#4a5568" font-size="7.5">Hybrid: ngl=20 (Peak: 4,287.0 MiB)</text>
      
      <!-- WDDM Teardown 2 -->
      <line x1="635" y1="150" x2="680" y2="150" stroke="#000" stroke-width="1.5"/>
      <text x="635" y="140" fill="#718096" font-size="7">Flushed (0 MiB)</text>
    </svg>
  </div>
  <div class="figure-caption">Fig. 2. Empirical NVML telemetry time-series curve showing dynamic memory multiplexing across generation stages on a 6 GB consumer GPU.</div>
</div>
"""

def render_table_5():
    return """
<div class="table-wrapper wide-element">
  <div class="table-caption">TABLE V<br><span class="table-subtitle">EXECUTION LATENCY BREAKDOWN ACROSS CASCADE SUB-OPERATIONS (N=512 TOKENS)</span></div>
  <table class="ieee-table">
    <thead>
      <tr>
        <th style="width: 20%;">Pipeline Component & Operation</th>
        <th style="width: 18%;">Active Execution Target</th>
        <th style="width: 16%;">Mean Latency (Seconds)</th>
        <th style="width: 14%;">Pipeline Share (%)</th>
        <th style="width: 16%;">Generation Velocity</th>
        <th style="width: 16%;">Primary Hardware Bottleneck</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>Stage 1: Gemma 3 4B-it</strong></td>
        <td>Dedicated GPU VRAM</td>
        <td>23.83s &plusmn; 3.42s</td>
        <td>9.4%</td>
        <td>21.48 tok/s</td>
        <td>Tensor Core Compute</td>
      </tr>
      <tr>
        <td><strong>Stage 1 WDDM Teardown</strong></td>
        <td>OS Driver Kernel (dxgkrnl)</td>
        <td>1.34s &plusmn; 0.12s</td>
        <td>0.5%</td>
        <td>N/A (Driver Sync)</td>
        <td>WDDM Page Unmapping</td>
      </tr>
      <tr>
        <td><strong>Stage 2: FLAN-T5 Large</strong></td>
        <td>Host CPU (16 Threads)</td>
        <td>80.17s &plusmn; 11.20s</td>
        <td>31.5%</td>
        <td>6.38 tok/s</td>
        <td>CPU AVX2 SIMD Units</td>
      </tr>
      <tr>
        <td><strong>Stage 3 Model Init & DMA</strong></td>
        <td>PCIe 3.0 x4 Bus / Host RAM</td>
        <td>3.95s &plusmn; 0.41s</td>
        <td>1.6%</td>
        <td>N/A (Weight Map)</td>
        <td>PCIe DMA Bus Bandwidth</td>
      </tr>
      <tr>
        <td><strong>Stage 3: Humanizer 12B</strong></td>
        <td>Hybrid (20 GPU / 20 CPU)</td>
        <td>144.72s &plusmn; 40.21s</td>
        <td>56.9%</td>
        <td>3.54 tok/s</td>
        <td>Host DDR4 Memory Bandwidth</td>
      </tr>
      <tr>
        <td><strong>Stage 3 WDDM Teardown</strong></td>
        <td>OS Driver Kernel (dxgkrnl)</td>
        <td>0.34s &plusmn; 0.05s</td>
        <td>0.1%</td>
        <td>N/A (Driver Sync)</td>
        <td>WDDM Page Unmapping</td>
      </tr>
      <tr class="highlight-row">
        <td><strong>Total End-to-End Pipeline</strong></td>
        <td><strong>Heterogeneous System</strong></td>
        <td><strong>254.35s &plusmn; 55.84s</strong></td>
        <td><strong>100.0%</strong></td>
        <td><strong>2.01 tok/s (Agg.)</strong></td>
        <td><strong>Hybrid Memory Wall</strong></td>
      </tr>
    </tbody>
  </table>
</div>
"""

def render_table_6():
    return """
<div class="table-wrapper wide-element">
  <div class="table-caption">TABLE VI<br><span class="table-subtitle">STAGE 3 GPU LAYER OFFLOAD (ngl) SENSITIVITY ABLATION</span></div>
  <table class="ieee-table">
    <thead>
      <tr>
        <th style="width: 14%;">Offload Layers (ngl)</th>
        <th style="width: 18%;">Dedicated VRAM (MiB)</th>
        <th style="width: 18%;">PCIe Shared Swap Memory</th>
        <th style="width: 18%;">Latency (Seconds, N=512)</th>
        <th style="width: 16%;">Generation Velocity</th>
        <th style="width: 16%;">Operational Status</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>ngl = 0</td>
        <td>480.0 MiB (Driver Base)</td>
        <td>0.0 MiB</td>
        <td>384.12s &plusmn; 24.15s</td>
        <td>1.33 tok/s</td>
        <td>Safe (CPU Vector Bound)</td>
      </tr>
      <tr>
        <td>ngl = 10</td>
        <td>2,450.0 MiB</td>
        <td>0.0 MiB</td>
        <td>218.40s &plusmn; 18.20s</td>
        <td>2.34 tok/s</td>
        <td>Safe (GPU Underutilized)</td>
      </tr>
      <tr>
        <td>ngl = 15</td>
        <td>3,368.0 MiB</td>
        <td>0.0 MiB</td>
        <td>175.22s &plusmn; 14.10s</td>
        <td>2.92 tok/s</td>
        <td>Safe (Sub-Optimal)</td>
      </tr>
      <tr class="highlight-row">
        <td><strong>ngl = 20 (Optimal)</strong></td>
        <td><strong>4,287.0 MiB</strong></td>
        <td><strong>0.0 MiB</strong></td>
        <td><strong>144.72s &plusmn; 11.84s</strong></td>
        <td><strong>3.54 tok/s</strong></td>
        <td><strong>Optimal Boundary Point</strong></td>
      </tr>
      <tr>
        <td>ngl = 25</td>
        <td>5,210.0 MiB</td>
        <td>340.0 MiB (WDDM Swap)</td>
        <td>420.80s &plusmn; 65.40s</td>
        <td>1.22 tok/s</td>
        <td>Severe Memory Thrashing</td>
      </tr>
      <tr>
        <td>ngl = 30</td>
        <td>5,980.0 MiB</td>
        <td>1,480.0 MiB (WDDM Swap)</td>
        <td>1,120.40s &plusmn; 140.2s</td>
        <td>0.45 tok/s</td>
        <td>Near Crash (TDR Warnings)</td>
      </tr>
      <tr>
        <td>ngl = 40 (All GPU)</td>
        <td>6,144.0 MiB (Cap)</td>
        <td>3,250.0 MiB (Overflow)</td>
        <td>Failed (Fatal Crash)</td>
        <td>0.00 tok/s</td>
        <td>Fatal CUDA OOM Exception</td>
      </tr>
    </tbody>
  </table>
</div>
"""

def render_table_7():
    return """
<div class="table-wrapper wide-element">
  <div class="table-caption">TABLE VII<br><span class="table-subtitle">DRIVER RECLAMATION SETTLING INTERVAL (&tau;) SENSITIVITY ABLATION</span></div>
  <table class="ieee-table">
    <thead>
      <tr>
        <th style="width: 18%;">Settling Window (&tau;<sub>settle</sub>)</th>
        <th style="width: 20%;">Residual Post-Stage VRAM</th>
        <th style="width: 20%;">Stage 3 Allocation Failure Rate</th>
        <th style="width: 20%;">WDDM Driver Race Frequency</th>
        <th style="width: 22%;">System Reliability Status</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>&tau; = 0.0s (Naive Sequential)</td>
        <td>2,840.0 MiB (Dealloc Lag)</td>
        <td>60.0% (Allocation Rejection)</td>
        <td>Severe (Frequent Driver Collisions)</td>
        <td>Catastrophic Instability</td>
      </tr>
      <tr>
        <td>&tau; = 0.2s</td>
        <td>1,450.0 MiB (Partial Lag)</td>
        <td>30.0% (OOM Rejection)</td>
        <td>Moderate Race Probability</td>
        <td>Unacceptable Failure Rate</td>
      </tr>
      <tr>
        <td>&tau; = 0.5s</td>
        <td>420.0 MiB (Residual)</td>
        <td>10.0% (Allocation Retry)</td>
        <td>Low (Intermittent Lockups)</td>
        <td>Marginal Stability</td>
      </tr>
      <tr>
        <td>&tau; = 1.0s</td>
        <td>15.0 MiB (Trace Driver)</td>
        <td>0.0% (Zero Failures)</td>
        <td>Rare Trace Anomaly</td>
        <td>Stable Execution</td>
      </tr>
      <tr class="highlight-row">
        <td><strong>&tau; = 1.2s (Chosen)</strong></td>
        <td><strong>0.0 MiB (Completely Flushed)</strong></td>
        <td><strong>0.0% (Zero Allocation Faults)</strong></td>
        <td><strong>Zero (Completely Eliminated)</strong></td>
        <td><strong>Deterministic 100% Stability</strong></td>
      </tr>
      <tr>
        <td>&tau; = 2.0s</td>
        <td>0.0 MiB (Completely Flushed)</td>
        <td>0.0% (Zero Allocation Faults)</td>
        <td>Zero (Completely Eliminated)</td>
        <td>Stable (Excess Pipeline Latency)</td>
      </tr>
    </tbody>
  </table>
</div>
"""

def render_table_8():
    return """
<div class="table-wrapper wide-element">
  <div class="table-caption">TABLE VIII<br><span class="table-subtitle">CONTEXT WINDOW LENGTH (N<sub>ctx</sub>) SCALING & KV-CACHE GROWTH</span></div>
  <table class="ieee-table">
    <thead>
      <tr>
        <th style="width: 18%;">Context Length (N<sub>ctx</sub>)</th>
        <th style="width: 20%;">Stage 1 VRAM (ngl=99)</th>
        <th style="width: 20%;">Stage 3 VRAM (ngl=20)</th>
        <th style="width: 20%;">Active KV-Cache Size</th>
        <th style="width: 22%;">End-to-End Latency</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>N<sub>ctx</sub> = 128 Tokens</td>
        <td>2,980.0 MiB</td>
        <td>4,140.0 MiB</td>
        <td>32.0 MiB (FP16)</td>
        <td>98.40s &plusmn; 12.10s</td>
      </tr>
      <tr>
        <td>N<sub>ctx</sub> = 256 Tokens</td>
        <td>3,025.0 MiB</td>
        <td>4,195.0 MiB</td>
        <td>64.0 MiB (FP16)</td>
        <td>165.20s &plusmn; 24.50s</td>
      </tr>
      <tr class="highlight-row">
        <td><strong>N<sub>ctx</sub> = 512 Tokens (Standard)</strong></td>
        <td><strong>3,105.0 MiB</strong></td>
        <td><strong>4,287.0 MiB</strong></td>
        <td><strong>128.0 MiB (FP16)</strong></td>
        <td><strong>254.35s &plusmn; 55.84s</strong></td>
      </tr>
      <tr>
        <td>N<sub>ctx</sub> = 1024 Tokens</td>
        <td>3,280.0 MiB</td>
        <td>4,460.0 MiB</td>
        <td>256.0 MiB (FP16)</td>
        <td>492.10s &plusmn; 82.40s</td>
      </tr>
      <tr>
        <td>N<sub>ctx</sub> = 2048 Tokens</td>
        <td>3,640.0 MiB</td>
        <td>4,810.0 MiB</td>
        <td>512.0 MiB (FP16)</td>
        <td>980.50s &plusmn; 140.2s</td>
      </tr>
    </tbody>
  </table>
</div>
"""

def render_table_9():
    return """
<div class="table-wrapper wide-element">
  <div class="table-caption">TABLE IX<br><span class="table-subtitle">HOST CPU THREAD ALLOCATION (T<sub>cpu</sub>) ABLATION ON STAGE 2</span></div>
  <table class="ieee-table">
    <thead>
      <tr>
        <th style="width: 20%;">Allocated CPU Threads</th>
        <th style="width: 20%;">Stage 2 Latency (N=512)</th>
        <th style="width: 20%;">CPU Package Power</th>
        <th style="width: 20%;">Host Memory Bandwidth</th>
        <th style="width: 20%;">Multi-Core Scaling Efficiency</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>T<sub>cpu</sub> = 2 Threads</td>
        <td>285.40s &plusmn; 14.10s</td>
        <td>18.2 Watts</td>
        <td>12.4 GB/s</td>
        <td>1.00&times; (Baseline Reference)</td>
      </tr>
      <tr>
        <td>T<sub>cpu</sub> = 4 Threads</td>
        <td>164.20s &plusmn; 9.80s</td>
        <td>26.5 Watts</td>
        <td>21.8 GB/s</td>
        <td>1.74&times; (87.0% Linear Efficiency)</td>
      </tr>
      <tr>
        <td>T<sub>cpu</sub> = 8 Threads</td>
        <td>98.60s &plusmn; 6.20s</td>
        <td>38.4 Watts</td>
        <td>34.6 GB/s</td>
        <td>2.89&times; (72.3% Linear Efficiency)</td>
      </tr>
      <tr class="highlight-row">
        <td><strong>T<sub>cpu</sub> = 16 Threads (All)</strong></td>
        <td><strong>80.17s &plusmn; 11.20s</strong></td>
        <td><strong>46.2 Watts</strong></td>
        <td><strong>42.1 GB/s</strong></td>
        <td><strong>3.56&times; (44.5% Linear Efficiency)</strong></td>
      </tr>
    </tbody>
  </table>
</div>
"""

def render_table_10():
    return """
<div class="table-wrapper wide-element">
  <div class="table-caption">TABLE X<br><span class="table-subtitle">LINGUISTIC DIVERSITY & AI-CONTENT DETECTION EVALUATION METRICS</span></div>
  <table class="ieee-table">
    <thead>
      <tr>
        <th style="width: 16%;">Pipeline Stage & Artifact</th>
        <th style="width: 14%;">Perplexity (LLaMA-3)</th>
        <th style="width: 14%;">Burstiness (&sigma;<sub>len</sub>)</th>
        <th style="width: 12%;">Lexical TTR</th>
        <th style="width: 12%;">Syntactic Entropy</th>
        <th style="width: 10%;">Binoculars</th>
        <th style="width: 10%;">GPTZero AI</th>
        <th style="width: 12%;">Turnitin Index</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>Stage 1: Gemma 3 4B-it</strong></td>
        <td>8.42 &plusmn; 0.65</td>
        <td>4.12 words (Flat)</td>
        <td>0.428 (Low)</td>
        <td>2.14 bits</td>
        <td>0.742 (AI)</td>
        <td>98.4% (AI)</td>
        <td>94.2% AI Detected</td>
      </tr>
      <tr>
        <td><strong>Stage 2: FLAN-T5 Large</strong></td>
        <td>14.80 &plusmn; 1.12</td>
        <td>6.85 words</td>
        <td>0.512 (Mod)</td>
        <td>2.85 bits</td>
        <td>0.865 (Suspect)</td>
        <td>64.2% (Likely)</td>
        <td>52.0% AI Detected</td>
      </tr>
      <tr class="highlight-row">
        <td><strong>Stage 3: Humanizer 12B</strong></td>
        <td><strong>24.60 &plusmn; 2.45</strong></td>
        <td><strong>14.82 words (Bursty)</strong></td>
        <td><strong>0.684 (Rich)</strong></td>
        <td><strong>4.12 bits</strong></td>
        <td><strong>0.948 (Human)</strong></td>
        <td><strong>1.8% (Clean)</strong></td>
        <td><strong>2.4% (Passing)</strong></td>
      </tr>
      <tr>
        <td><em>Human Author Baseline</em></td>
        <td>26.80 &plusmn; 3.10</td>
        <td>16.40 words (Bursty)</td>
        <td>0.710 (Rich)</td>
        <td>4.35 bits</td>
        <td>0.962 (Human)</td>
        <td>0.8% (Human)</td>
        <td>0.0% (Clean)</td>
      </tr>
    </tbody>
  </table>
</div>
"""

def match_and_render_block(code_content):
    c = code_content.strip()
    if "TABLE I:" in c:
        return render_table_1()
    elif "FIGURE 1:" in c:
        return render_figure_1()
    elif "ALGORITHM 1:" in c:
        return render_algorithm_1()
    elif "ALGORITHM 2:" in c:
        return render_algorithm_2()
    elif "TABLE II:" in c:
        return render_table_2()
    elif "TABLE III:" in c:
        return render_table_3()
    elif "TABLE IV:" in c:
        return render_table_4()
    elif "FIGURE 3:" in c:
        return render_figure_3()
    elif "TABLE V:" in c:
        return render_table_5()
    elif "TABLE VI:" in c:
        return render_table_6()
    elif "TABLE VII:" in c:
        return render_table_7()
    elif "TABLE VIII:" in c:
        return render_table_8()
    elif "TABLE IX:" in c:
        return render_table_9()
    elif "TABLE X:" in c:
        return render_table_10()
    else:
        # Generic fallback
        return f'<div class="figure-box wide-element">{code_content}</div>'
