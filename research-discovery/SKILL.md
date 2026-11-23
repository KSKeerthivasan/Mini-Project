---
name: unified-academic-researcher
description: "End-to-end academic AI research pipeline: literature gap discovery, prior-art landscaping, paper deconstruction, formal experimental design, statistical validation, adversarial peer-review stress-testing, and IEEE/ACM manuscript preparation."
---

# Unified Academic Researcher (SKILL.md)

An all-in-one meta-skill consolidating literature discovery, paper deconstruction, experimental protocol design, statistical analysis, adversarial peer-review stress-testing, and publication drafting for Computer Science, NLP, and MLSys venues (IEEE, ACM, ACL, NeurIPS).

---

## 1. Operational Philosophy & Principles

1. **Evidence Precedes Claims**: Never declare novelty, superior performance, or state-of-the-art (SOTA) without verified empirical benchmarks and explicit prior-art citations.
2. **Methodological Rigor**: Hypotheses must follow the strict `If / Then / Because` format with predefined primary, guardrail, and secondary metrics.
3. **Reproducibility by Design**: All experiment definitions must specify explicit random seeds, exact hardware constraints, model quantization schemas (e.g., GGUF Q4_K_M), context bounds, and parameter temperatures.
4. **Adversarial Self-Audit**: Proactively interrogate every claim from the perspective of an aggressive peer reviewer (e.g., Reviewer #2) before committing to a research direction or manuscript submission.

---

## 2. Module 1: Literature Discovery & Prior Art Landscaping
*(Synthesizes: `research`, `litreview`, `pulse`, `patent`)*

### 2.1 The PICO / SPIDER Search Strategy
When investigating a candidate research topic, formulate systematic queries across ArXiv, IEEE Xplore, Semantic Scholar, and Google Scholar using structured facets:
- **Population (P)**: Target models or hardware domain (e.g., *autoregressive LLMs, seq2seq models, edge GPUs, sub-8GB VRAM*).
- **Intervention (I)**: The proposed novel mechanism (e.g., *heterogeneous cascaded pipelines, proposition-level attention bounding, temporal memory multiplexing*).
- **Comparison (C)**: Standard baselines (e.g., *monolithic causal generation, DIPPER, FlexGen, zero-shot prompting*).
- **Outcome (O)**: Quantifiable improvements (e.g., *AI detector AUROC reduction, named entity recall, peak VRAM ceiling*).

### 2.2 Prior-Art Gap Triangulation
For every related paper identified, populate the **Prior-Art Triangulation Matrix**:

| Prior Art / Paper | Core Mechanism | Documented Limitation | Our Distinct Research Gap |
|---|---|---|---|
| *Paper A (e.g., DIPPER)* | T5-XXL paragraph paraphrasing | High lexical variance causes entity hallucination | Decouple syntax from style across heterogeneous model families |
| *Paper B (e.g., Binoculars)* | Cross-perplexity ratio scoring | Exploits uniform autoregressive n-gram predictability | Break token distribution via bidirectional encoder reordering |
| *Paper C (e.g., FlexGen)* | Server-scale offload engine | High PCIe bandwidth and server-scale RAM required | Asymmetric hybrid layer offload on 6 GB consumer mobile GPUs |

---

## 3. Module 2: Paper Deconstruction & Extraction
*(Synthesizes: `research-summarizer`, `deepread`)*

When analyzing reference papers, apply the **6-Point Deep Deconstruction Protocol**:
1. **Core Problem Statement**: What exact bottleneck (not symptoms) does the paper solve?
2. **Theoretical Hypothesis**: What is the mathematical or architectural intuition underlying their solution?
3. **Model & Hyperparameter Profile**: Exact weights, layer count, context window ($c$), temperature ($T$), top-$p$, top-$k$, and quantization formats.
4. **Dataset Provenance**: Are benchmarks synthetic, human-curated, or public corpora (e.g., CNN/DailyMail, XSum, RAID)?
5. **Reported vs. Actual Metrics**: Distinguish between headline marketing claims and the actual ablation tables in appendices.
6. **Vulnerabilities & Attack Surface**: Where does the method fail? What edge cases did the authors omit or concede in the limitations section?

---

## 4. Module 3: Formal Experimental Design
*(Synthesizes: `experiment-designer`)*

### 4.1 Formal Hypothesis Structuring
All research hypotheses must be written in the validated format:
$$\text{If } [\text{Intervention } \mathcal{I}] \implies \text{Then } [\text{Metric } \mathcal{M} \text{ shifts by } \delta] \impliedby \text{Because } [\text{Causal Mechanism } \mathcal{C}]$$

- **Example $H_1$**: *If* machine-generated text is routed through an encoder-decoder paraphraser bounded to $k$-word sentence propositions prior to causal tone polishing, *then* detector AUROC on *Binoculars* will decrease by $\ge 0.30$ while retaining $\ge 90\%$ of named entities, *because* bidirectional attention reorders constituent syntax trees without left-to-right token probability collapse.

### 4.2 Metric Taxonomy

```
├── Primary Decision Metrics (Direct Proof of Novelty)
│   ├── AI Detection Evasion: AUROC, TPR @ 1% FPR (Binoculars, Fast-DetectGPT)
│   └── Factual Integrity: Named Entity Recall (R_NER), Numerical Retention (P_Num)
│
├── Guardrail Metrics (Safety & Quality Controls)
│   ├── Semantic Preservation: BERTScore (F1), Levenshtein Distance
│   └── Linguistic Fluency: Reference Perplexity (PPL), Mauve Score
│
└── Secondary Systems Metrics (Cost & Deployment Feasibility)
    ├── Memory Ceiling: Peak Allocated VRAM (MiB via nvidia-smi)
    └── Latency Breakdown: Inference Latency per stage (s), Process Swapping Overhead (s)
```

### 4.3 Statistical Sample Size Sizing
Determine sample size ($N$) before running tests to prevent underpowered results:
$$N = \frac{2 \cdot (Z_{\alpha/2} + Z_{\beta})^2 \cdot \sigma^2}{\Delta^2}$$
- **Default Standards**: Significance level $\alpha = 0.05$ ($Z_{\alpha/2} = 1.96$), Statistical Power $1 - \beta = 0.80$ ($Z_{\beta} = 0.84$).
- **Minimum Evaluation Set**: For text generation benchmarks, standard sample size is $N = 100 \text{ to } 200$ balanced prompts across technical, news, and conversational domains.

---

## 5. Module 4: Statistical Validation & Ablation Protocols
*(Synthesizes: `statistical-analyst`, `eval`, `autoresearch-agent`)*

### 5.1 The 4-Way Canonical Ablation Protocol
To prove that every component in a cascaded architecture is essential, enforce the **4-Way Factorial Matrix**:
1. **Configuration A (Baseline)**: Stage 1 alone (Generative Draft).
2. **Configuration B (Structural Intervention Alone)**: Stage 1 + Stage 2 (Draft + Seq2Seq Paraphraser).
3. **Configuration C (Stylistic Intervention Alone)**: Stage 1 + Stage 3 (Draft + Causal Humanizer, skipping Stage 2).
4. **Configuration D (Proposed Framework)**: Full Cascaded Pipeline (Draft + Seq2Seq + Causal Humanizer).

### 5.2 Hypothesis Significance Testing
- Compute $p$-values using **paired two-sample $t$-tests** for normal distributions or **Wilcoxon signed-rank tests** for non-parametric metric distributions (e.g., token perplexity).
- Report effect sizes using **Cohen's $d$**:
  $$d = \frac{\bar{x}_1 - \bar{x}_2}{s_{\text{pooled}}}$$
  *(Ensure $d \ge 0.5$ for medium effect, $d \ge 0.8$ for large effect).*

---

## 6. Module 5: Adversarial Reviewer Stress-Testing
*(Synthesizes: `grill-me` Reviewer #2 Simulation)*

Before submitting or finalizing any manuscript, execute an adversarial critique answering these four mandatory reviewer attack vectors:

### Vector 1: The "Why Not a Single Model?" Attack
- *Objection*: "Why chain multiple models when a single larger model (e.g., Llama-3-70B or GPT-4o) can follow complex system prompts in one pass?"
- *Required Counter*: Prove **Hardware Accessibility & Autoregressive Invariance**. A single 70B model requires datacenter-class hardware ($>40$ GB VRAM). Furthermore, causal decoders still exhibit left-to-right probability signatures even when prompted to sound human; bidirectional encoder-decoders introduce structural syntax mutations that causal prompts cannot replicate.

### Vector 2: The "It's Just an Engineering Wrapper" Attack
- *Objection*: "Your architecture is merely an OS process orchestration script calling C++ binaries. Where is the algorithmic novelty?"
- *Required Counter*: Emphasize **Functional Decoupling as an Algorithmic Principle**. The contribution is not the subprocess call, but the proof that decoupling syntactic constituent restructuring from lexical cadence normalization achieves a Pareto-optimal trade-off between detector evasion and factual invariant preservation.

### Vector 3: The "Dual-Use & Academic Dishonesty" Attack
- *Objection*: "Does this tool simply help students evade plagiarism and AI checkers?"
- *Required Counter*: Frame the research as **Security Robustness & Vulnerability Assessment**. Statistical AI detectors are actively deployed in high-stakes settings (academia, journalism, law). Demonstrating their empirical failure against syntactic reordering exposes their technical unreliability, motivating robust watermarking over fragile post-hoc statistical classification.

### Vector 4: The Latency Penalty Attack
- *Objection*: "Cascading 3 models takes ~110s per generation. This is too slow for real-world usage."
- *Required Counter*: Differentiate between **Interactive Chat** and **High-Stakes Document Publishing**. For publication-grade essays, technical documentation, and reports, quality, privacy, and factual retention completely supersede real-time token streaming.

---

## 7. Module 6: Empirical Hardware & Codebase Telemetry
*(Synthesizes: `GitNexus`, System Hardware Telemetry)*

When benchmarking on constrained workstations (e.g., NVIDIA RTX 3050 6GB Laptop GPU):
1. **VRAM Memory Logging**: Continuously poll `nvidia-smi` at 100ms intervals during execution to capture:
   - Base system idle VRAM ($V_{\text{idle}}$).
   - Peak allocated VRAM per stage ($V_{\text{peak}}$).
   - Memory release valley between stages ($V_{\text{valley}}$) to verify zero driver leaks.
2. **Codebase AST Verification**: Interrogate AST symbols and call graphs via GitNexus CLI to ensure no cyclic imports or unhandled process locks exist in the execution path.
3. **Execution Timing**: Separate pure GPU matrix execution time ($t_{\text{compute}}$) from disk weight-loading time ($t_{\text{load}}$) and socket readiness polling time ($t_{\text{ready}}$).

---

## 8. Module 7: IEEE/ACM Publication Pipeline

Format the manuscript adhering to the **IEEE Standard 2-Column Section Structure**:

```
├── Title: Decoupled Multi-Model Cascades for Machine Text Naturalization
├── Abstract: (200-250 words: Context -> Problem -> Method -> Key Metric Results)
├── Section I: Introduction
│   ├── The memory wall of consumer hardware
│   ├── The failure modes of monolithic paraphrasers (hallucination vs. evasion)
│   └── 3 Core Technical Contributions
├── Section II: Related Work
│   ├── Statistical AI Detectors (Binoculars, DetectGPT, RoBERTa)
│   ├── Paraphrasing & Style Transfer (DIPPER, T5, Controllable Generation)
│   └── Edge Inference & Memory Offloading (FlexGen, DeepSpeed, llama.cpp)
├── Section III: Proposed Cascaded Framework
│   ├── Theoretical Motivation of Structural-Stylistic Decoupling
│   ├── Stage 1: Causal Drafting (Gemma 3 4B)
│   ├── Stage 2: Proposition-Bounded Seq2Seq Restructuring (FLAN-T5)
│   ├── Stage 3: Constrained Cadence Polishing (Humanizer 12B)
│   └── Asynchronous VRAM Multiplexing & Mutex Lifecycle
├── Section IV: Experimental Setup
│   ├── Benchmark Dataset (200 balanced cross-domain prompts)
│   ├── Baseline Systems (Raw Draft, T5-only, Humanizer-only)
│   └── Evaluation Metrics (AUROC, R_NER, P_Num, BERTScore, Burstiness, VRAM)
├── Section V: Results & Empirical Discussion
│   ├── 4-Way Ablation Analysis
│   ├── Factual Invariant Retention Trade-offs
│   ├── AI Detector Evasion Curves
│   └── Edge Hardware Footprint (RTX 3050 Telemetry)
├── Section VI: Limitations & Ethical Considerations
└── Section VII: Conclusion & Future Outlook
```
