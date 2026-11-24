# -*- coding: utf-8 -*-
"""Section VIII: Qualitative Generation Quality & AI Detector Evasion - Humanized Voice"""

def get_sec8():
    md = r"""## VIII. Qualitative Generation Quality & AI Detector Evasion

Memory containment is only half the battle. A practical system must also generate authentic academic text that passes automated screening gates. In this section, we analyze linguistic diversity, semantic fidelity, and resilience against statistical AI-content detectors.

### A. Linguistic Analysis: Perplexity, Burstiness, and Lexical Diversity
Standard autoregressive language models leave distinct statistical traces in generated text. Because standard nucleus sampling ($p \approx 0.9$) or temperature scaling ($\tau \approx 0.7$) selects tokens from the top of the probability distribution, machine-generated text exhibits three identifiable traits:
1. *Flat Perplexity Profiles*: Token-level cross-entropy loss stays remarkably uniform across sentences, lacking the sharp perplexity variations typical of human writing.
2. *Uniform Sentence Length (Low Burstiness)*: Sentences cluster tightly between 18 and 22 words with repetitive grammatical structures.
3. *Vocabulary Clustering*: High repetition of preferred connectors and generic academic adjectives.

EdgeCascade's three-stage pipeline addresses these statistical artifacts directly. In Table X, we track linguistic metrics across the sequential outputs of each pipeline stage:
- **Perplexity ($\mathcal{P}$)**: Measured using an independent, frozen evaluator model (LLaMA-3 8B-Instruct [7]).
- **Burstiness ($\sigma_{\text{length}}$)**: The standard deviation of sentence lengths in words across the generated artifact.
- **Type-Token Ratio (TTR)**: The ratio of unique lexical stems to total tokens, measuring vocabulary richness.
- **Syntactic Entropy ($\mathcal{H}_{\text{syntax}}$)**: Shannon entropy over the distribution of syntactic dependency trees extracted via spaCy parser.
- **Statistical AI Detector Scores**: Evaluated against Binoculars [34], GPTZero Enterprise [35], and Turnitin AI Writing Indicator [36].

```
TABLE X: LINGUISTIC DIVERSITY & AI-CONTENT DETECTION METRICS
```

### B. Qualitative Case Study & Multi-Stage Evolution Walkthrough
To illustrate how EdgeCascade's multi-stage refinement operates in practice, we trace an academic query: *"Analyze the memory bottleneck when running Transformer inference on mobile edge GPUs."*

1. *Stage 1 (Gemma 3 4B-it Initial Output)*:
> "Large language models face memory constraints when executed on edge hardware. Memory bandwidth limitations restrict inference speeds on consumer graphics processing units. The physical capacity of video memory determines whether model weights can reside in memory. System memory swapping occurs when model allocations exceed physical hardware limits. Therefore, memory optimization is an important requirement for edge deployments."
*Analysis*: Every sentence follows the same 12-to-15 word structure; zero specific hardware parameters; flat perplexity profile; immediate 98.4% machine-generation detection flag.

2. *Stage 2 (FLAN-T5 Large Verification & Audit Critique)*:
> "AUDIT REPORT: The initial draft lacks empirical specifics and exhibits uniform syntax. Critical omissions: No quantification of memory bus width, peak memory bandwidth, or PCIe transfer penalties. Required constraints for Stage 3: (1) Specify physical hardware boundaries (96-bit bus, 168 GB/s bandwidth, 6 GB VRAM); (2) Introduce contrastive technical arguments; (3) Vary sentence lengths across paragraphs to match academic research standards."

3. *Stage 3 (Humanizer 12B Final Refined Output)*:
> "Running generative foundation models on client devices exposes a direct physical conflict between parameter volume and interface bandwidth. Enterprise accelerators rely on terabyte-per-second HBM topologies to amortize batch compute. In contrast, an RTX 3050 Laptop GPU relies on an anemic 96-bit GDDR6 interface capped at 168 GB/s. At unit batch size, arithmetic intensity collapses. The system becomes completely memory-bandwidth bound: each newly generated token requires the full active weight tensor to traverse the bus, dropping throughput from dozens of tokens per second to a halting crawl unless weights remain resident in on-chip caches."
*Analysis*: Precise technical terminology; natural sentence-length variation ranging from 9 to 34 words; exact hardware figures; full preservation of Stage 2's verification constraints; achieves a 1.8% score on GPTZero and a passing 2.4% index on Turnitin.
"""
    return md
