# -*- coding: utf-8 -*-
"""
Full Assembly Script for Expanded IEEE Journal Manuscript & Camera-Ready HTML/PDF.
Converts Markdown and modular sections into:
1. research-discovery/edge_cascade_ieee_manuscript.md (clean academic Markdown)
2. research-discovery/edge_cascade_ieee_print.html (publication-grade IEEE two-column HTML with zero cut-off tables)
3. research-discovery/EdgeCascade_IEEE_Paper.pdf (rendered via Headless Chrome)
"""
import sys
import os
import re
import subprocess
import pypdf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sections.sec1_intro import get_sec1
from sections.sec2_related import get_sec2
from sections.sec3_theory import get_sec3
from sections.sec4_arch import get_sec4
from sections.sec5_method import get_sec5
from sections.sec6_results import get_sec6
from sections.sec7_ablations import get_sec7
from sections.sec8_qualitative import get_sec8
from sections.sec9_discussion import get_sec9
from sections.sec10_conclusion import get_sec10
from sections.references import get_references
from render_tables_and_figures import match_and_render_block

def build_markdown():
    title = "# EdgeCascade: Dynamic Memory-Multiplexed Orchestration of Heterogeneous LLM Ensembles on 6 GB Consumer GPUs\n\n"
    author_block = "**Keerthivasan K. S.**  \n*Department of Computer Science and Engineering, Research Laboratory for Edge Intelligence Systems*\n\n"
    
    abstract_text = (
        "### Abstract\n"
        "*Modern frontier large language models (LLMs) increasingly exceed the physical memory "
        "boundaries of consumer-grade edge workstations. While cloud-hosted compound AI architectures "
        "rely on multi-GPU server clusters with hundreds of gigabytes of aggregate High Bandwidth "
        "Memory (HBM), local edge deployments are strictly bounded by hardware cost, thermal dissipation, "
        "and physical memory capacity. Consumer computing hardware, typically equipped with 6 GB GDDR6 "
        "dedicated video RAM (VRAM), suffers catastrophic Out-Of-Memory (OOM) failures or severe "
        "operating system memory swapping when attempting to execute multi-model ensembles. In this paper, "
        "we introduce EdgeCascade, an architectural orchestration framework that enables the deterministic, "
        "low-latency execution of a 17.2-billion aggregate parameter heterogeneous compound AI system on a "
        "single 6 GB consumer GPU (NVIDIA GeForce RTX 3050 Laptop). EdgeCascade resolves the physical memory "
        "wall through three core mechanisms: (1) a mutex-guarded dynamic memory-multiplexing lifecycle protocol "
        "that enforces mutually exclusive process instantiation across sequential execution phases; (2) an "
        "asymmetric hybrid offloading allocation that pins intermediate deterministic verification workloads "
        "to the host CPU while dynamically partitioning heavy generative workloads between GPU VRAM and system "
        "RAM; and (3) a proactive Windows Display Driver Model (WDDM 3.1) memory release pacing engine that "
        "eliminates driver allocation race conditions and asynchronous deallocation lag. In empirical "
        "evaluations across Short (128 tokens), Medium (256 tokens), and Standard (512 tokens) workloads, "
        "EdgeCascade achieves an aggregate peak VRAM footprint of 4,287.0 MiB, preserving a 1,857.0 MiB "
        "(30.2%) physical safety cushion beneath the 6,144 MiB hardware boundary with a 0.0% OOM failure "
        "rate across all trials. EdgeCascade completes the end-to-end multi-stage cascade with a total execution "
        "latency of 254.35s ± 55.84s, outperforming state-of-the-art offloading baselines that either crash "
        "or degrade into catastrophic paging thrashing. Furthermore, qualitative analysis reveals that the "
        "multi-stage verification and refinement pipeline systematically elevates syntactic diversity and "
        "reduces machine-generated n-gram artifacts, reducing statistical AI-detector confidence from 98.4% "
        "to 1.8% while preserving strict semantic faithfulness.*\n\n"
        "**Keywords**—*Edge AI, Heterogeneous LLM Ensembles, Memory Multiplexing, Resource-Constrained Inference, "
        "Windows Display Driver Model (WDDM), Consumer GPU Optimization, Compound AI Systems, Quantization.*\n\n"
    )
    
    body_sections = [
        get_sec1(),
        get_sec2(),
        get_sec3(),
        get_sec4(),
        get_sec5(),
        get_sec6(),
        get_sec7(),
        get_sec8(),
        get_sec9(),
        get_sec10()
    ]
    
    refs = get_references()
    ref_text = "## References\n\n" + "\n\n".join(refs) + "\n"
    
    return title + author_block + abstract_text + "\n\n".join(body_sections) + "\n\n" + ref_text

def format_equation(eq_str):
    s = eq_str.strip()
    match = re.search(r'\\quad\s*\(([0-9]+)\)$', s)
    eq_num = ""
    if match:
        eq_num = f'({match.group(1)})'
        s = s[:match.start()].strip()
    
    replacements = [
        (r'\\left\(', '('),
        (r'\\right\)', ')'),
        (r'\\left\[', '['),
        (r'\\right\]', ']'),
        (r'\\blacksquare', '&#9632;'),
        (r'\\le', '&le;'),
        (r'\\ge', '&ge;'),
        (r'\\pm', '&plusmn;'),
        (r'\\times', '&times;'),
        (r'\\approx', '&asymp;'),
        (r'\\tau_d', '<i>&tau;</i><sub>d</sub>'),
        (r'\\tau_\\text\{settle\}', '<i>&tau;</i><sub>settle</sub>'),
        (r'\\tau_\{\\text\{settle\}\}', '<i>&tau;</i><sub>settle</sub>'),
        (r'\\tau', '<i>&tau;</i>'),
        (r'\\max_\{[^}]*\}', 'max'),
        (r'\\max', 'max'),
        (r'\\sum_\{[^}]*\}', '&Sigma;'),
        (r'\\exp', 'exp'),
        (r'\\ln', 'ln'),
        (r'\\dots', '...'),
        (r'\\cdot', '&middot;'),
        (r'\\_', '_'),
        (r'\\text\{([^}]+)\}', r'<span>\1</span>'),
        (r'_\{([^}]+)\}', r'<sub>\1</sub>'),
        (r'\^\{([^}]+)\}', r'<sup>\1</sup>'),
        (r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1 / \2)'),
        (r'\\mathcal\{([A-Za-z])\}', r'<i>\1</i>'),
        (r'\\([a-zA-Z]+)', r'\1')
    ]
    for pattern, repl in replacements:
        s = re.sub(pattern, repl, s)
        
    s = s.replace('$', '').strip()
    if eq_num:
        return f'<div class="equation-box"><span class="eq-content">{s}</span><span class="eq-num">{eq_num}</span></div>'
    return f'<div class="equation-box"><span class="eq-content">{s}</span></div>'

def format_inline_math(text):
    def replace_math(m):
        raw = m.group(1)
        replacements = [
            (r'\\left\(', '('),
            (r'\\right\)', ')'),
            (r'\\left\[', '['),
            (r'\\right\]', ']'),
            (r'\\blacksquare', '&#9632;'),
            (r'\\le', '&le;'),
            (r'\\ge', '&ge;'),
            (r'\\pm', '&plusmn;'),
            (r'\\times', '&times;'),
            (r'\\approx', '&asymp;'),
            (r'\\tau_d', '&tau;<sub>d</sub>'),
            (r'\\tau_{\text{settle}}', '&tau;<sub>settle</sub>'),
            (r'\\tau', '&tau;'),
            (r'\\max_\{[^}]*\}', 'max'),
            (r'\\max', 'max'),
            (r'\\sum_\{[^}]*\}', '&Sigma;'),
            (r'\\exp', 'exp'),
            (r'\\ln', 'ln'),
            (r'\\dots', '...'),
            (r'\\cdot', '&middot;'),
            (r'\\_', '_'),
            (r'\\text\{([^}]+)\}', r'<span>\1</span>'),
            (r'_\{([^}]+)\}', r'<sub>\1</sub>'),
            (r'\^\{([^}]+)\}', r'<sup>\1</sup>'),
            (r'_([a-zA-Z0-9])', r'<sub>\1</sub>'),
            (r'\\mathcal\{([A-Za-z])\}', r'<i>\1</i>'),
            (r'\\([a-zA-Z]+)', r'\1')
        ]
        for p, r in replacements:
            raw = re.sub(p, r, raw)
        return f'<i>{raw}</i>'
    
    text = text.replace(r'\blacksquare', '&#9632;')
    return re.sub(r'\$([^$]+)\$', replace_math, text)

def build_html(full_md):
    refs = get_references()
    ref_list_html = "".join([f'<li class="ref-item">{r}</li>\n' for r in refs])
    
    html_header = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>EdgeCascade: Dynamic Memory-Multiplexed Orchestration of Heterogeneous LLM Ensembles on 6 GB Consumer GPUs</title>
<style>
  @page {
    size: letter;
    margin: 0.70in 0.60in 0.80in 0.60in;
    @bottom-right {
      content: counter(page);
    }
  }
  
  *, *:before, *:after {
    box-sizing: border-box;
  }
  
  body {
    font-family: "Times New Roman", Times, serif;
    font-size: 10pt;
    line-height: 1.18;
    color: #000;
    margin: 0;
    padding: 0;
    text-align: justify;
    text-justify: inter-word;
    background: #fff;
  }
  
  .header-container {
    width: 100%;
    text-align: center;
    margin-bottom: 12pt;
    padding-bottom: 4pt;
    border-bottom: 0.5pt solid #ccc;
  }
  
  h1.paper-title {
    font-size: 20pt;
    font-weight: bold;
    margin: 0 0 8pt 0;
    line-height: 1.15;
  }
  
  .author-block {
    font-size: 10.5pt;
    margin-bottom: 10pt;
  }
  
  .author-name {
    font-weight: bold;
    font-size: 11pt;
  }
  
  .author-affil {
    font-style: italic;
    font-size: 9.5pt;
    color: #222;
  }
  
  .abstract-keywords-box {
    margin: 0 0.25in 12pt 0.25in;
    font-size: 9pt;
    line-height: 1.20;
    text-align: justify;
  }
  
  .abstract-heading {
    font-weight: bold;
    font-style: italic;
    display: inline;
  }
  
  .abstract-text {
    font-weight: bold;
    font-style: italic;
    display: inline;
  }
  
  .keywords-block {
    margin-top: 4pt;
    font-size: 9pt;
  }
  
  .keywords-heading {
    font-weight: bold;
    font-style: italic;
    display: inline;
  }
  
  .keywords-text {
    font-style: italic;
    display: inline;
  }
  
  .two-column-body {
    column-count: 2;
    column-gap: 0.24in;
    column-fill: balance;
  }
  
  .wide-element {
    column-span: all;
    width: 100%;
    margin: 10pt 0;
    page-break-inside: avoid;
    break-inside: avoid;
  }
  
  h2.sec-heading {
    font-size: 10pt;
    font-weight: bold;
    text-align: center;
    text-transform: uppercase;
    margin: 10pt 0 4pt 0;
    break-after: avoid;
  }
  
  h3.subsec-heading {
    font-size: 9.5pt;
    font-weight: bold;
    font-style: italic;
    margin: 7pt 0 2pt 0;
    break-after: avoid;
  }
  
  p {
    margin: 0 0 5pt 0;
    text-indent: 1.2em;
  }
  
  p.no-indent {
    text-indent: 0;
  }
  
  .equation-box {
    width: 100%;
    margin: 5pt 0;
    font-size: 9.2pt;
    text-align: center;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  
  .eq-content {
    margin: 0 auto;
  }
  
  .eq-num {
    font-style: normal;
    font-weight: normal;
    text-align: right;
  }
  
  /* IEEE Table Styles */
  .table-wrapper {
    margin: 9pt 0;
    page-break-inside: avoid;
    break-inside: avoid;
  }
  
  .table-caption {
    font-family: "Times New Roman", Times, serif;
    font-size: 8pt;
    font-weight: bold;
    text-align: center;
    text-transform: uppercase;
    margin-bottom: 2pt;
    letter-spacing: 0.3px;
  }
  
  .table-subtitle {
    font-size: 7.5pt;
    font-weight: normal;
    letter-spacing: 0.5px;
  }
  
  table.ieee-table {
    width: 100%;
    border-collapse: collapse;
    font-family: "Times New Roman", Times, serif;
    font-size: 7.5pt;
    line-height: 1.15;
    border-top: 1.2pt solid #000;
    border-bottom: 1.2pt solid #000;
    margin: 0 auto;
  }
  
  table.ieee-table th {
    border-bottom: 0.8pt solid #000;
    padding: 3.5pt 4pt;
    font-weight: bold;
    text-align: left;
    background: #fafafa;
    vertical-align: bottom;
  }
  
  table.ieee-table td {
    padding: 3pt 4pt;
    border-bottom: 0.3pt solid #e0e0e0;
    vertical-align: middle;
  }
  
  table.ieee-table tr.highlight-row td {
    background: #f7fafc;
    border-bottom: 0.8pt solid #000;
  }
  
  /* Figure Styles */
  .figure-wrapper {
    margin: 9pt 0;
    page-break-inside: avoid;
    break-inside: avoid;
    text-align: center;
  }
  
  .figure-graphic-box {
    width: 100%;
    border: 0.8pt solid #cbd5e0;
    background: #fff;
    padding: 6pt;
    border-radius: 2pt;
  }
  
  .figure-caption {
    font-family: "Times New Roman", Times, serif;
    font-size: 8pt;
    font-style: italic;
    text-align: center;
    margin-top: 4pt;
    line-height: 1.15;
  }
  
  /* Algorithm Styles */
  .algorithm-container {
    width: 100%;
    border-top: 1.2pt solid #000;
    border-bottom: 1.2pt solid #000;
    background: #fdfdfd;
    padding: 5pt 7pt;
    margin: 8pt 0;
    font-family: "Times New Roman", Times, serif;
    font-size: 7.5pt;
    line-height: 1.20;
    page-break-inside: avoid;
    break-inside: avoid;
  }
  
  .algorithm-header {
    font-weight: bold;
    font-size: 8pt;
    margin-bottom: 2pt;
  }
  
  .algorithm-divider {
    border-bottom: 0.6pt solid #000;
    margin-bottom: 3pt;
  }
  
  .algorithm-body {
    padding: 2pt 0;
  }
  
  .alg-line {
    font-family: "Courier New", Courier, monospace;
    font-size: 7.2pt;
    line-height: 1.25;
  }
  
  .alg-num {
    color: #4a5568;
    display: inline-block;
    width: 18pt;
    font-family: "Courier New", Courier, monospace;
  }
  
  .alg-comment {
    color: #718096;
    font-style: italic;
  }
  
  .proof-box {
    font-size: 9.2pt;
    margin: 4pt 0 6pt 0;
    padding-left: 6pt;
    border-left: 1.5pt solid #555;
  }
  
  ol.references-list {
    list-style-type: none;
    padding-left: 0;
    margin: 0;
    font-size: 8pt;
    line-height: 1.14;
  }
  
  li.ref-item {
    margin-bottom: 3.5pt;
    text-indent: -1.8em;
    padding-left: 1.8em;
    text-align: justify;
  }
  
  blockquote {
    font-size: 8.8pt;
    font-style: italic;
    margin: 4pt 8pt;
    padding: 3pt 6pt;
    border-left: 2pt solid #888;
    background: #f7f7f7;
  }
</style>
</head>
<body>

<div class="header-container">
  <h1 class="paper-title">EdgeCascade: Dynamic Memory-Multiplexed Orchestration of Heterogeneous LLM Ensembles on 6 GB Consumer GPUs</h1>
  <div class="author-block">
    <div class="author-name">Keerthivasan K. S.</div>
    <div class="author-affil">Department of Computer Science and Engineering, Research Laboratory for Edge Intelligence Systems</div>
  </div>
  
  <div class="abstract-keywords-box">
    <div class="abstract-heading">Abstract—</div>
    <div class="abstract-text">
      Modern frontier large language models (LLMs) increasingly exceed the physical memory boundaries of consumer-grade edge workstations. While cloud-hosted compound AI architectures rely on multi-GPU server clusters with hundreds of gigabytes of aggregate High Bandwidth Memory (HBM), local edge deployments are strictly bounded by hardware cost, thermal dissipation, and physical memory capacity. Consumer computing hardware, typically equipped with 6 GB GDDR6 dedicated video RAM (VRAM), suffers catastrophic Out-Of-Memory (OOM) failures or severe operating system memory swapping when attempting to execute multi-model ensembles. In this paper, we introduce EdgeCascade, an architectural orchestration framework that enables the deterministic, low-latency execution of a 17.2-billion aggregate parameter heterogeneous compound AI system on a single 6 GB consumer GPU (NVIDIA GeForce RTX 3050 Laptop). EdgeCascade resolves the physical memory wall through three core mechanisms: (1) a mutex-guarded dynamic memory-multiplexing lifecycle protocol that enforces mutually exclusive process instantiation across sequential execution phases; (2) an asymmetric hybrid offloading allocation that pins intermediate deterministic verification workloads to the host CPU while dynamically partitioning heavy generative workloads between GPU VRAM and system RAM; and (3) a proactive Windows Display Driver Model (WDDM 3.1) memory release pacing engine that eliminates driver allocation race conditions and asynchronous deallocation lag. In empirical evaluations across Short (128 tokens), Medium (256 tokens), and Standard (512 tokens) workloads, EdgeCascade achieves an aggregate peak VRAM footprint of 4,287.0 MiB, preserving a 1,857.0 MiB (30.2%) physical safety cushion beneath the 6,144 MiB hardware boundary with a 0.0% OOM failure rate across all trials. EdgeCascade completes the end-to-end multi-stage cascade with a total execution latency of 254.35s ± 55.84s, outperforming state-of-the-art offloading baselines that either crash or degrade into catastrophic paging thrashing. Furthermore, qualitative analysis reveals that the multi-stage verification and refinement pipeline systematically elevates syntactic diversity and reduces machine-generated n-gram artifacts, reducing statistical AI-detector confidence from 98.4% to 1.8% while preserving strict semantic faithfulness.
    </div>
    <div class="keywords-block">
      <div class="keywords-heading">Index Terms—</div>
      <div class="keywords-text">Edge AI, Heterogeneous LLM Ensembles, Memory Multiplexing, Resource-Constrained Inference, Windows Display Driver Model (WDDM), Consumer GPU Optimization, Compound AI Systems, Quantization.</div>
    </div>
  </div>
</div>

<div class="two-column-body">
"""

    raw_sections = [
        get_sec1(),
        get_sec2(),
        get_sec3(),
        get_sec4(),
        get_sec5(),
        get_sec6(),
        get_sec7(),
        get_sec8(),
        get_sec9(),
        get_sec10()
    ]
    
    body_html_parts = []
    for s in raw_sections:
        lines = s.splitlines()
        in_code = False
        code_block = []
        for line in lines:
            line_str = line.strip()
            if line_str.startswith('```'):
                if in_code:
                    in_code = False
                    code_content = "\n".join(code_block)
                    rendered_block = match_and_render_block(code_content)
                    body_html_parts.append(rendered_block + '\n')
                    code_block = []
                else:
                    in_code = True
                    code_block = []
                continue
            
            if in_code:
                code_block.append(line)
                continue
            
            if line_str.startswith('## '):
                heading = line_str.replace('## ', '')
                body_html_parts.append(f'<h2 class="sec-heading">{heading}</h2>\n')
            elif line_str.startswith('### '):
                heading = line_str.replace('### ', '')
                body_html_parts.append(f'<h3 class="subsec-heading">{heading}</h3>\n')
            elif line_str.startswith('$$'):
                eq_text = line_str.replace('$$', '').strip()
                body_html_parts.append(format_equation(eq_text) + '\n')
            elif line_str.startswith('> '):
                quote = line_str.replace('> ', '').strip()
                body_html_parts.append(f'<blockquote>{format_inline_math(quote)}</blockquote>\n')
            elif line_str.startswith('**Lemma') or line_str.startswith('**Theorem'):
                body_html_parts.append(f'<div class="proof-box"><strong>{format_inline_math(line_str)}</strong></div>\n')
            elif line_str.startswith('*Proof*'):
                body_html_parts.append(f'<p class="proof-box"><em>{format_inline_math(line_str)}</em></p>\n')
            elif line_str:
                formatted_p = format_inline_math(line_str)
                body_html_parts.append(f'<p>{formatted_p}</p>\n')

    footer_html = f"""
  <h2 class="sec-heading">References</h2>
  <ol class="references-list">
    {ref_list_html}
  </ol>
</div>

</body>
</html>
"""
    return html_header + "".join(body_html_parts) + footer_html

def main():
    print("Building full markdown manuscript...")
    md = build_markdown()
    md_path = os.path.abspath("research-discovery/edge_cascade_ieee_manuscript.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Wrote Markdown: {md_path} ({len(md.split())} words, {len(md.splitlines())} lines)")
    
    print("Building full HTML print template...")
    html = build_html(md)
    html_path = os.path.abspath("research-discovery/edge_cascade_ieee_print.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote HTML: {html_path} ({len(html.split())} words, {len(html.splitlines())} lines)")
    
    print("Compiling PDF via Headless Chrome...")
    chrome = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    pdf_path = os.path.abspath("research-discovery/EdgeCascade_IEEE_Paper.pdf")
    html_url = "file:///" + html_path.replace("\\", "/")
    
    cmd = [
        chrome,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        "--run-all-compositor-stages-before-draw",
        f"--print-to-pdf={pdf_path}",
        html_url
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print("Chrome execution finished with exit code:", res.returncode)
    
    reader = pypdf.PdfReader(pdf_path)
    num_pages = len(reader.pages)
    print(f"Generated PDF total pages: {num_pages}")
    
    ref_start_page = 0
    for i, p in enumerate(reader.pages):
        text = p.extract_text() or ""
        if "REFERENCES" in text.upper():
            ref_start_page = i + 1
            break
            
    print(f"References start on page: {ref_start_page}")
    body_pages = ref_start_page - 1 if ref_start_page > 0 else num_pages
    print(f"Technical body pages (excluding references): {body_pages} pages")

if __name__ == "__main__":
    main()
