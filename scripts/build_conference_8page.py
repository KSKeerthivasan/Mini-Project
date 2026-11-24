# -*- coding: utf-8 -*-
"""
Build Exact 8-Page IEEE Conference Paper with ALL 60 REFERENCES
Title: SLM-Based EdgeCascade: Dynamic Memory-Aware Orchestration of Heterogeneous Language Models for Fact-Grounded, Plagiarism-Resistant and Human-Like short Content Generation on Consumer Hardware
"""

import os
import sys
import re
import subprocess
import pypdf

SCRIPTS_DIR = os.path.abspath(os.path.dirname(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from render_tables_and_figures import match_and_render_block
import test_condensed_text
import all_60_references

ORIGINAL_PDF_1 = os.path.abspath("research-discovery/EdgeCascade_IEEE_Paper.pdf")
ORIGINAL_PDF_2 = os.path.abspath("research-discovery/EdgeCascade_IEEE_Paper_Humanized.pdf")

assert os.path.exists(ORIGINAL_PDF_1), f"Original PDF missing: {ORIGINAL_PDF_1}"
assert os.path.exists(ORIGINAL_PDF_2), f"Humanized PDF missing: {ORIGINAL_PDF_2}"

TARGET_MD = os.path.abspath("research-discovery/EdgeCascade_IEEE_Conference_A4_Manuscript.md")
TARGET_HTML = os.path.abspath("research-discovery/edge_cascade_ieee_conference_a4_print.html")
TARGET_PDF = os.path.abspath("research-discovery/EdgeCascade_IEEE_Conference_A4.pdf")

TITLE_TEXT = test_condensed_text.TITLE_TEXT
ABSTRACT_TEXT = test_condensed_text.ABSTRACT_TEXT
KEYWORDS_TEXT = test_condensed_text.KEYWORDS_TEXT

def get_condensed_markdown_with_60_refs():
    base_md = test_condensed_text.get_condensed_markdown_8p()
    all_refs_text = "\n## REFERENCES\n"
    for num, txt in all_60_references.ALL_60_REFS:
        cleaned = re.sub(r'\s+\d+\.\s*$', '', txt).strip()
        all_refs_text += f"[{num}] {cleaned}\n"
    if "## REFERENCES" in base_md:
        return re.sub(r'## REFERENCES.*', all_refs_text, base_md, flags=re.DOTALL)
    else:
        return base_md + all_refs_text

def build_conference_html(full_md, custom_css=""):
    refs_match = re.search(r'## REFERENCES(.*)', full_md, re.DOTALL)
    ref_items = []
    if refs_match:
        ref_text = refs_match.group(1).strip()
        items = re.findall(r'\[(\d+)\]\s+(.*?)(?=\[\d+\]|\Z)', ref_text, re.DOTALL)
        for num, txt in items:
            ref_items.append(f'<li class="ref-item">{txt.strip()}</li>')
    
    ref_list_html = "\n    ".join(ref_items)

    html_header = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{TITLE_TEXT}</title>
<style>
  @page {{
    size: A4 portrait; /* 210mm x 297mm */
    margin-top: 19mm;
    margin-bottom: 43mm;
    margin-left: 13mm;
    margin-right: 13mm;
  }}
  @page :first {{
    margin-top: 35mm;
    margin-bottom: 43mm;
    margin-left: 13mm;
    margin-right: 13mm;
  }}
  
  *, *:before, *:after {{
    box-sizing: border-box;
  }}
  
  body {{
    font-family: "Times New Roman", Times, serif;
    font-size: 9.3pt;
    line-height: 1.09;
    color: #000;
    margin: 0;
    padding: 0;
    text-align: justify;
    text-justify: inter-word;
    background: #fff;
  }}
  
  .header-container {{
    width: 100%;
    text-align: center;
    margin-bottom: 7pt;
  }}
  
  h1.conference-title {{
    font-size: 16.5pt;
    font-weight: normal;
    text-align: center;
    margin: 0 0 7pt 0;
    line-height: 1.15;
    font-family: "Times New Roman", Times, serif;
  }}
  
  .authors-container {{
    width: 100%;
    margin-bottom: 7pt;
  }}
  
  .authors-row {{
    display: flex;
    justify-content: space-around;
    margin-bottom: 4pt;
  }}
  
  .author-box {{
    flex: 1;
    text-align: center;
    font-size: 8.5pt;
    line-height: 1.12;
    padding: 0 3pt;
  }}
  
  .author-name {{
    font-weight: bold;
    font-size: 9pt;
  }}
  
  .author-dept {{
    font-size: 8pt;
    font-style: italic;
  }}
  
  .author-inst {{
    font-size: 8pt;
  }}
  
  .author-loc {{
    font-size: 8pt;
  }}
  
  .author-email {{
    font-family: "Courier New", Courier, monospace;
    font-size: 7.8pt;
  }}
  
  .abstract-keywords-box {{
    margin: 0 0 6pt 0;
    font-size: 8.5pt;
    line-height: 1.12;
    text-align: justify;
  }}
  
  .abstract-heading {{
    font-weight: bold;
    font-style: italic;
  }}
  
  .abstract-text {{
    font-weight: bold;
  }}
  
  .keywords-heading {{
    font-weight: bold;
    font-style: italic;
  }}
  
  .keywords-text {{
    font-weight: bold;
    font-style: italic;
  }}
  
  .two-column-layout {{
    column-count: 2;
    column-gap: 4.8mm;
    column-fill: balance;
    width: 100%;
  }}
  
  h2.heading-1 {{
    font-size: 9.2pt;
    font-weight: normal;
    text-align: center;
    text-transform: uppercase;
    font-variant: small-caps;
    margin: 3.5pt 0 1.2pt 0;
    break-after: avoid;
    break-inside: avoid;
  }}
  
  h3.heading-2 {{
    font-size: 8.8pt;
    font-weight: normal;
    font-style: italic;
    text-align: left;
    margin: 2.5pt 0 0.8pt 0;
    break-after: avoid;
    break-inside: avoid;
  }}
  
  p {{
    margin: 0 0 1.2pt 0;
    text-indent: 3mm;
  }}
  
  p.no-indent {{
    text-indent: 0;
  }}
  
  ul, ol {{
    margin: 1.5pt 0 2pt 1.1em;
    padding: 0;
  }}
  
  li {{
    margin-bottom: 0.8pt;
  }}
  
  .math-equation {{
    text-align: center;
    margin: 1.8pt 0;
    font-style: italic;
    font-size: 8.2pt;
  }}
  
  .wide-element {{
    column-span: all;
    width: 100%;
    margin: 2pt 0;
    page-break-inside: avoid;
    break-inside: avoid;
  }}
  
  .single-col-table {{
    column-span: none !important;
    width: 100%;
    margin: 1.5pt 0;
    page-break-inside: avoid;
    break-inside: avoid;
  }}
  
  .table-caption {{
    font-size: 7.2pt;
    font-variant: small-caps;
    text-align: center;
    margin-bottom: 1.2pt;
  }}
  
  .table-subtitle {{
    font-size: 6.8pt;
    font-style: italic;
    display: block;
    margin-top: 0.8pt;
  }}
  
  .figure-caption {{
    font-size: 7.2pt;
    text-align: justify;
    margin-top: 1.5pt;
    line-height: 1.1;
  }}
  
  .fig-label {{
    font-weight: normal;
  }}
  
  .ieee-table {{
    width: 100%;
    border-collapse: collapse;
    font-family: "Times New Roman", Times, serif;
    font-size: 5.8pt;
    line-height: 1.03;
    border-top: 1.1pt solid #000;
    border-bottom: 1.1pt solid #000;
    margin: 0 auto;
  }}
  
  .ieee-table th {{
    border-bottom: 0.7pt solid #000;
    padding: 1.0pt 1.5pt;
    font-weight: bold;
    text-align: left;
    background: #fafafa;
    vertical-align: bottom;
  }}
  
  .ieee-table td {{
    padding: 1.0pt 1.5pt;
    border-bottom: 0.3pt solid #e0e0e0;
    vertical-align: middle;
  }}
  
  .ieee-table tr.highlight-row td {{
    background: #f7fafc;
    border-bottom: 0.7pt solid #000;
  }}
  
  .diagram-container {{
    width: 100%;
    border: 0.5pt solid #999;
    padding: 2pt;
    background: #fafafa;
    border-radius: 2pt;
    margin: 1.5pt 0;
    page-break-inside: avoid;
  }}
  
  .algorithm-box {{
    width: 100%;
    border-top: 1.1pt solid #000;
    border-bottom: 1.1pt solid #000;
    padding: 1pt 0;
    margin: 2pt 0;
    font-size: 5.8pt;
    line-height: 1.05;
    page-break-inside: avoid;
    break-inside: avoid;
  }}
  
  .algo-title {{
    font-weight: bold;
    border-bottom: 0.5pt solid #000;
    padding-bottom: 1pt;
    margin-bottom: 1.5pt;
    text-transform: uppercase;
  }}
  
  .algo-line {{
    font-family: "Courier New", Courier, monospace;
    white-space: pre-wrap;
    padding-left: 3pt;
  }}
  
  .figure-graphic-box svg {{
    max-height: 95pt;
  }}
  
  .references-container {{
    margin-top: 2.5pt;
    font-size: 5.4pt;
    line-height: 1.01;
  }}
  
  ol.ref-list {{
    margin: 0;
    padding-left: 8pt;
  }}
  
  li.ref-item {{
    margin-bottom: 0.3pt;
    text-align: justify;
  }}
  {custom_css}
</style>
</head>
<body>

<div class="header-container">
  <h1 class="conference-title">{TITLE_TEXT}</h1>
  
  <div class="authors-container">
    <!-- Row 1: 3 Authors -->
    <div class="authors-row">
      <div class="author-box">
        <div class="author-name">Dr. J Varalakshmi</div>
        <div class="author-dept">Dept. of Artificial Intelligence & Data Science</div>
        <div class="author-inst">Saveetha Engineering College</div>
        <div class="author-loc">Chennai, India</div>
        <div class="author-email">varalakshmij@saveetha.ac.in</div>
      </div>
      <div class="author-box">
        <div class="author-name">Dr. Nalinipriya G (Mentor)</div>
        <div class="author-dept">Dept. of Artificial Intelligence & Data Science</div>
        <div class="author-inst">Saveetha Engineering College</div>
        <div class="author-loc">Tamil Nadu, India</div>
        <div class="author-email">xxx@gmail.com</div>
      </div>
      <div class="author-box">
        <div class="author-name">Vinay Suhirthan K S</div>
        <div class="author-dept">Dept. of Artificial Intelligence & Data Science</div>
        <div class="author-inst">Saveetha Engineering College</div>
        <div class="author-loc">Chennai, India</div>
        <div class="author-email">yyy@gmail.com</div>
      </div>
    </div>
    <!-- Row 2: 2 Authors (Centered) -->
    <div class="authors-row" style="max-width: 66%; margin: 0 auto;">
      <div class="author-box">
        <div class="author-name">Praveen J</div>
        <div class="author-dept">Dept. of Artificial Intelligence & Data Science</div>
        <div class="author-inst">Saveetha Engineering College</div>
        <div class="author-loc">Chennai, India</div>
        <div class="author-email">zzz@gmail.com</div>
      </div>
      <div class="author-box">
        <div class="author-name">Keerthivasan K S</div>
        <div class="author-dept">Dept. of Artificial Intelligence & Data Science</div>
        <div class="author-inst">Saveetha Engineering College</div>
        <div class="author-loc">Chennai, India</div>
        <div class="author-email">kskvasan2007@gmail.com</div>
      </div>
    </div>
  </div>
  
  <div class="abstract-keywords-box">
    <p class="no-indent">
      <span class="abstract-heading">Abstract</span>—<span class="abstract-text">{ABSTRACT_TEXT}</span>
    </p>
    <p class="no-indent" style="margin-top: 2.5pt;">
      <span class="keywords-heading">Keywords</span>—<span class="keywords-text">{KEYWORDS_TEXT}</span>
    </p>
  </div>
</div>

<div class="two-column-layout">
"""

    lines = full_md.split("\n")
    body_html_parts = []
    in_header_meta = True
    in_code_block = False
    code_block_content = []
    
    for line in lines:
        if in_header_meta:
            if line.startswith("## I. ") or line.startswith("## I. INTRODUCTION"):
                in_header_meta = False
            else:
                continue
                
        if line.startswith("## REFERENCES"):
            break
            
        if line.startswith("```"):
            if not in_code_block:
                in_code_block = True
                code_block_content = []
            else:
                in_code_block = False
                block_raw = "\n".join(code_block_content)
                rendered = match_and_render_block(block_raw)
                # Convert specific tables to single-column to prevent excessive column breaks
                if any(k in block_raw for k in ["TABLE II:", "TABLE III:", "TABLE V:", "TABLE VII:"]):
                    rendered = rendered.replace('class="table-wrapper wide-element"', 'class="table-wrapper single-col-table"')
                body_html_parts.append(rendered)
            continue
            
        if in_code_block:
            code_block_content.append(line)
            continue
            
        if line.startswith("## "):
            sec_title = line.replace("## ", "").strip()
            body_html_parts.append(f'<h2 class="heading-1">{sec_title}</h2>\n')
            continue
            
        if line.startswith("### "):
            subsec_title = line.replace("### ", "").strip()
            body_html_parts.append(f'<h3 class="heading-2">{subsec_title}</h3>\n')
            continue
            
        if line.startswith("$$") and line.endswith("$$"):
            math_content = line.strip("$").strip()
            body_html_parts.append(f'<div class="math-equation">{math_content}</div>\n')
            continue
            
        sline = line.strip()
        if not sline:
            continue
            
        line_clean = sline
        line_clean = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', line_clean)
        line_clean = re.sub(r'\*(.*?)\*', r'<em>\1</em>', line_clean)
        line_clean = re.sub(r'`(.*?)`', r'<code>\1</code>', line_clean)
        
        body_html_parts.append(f'<p>{line_clean}</p>\n')

    references_html = f"""
  <div class="references-container">
    <h2 class="heading-1">REFERENCES</h2>
    <ol class="ref-list">
      {ref_list_html}
    </ol>
  </div>
"""

    footer_html = f"""
  {references_html}
</div>

</body>
</html>
"""
    return html_header + "".join(body_html_parts) + footer_html

def compile_pdf(custom_css=""):
    md = get_condensed_markdown_with_60_refs()
    with open(TARGET_MD, "w", encoding="utf-8") as f:
        f.write(md)
        
    html = build_conference_html(md, custom_css)
    with open(TARGET_HTML, "w", encoding="utf-8") as f:
        f.write(html)
        
    chrome = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    html_url = "file:///" + TARGET_HTML.replace("\\", "/")
    cmd = [
        chrome,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        "--run-all-compositor-stages-before-draw",
        f"--print-to-pdf={TARGET_PDF}",
        html_url
    ]
    subprocess.run(cmd, capture_output=True, text=True)
    
    assert os.path.exists(TARGET_PDF), "Target PDF failed to generate!"
    reader = pypdf.PdfReader(TARGET_PDF)
    return len(reader.pages)

if __name__ == "__main__":
    pages = compile_pdf()
    print(f"Generated PDF Total Pages: {pages}")
