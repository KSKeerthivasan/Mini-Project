# LocalAIContentStudio (ChatGPT UI) 🚀

**LocalAIContentStudio** is a high-performance, 100% offline, locally running multi-model AI chatbot and refinement studio designed for Windows 11 with NVIDIA RTX GPU acceleration via `llama.cpp`.

Featuring a sleek **ChatGPT conversational interface**, you can ask any question or request content, and have it generated, restructured, and humanized through a 3-stage local AI pipeline.

---

## 🌟 Architecture & Workflow

When you ask a question or submit a prompt, the pipeline executes sequentially through three specialized local models:

```
[ User Prompt / Question ]
       │
       ▼
1. Gemma 3 4B-it (Q4_K_M)
   └─ Generates comprehensive, fact-grounded initial answer/draft
       │
       ▼ (Model unloaded, VRAM cleared)
2. FLAN-T5 Paraphraser (Fine-Tuned Checkpoint)
   └─ Restructures sentences, improves syntax, preserves facts & entities
       │
       ▼ (Model unloaded, VRAM cleared)
3. Humanizer 12B (Q4_K_M)
   └─ Cuts AI hedging, varies sentence cadence, delivers natural conversational voice
       │
       ▼
[ Final Refined Answer in ChatGPT UI ]
```

### 💬 ChatGPT Interface Features
- **Sidebar & History**: Left sidebar with `+ New chat`, persistent conversation history in `localStorage`, and live RTX 3050 VRAM monitor.
- **Reasoning & Stages Trace**: Expandable thought process pill showing exact execution times and raw outputs for Gemma, FLAN-T5, and Humanizer.
- **Stage Comparison Modal**: Side-by-side view comparing Stage 1 (Gemma Draft) vs Stage 2 (FLAN-T5 Restructured) vs Stage 3 (Humanizer Final).
- **Sticky Input Box**: Auto-growing textarea with `Enter` to send, `Shift+Enter` for newline, and prompt suggestion cards.
- **Copy & Export**: One-click copy and Markdown (`.md`) download for answers.

### Key Technical Safeguards
- **Zero Cloud / Zero API Costs**: Runs entirely offline on localhost.
- **Unified Engine**: Powered exclusively by `llama.cpp` (`llama-server` and `llama-cli`).
- **Dynamic VRAM Management**: Automatically unloads models between pipeline stages to guarantee stability on 6 GB VRAM GPUs.
- **Preservation of Factual Meaning**: Strict instructions preserve names, dates, numbers, units, and quotations.

---

## 💻 Hardware Configuration & Requirements

- **Operating System**: Windows 11 (64-bit)
- **GPU**: NVIDIA RTX 3050 Laptop GPU (6 GB VRAM, CUDA 12.4+ supported)
- **System RAM**: 16 GB RAM
- **Storage**: ~20 GB free space on local drive
- **Runtime Dependencies**:
  - Python 3.12+ (x64)
  - Node.js 20+ & npm

---

## 📁 Project Folder Structure

```
LocalAIContentStudio/
├── frontend/                     # React 19 + TypeScript + Vite UI
│   ├── src/                      # UI Components, styles & state
│   ├── package.json
│   └── vite.config.ts            # Proxy configured for /api -> 127.0.0.1:8000
│
├── backend/                      # FastAPI orchestration backend
│   ├── services/
│   │   ├── llama_manager.py      # llama.cpp server lifecycle & VRAM offload manager
│   │   ├── gemma_service.py      # Stage 1: Gemma 3 4B generator
│   │   ├── gemini_service.py     # Optional cloud Gemini generator for Stage 1
│   │   ├── flan_t5_service.py    # Stage 2: FLAN-T5 paraphraser
│   │   └── humanizer_service.py  # Stage 3: Humanizer 12B natural refinement
│   ├── main.py                   # FastAPI REST API endpoints
│   ├── requirements.txt
│   └── .venv/                    # Isolated Python virtual environment
│
├── models/
│   ├── gemma/
│   │   └── gemma-3-4b-it-Q4_K_M.gguf
│   ├── flan-t5/
│   │   └── flan-t5-large-paraphraser-Q4_K_M.gguf
│   └── humanizer/
│       ├── humanizer-12b-Q4_K_M.gguf
│       └── prompt_format.json
│
├── runtime/
│   └── llama.cpp/                # CUDA-accelerated llama-server.exe & llama-cli.exe
│
├── scripts/
│   ├── convert_flan_t5.py        # Official llama.cpp converter for FLAN-T5
│   └── download_gemma.py         # Gemma downloader
│
├── temp/                         # Temporary conversion workspace
├── logs/                         # Server & inference logs
├── start.bat                     # One-click Windows launcher
└── README.md
```

---

## 🧠 Model Details & Licenses

| Stage | Model Name | Role | Source / HF Repo | Quantization | Engine |
|---|---|---|---|---|---|
| **Stage 1** | **Gemma 3 4B-it** | Initial Generation | [`ggml-org/gemma-3-4b-it-GGUF`](https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF) | Q4_K_M (~2.48 GB) | `llama-server` (CUDA) |
| **Stage 2** | **FLAN-T5-Paraphraser** | Restructuring & Paraphrasing | [`alykassem/FLAN-T5-Paraphraser`](https://huggingface.co/alykassem/FLAN-T5-Paraphraser) | Q4_K_M / F16 | `llama-cli` |
| **Stage 3** | **Humanizer 12B** | Natural Writing Refinement | [`jialinyyzz/humanizer-GGUF`](https://huggingface.co/jialinyyzz/humanizer-GGUF) | Q4_K_M (~7.62 GB) | `llama-server` (Hybrid CUDA/CPU) |

---

## ⚙️ GGUF Conversion Procedure (FLAN-T5 Paraphraser)

The exact fine-tuned `alykassem/FLAN-T5-Paraphraser` checkpoint is converted into GGUF using the official `llama.cpp` tools:

1. **Download Raw Checkpoint**:
   - `pytorch_model.bin`, `config.json`, `tokenizer.json`, `special_tokens_map.json` from `alykassem/FLAN-T5-Paraphraser`.
   - Matching SentencePiece tokenizer `spiece.model` from `google/flan-t5-large`.
2. **Execute Conversion**:
   ```bash
   python temp/llama_repo/convert_hf_to_gguf.py temp/flan_t5_raw --outfile temp/flan-t5-paraphraser-f16.gguf --outtype f16
   ```
3. **Quantize**:
   ```bash
   runtime\llama.cpp\llama-quantize.exe temp\flan-t5-paraphraser-f16.gguf models\flan-t5\flan-t5-large-paraphraser-Q4_K_M.gguf Q4_K_M
   ```
4. **Automated Script**:
   Run `scripts/convert_flan_t5.py` to automate this complete process, verify inference with `llama-cli`, and automatically clean up temporary conversion weights.

---

## 🚀 Run the Application

### Quick start

1. Confirm the model files exist in `models\gemma`, `models\flan-t5`, and `models\humanizer`.
2. Double-click `start.bat` from the project root, or run this in PowerShell:

   ```powershell
   .\start.bat
   ```

3. Keep the two terminal windows open and use the browser page opened at `http://localhost:5173`.
4. Enter a prompt, select the content type and length from the settings panel, then send it. The answer moves through Gemma, FLAN-T5, and Humanizer automatically.

The launcher starts the FastAPI backend on `http://127.0.0.1:8000`, the React frontend on `http://localhost:5173`, and loads each model only while it is needed.

### Optional: use Gemini for Stage 1

Copy `.env.example` to `.env` in the project root and set the provider you want:

```dotenv
# Gemini cloud for Stage 1
MODEL1=true
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.5-flash
```

Set `MODEL1=false` (or `model1=false`) to use the existing local Gemma 3 4B model instead. Restart the backend after changing `.env`. The API key stays server-side and is ignored by Git.

### Run with Docker and NVIDIA GPU

Docker includes the three GGUF model families, compiles Linux CUDA-enabled `llama.cpp`, builds the React client, and serves the complete app on port `8000`.

1. Install Docker Desktop with the NVIDIA Container Toolkit / GPU support enabled.
2. Keep your Stage 1 settings in the root `.env` file (copy `.env.example` if needed).
3. Build and start:

   ```powershell
   docker compose up --build
   ```

4. Open `http://localhost:8000`. The container requires an NVIDIA GPU and `docker compose` GPU support; the models are large, so the first build creates a very large local image.

### First-time setup or virtual-environment repair

Install **Python 3.12 (64-bit)** and **Node.js 20+**. From the project root, run:

```powershell
python -m venv backend\.venv
.\backend\.venv\Scripts\python.exe -m pip install --upgrade pip
.\backend\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
Set-Location frontend
npm install
Set-Location ..
```

Then launch with `./start.bat`. The Python packages are large because local inference tooling includes PyTorch and Transformers; installation can take a while.

### Manual start (useful for debugging)

Open two PowerShell windows from the project root.

**Terminal 1 — backend**

```powershell
.\backend\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

**Terminal 2 — frontend**

```powershell
Set-Location frontend
npm run dev
```

Open `http://localhost:5173` in your browser.

### Output format

For multi-section answers, the app asks the models for clean Markdown: one title for articles, logical section headings, short paragraphs, and proper bulleted or numbered lists. The chat UI renders these headings, lists, quotes, bold text, and inline code directly. Copying or downloading keeps the Markdown source.

---

## 🛠️ Troubleshooting

- **Out of VRAM Error**: Click **"Unload Memory"** in the top navigation bar to terminate any active `llama-server` and force Windows to reclaim GPU VRAM.
- **Port Conflict (8000 or 5173 in use)**: Ensure no other web server is occupying ports 8000, 8080, or 5173.
- **CUDA Detection Failure**: Verify that `nvidia-smi` reports your RTX 3050 and that the CUDA DLLs in `runtime\llama.cpp` are intact.

