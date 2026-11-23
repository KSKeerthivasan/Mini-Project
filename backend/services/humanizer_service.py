import asyncio
import os
import time
import json
import logging
import requests
from backend.services.llama_manager import llama_manager

logger = logging.getLogger("humanizer_service")
logger.setLevel(logging.INFO)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HUMANIZER_DIR = os.path.join(ROOT_DIR, "models", "humanizer")
HUMANIZER_MODEL_PATH = os.path.join(HUMANIZER_DIR, "humanizer-12b-Q4_K_M.gguf")
PROMPT_FORMAT_PATH = os.path.join(HUMANIZER_DIR, "prompt_format.json")

class HumanizerService:
    def __init__(self, model_path: str = HUMANIZER_MODEL_PATH, prompt_format_path: str = PROMPT_FORMAT_PATH):
        self.model_path = model_path
        self.prompt_format_path = prompt_format_path
        self.model_name = "Humanizer-12B"
        self._prompt_template = None
        self._load_prompt_format()

    def _load_prompt_format(self):
        if os.path.exists(self.prompt_format_path):
            try:
                with open(self.prompt_format_path, "r", encoding="utf-8") as f:
                    self._prompt_template = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load prompt_format.json: {e}")

        if not self._prompt_template:
            self._prompt_template = {
                "instr": (
                    "Rewrite the text below so it reads like a person wrote it, not a language model.\n\n"
                    "Reorganize it as you see fit. Vary sentence length on purpose. Cut hedging,\n"
                    "throat-clearing, and any sentence that only announces what comes next.\n"
                    "Prefer the concrete word over the abstract one. It is fine to sound uneven.\n\n"
                    "Every fact, number, unit, date, name and quotation must survive unchanged."
                ),
                "sep": "\n\n### Rewritten:\n\n"
            }

    def is_model_available(self) -> bool:
        return os.path.exists(self.model_path) and os.path.getsize(self.model_path) > 1_000_000_000

    def get_model_size_mb(self) -> float:
        if os.path.exists(self.model_path):
            return os.path.getsize(self.model_path) / (1024 * 1024)
        return 0.0

    async def humanize(self, draft_text: str) -> dict:
        """
        Stage 3: Natural writing refinement using Humanizer 12B.
        Follows the model's exact prompt format.
        """
        if not self.is_model_available():
            raise FileNotFoundError(f"Humanizer model file not found at {self.model_path}")

        instr = self._prompt_template.get("instr", "")
        sep = self._prompt_template.get("sep", "\n\n### Rewritten:\n\n")

        # Byte-for-byte exact official Humanizer-12B prompt format: instr + draft + sep
        full_prompt = f"{instr}\n\n{draft_text.strip()}{sep}"

        start_time = time.time()
        logger.info("Ensuring Humanizer-12B server is running with CPU/GPU hybrid offload (ngl=20)...")

        # RTX 3050 6GB VRAM + 16GB RAM: ngl=20 uses ~4.2GB VRAM, remaining layers in RAM
        base_url = await llama_manager.ensure_server(
            model_name=self.model_name,
            model_path=self.model_path,
            ngl=20,
            ctx=4096
        )

        logger.info("Calling llama-server completion API for Humanizer-12B...")
        # Reserve room for the completion in the model's 4k context.  This is
        # deliberately a hard, visible limit rather than silently losing the
        # end of a long draft.
        prompt_tokens_estimate = max(1, len(full_prompt) // 4)
        available_output_tokens = 4096 - prompt_tokens_estimate - 256
        if available_output_tokens < 128:
            raise ValueError("Draft is too long to refine in one pass. Split it into smaller sections first.")

        payload = {
            "prompt": full_prompt,
            "temperature": 0.85,
            "top_p": 0.95,
            "top_k": 40,
            "repeat_penalty": 1.10,
            "n_predict": min(max(int(len(draft_text.split()) * 2.2), 256), 3000, available_output_tokens),
            "stop": ["### Instruction:", "### Original:", "<eos>", "<|im_end|>", "### Rewritten:"]
        }

        resp = await asyncio.to_thread(requests.post, f"{base_url}/completion", json=payload, timeout=300)
        if resp.status_code != 200:
            raise RuntimeError(f"Humanizer completion error ({resp.status_code}): {resp.text}")

        data = resp.json()
        raw_content = data.get("content", "").strip()

        # Clean up any trailing markers
        if "### Rewritten:" in raw_content:
            raw_content = raw_content.split("### Rewritten:")[-1].strip()

        elapsed = time.time() - start_time
        logger.info(f"Humanizer-12B refinement complete in {elapsed:.2f}s ({len(raw_content)} chars)")

        return {
            "stage": "Humanizer 12B Refined Content",
            "model": "humanizer-12b-Q4_K_M.gguf",
            "content": raw_content,
            "inference_time_sec": round(elapsed, 2)
        }

humanizer_service = HumanizerService()

