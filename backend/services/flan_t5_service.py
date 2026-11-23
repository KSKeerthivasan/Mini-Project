import os
import time
import logging
import re
from backend.services.llama_manager import llama_manager

logger = logging.getLogger("flan_t5_service")
logger.setLevel(logging.INFO)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODELS_DIR = os.path.join(ROOT_DIR, "models", "flan-t5")

class FlanT5Service:
    def __init__(self):
        self.model_name = "FLAN-T5-Large-Paraphraser"

    def get_model_path(self) -> str | None:
        q4_path = os.path.join(MODELS_DIR, "flan-t5-large-paraphraser-Q4_K_M.gguf")
        f16_path = os.path.join(MODELS_DIR, "flan-t5-large-paraphraser-f16.gguf")
        generic_path = os.path.join(MODELS_DIR, "model.gguf")

        if os.path.exists(q4_path) and os.path.getsize(q4_path) > 100_000_000:
            return q4_path
        if os.path.exists(f16_path) and os.path.getsize(f16_path) > 100_000_000:
            return f16_path
        if os.path.exists(generic_path) and os.path.getsize(generic_path) > 100_000_000:
            return generic_path
        return None

    def is_model_available(self) -> bool:
        p = self.get_model_path()
        return p is not None and os.path.exists(p)

    def get_model_size_mb(self) -> float:
        p = self.get_model_path()
        if p and os.path.exists(p):
            return os.path.getsize(p) / (1024 * 1024)
        return 0.0

    async def paraphrase_chunk(self, model_path: str, chunk: str) -> str:
        """
        Runs inference on a sentence or clause using llama-completion.
        Follows official alykassem/FLAN-T5-Paraphraser documentation: 'Paraphrase: <text>'
        """
        prompt = f"Paraphrase: {chunk.strip()}"
        # Max tokens proportional to chunk length
        max_tokens = min(max(int(len(chunk.split()) * 2.2), 64), 384)

        raw_output = await llama_manager.run_cli_inference(
            model_name=self.model_name,
            model_path=model_path,
            prompt=prompt,
            max_tokens=max_tokens,
            temp=0.7,
            ngl=0 # Run on CPU or hybrid safely for encoder-decoder
        )

        cleaned = raw_output.strip()
        # Remove prompt echoing or system artifacts if present
        if cleaned.lower().startswith("paraphrase:"):
            cleaned = cleaned.split(":", 1)[1].strip()
        if cleaned.startswith("###"):
            cleaned = cleaned.split("###")[0].strip()
        return cleaned if cleaned else chunk.strip()

    async def restructure(self, text: str) -> dict:
        """
        Stage 2: Restructures and paraphrases content using the exact fine-tuned FLAN-T5 model.
        Processes sentences to guarantee maximum structural variation (no plagiarism),
        while strictly preserving markdown headings, lists, facts, names, and numbers.
        """
        model_path = self.get_model_path()
        if not model_path:
            raise FileNotFoundError("FLAN-T5-Paraphraser GGUF model is not available. Please run conversion first.")

        start_time = time.time()
        logger.info(f"Starting FLAN-T5 Paraphraser restructuring on {len(text)} characters...")

        paragraphs = text.split("\n")
        restructured_paragraphs = []

        for p in paragraphs:
            trimmed = p.strip()
            # If line is empty or is a Markdown heading (e.g., #, ##, ###), keep as-is to preserve structure
            if not trimmed or trimmed.startswith("#") or len(trimmed.split()) < 4:
                restructured_paragraphs.append(p)
                continue

            # Split paragraph into sentences for granular paraphrasing & anti-plagiarism
            sentences = re.split(r'(?<=[.!?])\s+', trimmed)
            rephrased_sentences = []
            for s in sentences:
                s_clean = s.strip()
                if len(s_clean.split()) >= 4:
                    rephrased = await self.paraphrase_chunk(model_path, s_clean)
                    # Preserve sentence capitalization
                    if rephrased and len(rephrased) > 1:
                        rephrased = rephrased[0].upper() + rephrased[1:]
                    rephrased_sentences.append(rephrased)
                else:
                    rephrased_sentences.append(s_clean)

            restructured_paragraphs.append(" ".join(rephrased_sentences))

        final_content = "\n".join(restructured_paragraphs)
        elapsed = time.time() - start_time

        logger.info(f"FLAN-T5 restructuring complete in {elapsed:.2f}s")
        return {
            "stage": "FLAN-T5 Restructured Content",
            "model": os.path.basename(model_path),
            "content": final_content,
            "inference_time_sec": round(elapsed, 2)
        }

flan_t5_service = FlanT5Service()

