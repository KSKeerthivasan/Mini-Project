import asyncio
import os
import time
import logging
import requests
from backend.services.llama_manager import llama_manager

logger = logging.getLogger("gemma_service")
logger.setLevel(logging.INFO)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GEMMA_MODEL_PATH = os.path.join(ROOT_DIR, "models", "gemma", "gemma-3-4b-it-Q4_K_M.gguf")

LENGTH_MAP = {
    "short": {"words": "250-400 words", "max_tokens": 700},
    "medium": {"words": "500-800 words", "max_tokens": 1400},
    "long": {"words": "1000-1500 words", "max_tokens": 2500}
}

class GemmaService:
    def __init__(self, model_path: str = GEMMA_MODEL_PATH):
        self.model_path = model_path
        self.model_name = "Gemma-3-4B-IT"

    def is_model_available(self) -> bool:
        return os.path.exists(self.model_path) and os.path.getsize(self.model_path) > 1_000_000_000

    def get_model_size_mb(self) -> float:
        if os.path.exists(self.model_path):
            return os.path.getsize(self.model_path) / (1024 * 1024)
        return 0.0

    async def generate(self, prompt: str, content_type: str = "Blog", content_length: str = "Medium") -> dict:
        """
        Stage 1: Generates initial content from user prompt using Gemma 3 4B.
        """
        if not self.is_model_available():
            raise FileNotFoundError(f"Gemma model file is not available at {self.model_path}. Please complete download first.")

        len_config = LENGTH_MAP.get(content_length.lower(), LENGTH_MAP["medium"])
        word_target = len_config["words"]
        max_tokens = len_config["max_tokens"]

        # Dynamic instruction based on content type
        ct_lower = content_type.lower()
        anti_ai_rules = (
            "Write with authentic human clarity, varied sentence lengths, and natural flow. "
            "Never use robotic AI clichés, filler, or overused buzzwords such as 'delve', 'tapestry', 'beacon', "
            "'testament to', 'in today's fast-paced digital world', 'it is important to remember', 'furthermore', or 'in conclusion'. "
            "Use concrete details, active voice, and clear transitions."
        )

        if ct_lower in ["chat", "q&a", "question", "general", "assistant"]:
            system_instruction = (
                "You are an intelligent, articulate, and thoughtful assistant. "
                "Answer the user's question or prompt comprehensively, accurately, and naturally. "
                f"{anti_ai_rules} "
                "Return clean Markdown: use short paragraphs; use ## headings only for multi-section answers; use - for lists; "
                "and use numbered lists only for ordered steps. Do not use a heading for a one-paragraph answer. "
                "Do not include meta-commentary, introductory filler, or generic boilerplate."
            )
        else:
            system_instruction = (
                f"You are a professional content creator. Write a comprehensive, high-quality, and engaging {content_type.upper()} "
                f"of approximately {word_target} based strictly on the user's prompt. "
                f"{anti_ai_rules} "
                "Return clean Markdown with one # title, logical ## section headings, short paragraphs, and - list items where useful. "
                "Do not put headings inside list items or repeat the title. Use accurate facts and engaging phrasing. "
                "Do not include meta-commentary, introductory greetings, or filler. Begin directly with the title and content."
            )

        formatted_prompt = (
            f"<start_of_turn>user\n"
            f"{system_instruction}\n\n"
            f"Topic/Prompt: {prompt.strip()}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )

        start_time = time.time()
        logger.info(f"Ensuring Gemma-3-4B server is loaded on GPU (ngl=99)...")
        # RTX 3050 has 6GB VRAM; Gemma 3 4B Q4_K_M is ~2.5GB, so 99 layers offload fully into VRAM!
        base_url = await llama_manager.ensure_server(
            model_name=self.model_name,
            model_path=self.model_path,
            ngl=99,
            ctx=4096
        )

        logger.info(f"Calling llama-server completion API for Gemma...")
        # Keep prompt plus output inside the 4k context window.  Without this
        # guard, a long user prompt can lead to a truncated or failed response.
        prompt_tokens_estimate = max(1, len(formatted_prompt) // 4)
        available_output_tokens = 4096 - prompt_tokens_estimate - 256
        if available_output_tokens < 128:
            raise ValueError("Prompt is too long for this model context. Please use fewer than about 14,000 characters.")

        payload = {
            "prompt": formatted_prompt,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "n_predict": min(max_tokens, available_output_tokens),
            "stop": ["<end_of_turn>", "<start_of_turn>"]
        }

        # requests is synchronous; offload it so this FastAPI worker can still
        # serve status requests while a model is generating.
        resp = await asyncio.to_thread(requests.post, f"{base_url}/completion", json=payload, timeout=240)
        if resp.status_code != 200:
            raise RuntimeError(f"Gemma completion error ({resp.status_code}): {resp.text}")

        data = resp.json()
        generated_text = data.get("content", "").strip()
        elapsed = time.time() - start_time

        logger.info(f"Gemma generation completed in {elapsed:.2f}s ({len(generated_text)} chars)")
        return {
            "stage": "Gemma Initial Content",
            "model": "gemma-3-4b-it-Q4_K_M.gguf",
            "content": generated_text,
            "inference_time_sec": round(elapsed, 2),
            "content_type": content_type,
            "content_length": content_length
        }

gemma_service = GemmaService()

