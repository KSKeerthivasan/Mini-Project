import asyncio
import logging
import os
import time

import requests


logger = logging.getLogger("gemini_service")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ENV_FILE = os.path.join(ROOT_DIR, ".env")

LENGTH_MAP = {
    "short": {"words": "250-400 words", "max_tokens": 700},
    "medium": {"words": "500-800 words", "max_tokens": 1400},
    "long": {"words": "1000-1500 words", "max_tokens": 2500},
}


def load_project_env() -> None:
    """Load simple KEY=VALUE pairs from the project .env file.
    Reloads every time so switching MODEL1=true / false takes effect immediately.
    """
    if not os.path.isfile(ENV_FILE):
        return

    with open(ENV_FILE, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                os.environ[key] = value


load_project_env()


def model1_uses_gemini() -> bool:
    """Return True only when MODEL1 is explicitly enabled."""
    load_project_env()
    value = os.getenv("MODEL1") or os.getenv("model1") or "false"
    return value.strip().lower() == "true"


class GeminiService:
    @property
    def model_name(self) -> str:
        load_project_env()
        return os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite").strip()

    def is_configured(self) -> bool:
        load_project_env()
        return bool(os.getenv("GEMINI_API_KEY", "").strip())

    @staticmethod
    def _system_instruction(content_type: str, word_target: str) -> str:
        if content_type.lower() in ["chat", "q&a", "question", "general", "assistant"]:
            return (
                "You are an intelligent, articulate, and helpful AI assistant. "
                "Answer the user's question or prompt comprehensively, accurately, and naturally. "
                "Return clean Markdown: use short paragraphs; use ## headings only for multi-section answers; use - for lists; "
                "and use numbered lists only for ordered steps. Do not use a heading for a one-paragraph answer. "
                "Do not include meta-commentary, introductory filler, or generic boilerplate."
            )
        return (
            f"You are a professional content creator. Write a comprehensive, high-quality, and engaging {content_type.upper()} "
            f"of approximately {word_target} based strictly on the user's prompt. "
            "Return clean Markdown with one # title, logical ## section headings, short paragraphs, and - list items where useful. "
            "Do not put headings inside list items or repeat the title. Use accurate facts and engaging phrasing. "
            "Do not include meta-commentary, introductory greetings, or filler. Begin directly with the title and content."
        )

    async def generate(self, prompt: str, content_type: str = "Blog", content_length: str = "Medium") -> dict:
        load_project_env()
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required when MODEL1=true. Add it to the project .env file.")

        len_config = LENGTH_MAP.get(content_length.lower(), LENGTH_MAP["medium"])
        system_instruction = self._system_instruction(content_type, len_config["words"])
        payload = {
            "systemInstruction": {"parts": [{"text": system_instruction}]},
            "contents": [{"role": "user", "parts": [{"text": prompt.strip()}]}],
            "generationConfig": {
                "temperature": 0.7,
                "topP": 0.9,
                "topK": 40,
                "maxOutputTokens": len_config["max_tokens"],
            },
        }

        # Try user's configured model, with fallback to active flash models if deprecated
        models_to_try = [self.model_name]
        if self.model_name != "gemini-3.1-flash-lite":
            models_to_try.append("gemini-3.1-flash-lite")
        if "gemini-3.8-flash" not in models_to_try:
            models_to_try.append("gemini-3.8-flash")

        start_time = time.time()
        last_error = None

        for m_name in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{m_name}:generateContent"
            logger.info("Calling Gemini model %s for Stage 1", m_name)
            try:
                response = await asyncio.to_thread(
                    requests.post,
                    url,
                    headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                    json=payload,
                    timeout=60,
                )
                if response.status_code == 200:
                    data = response.json()
                    parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                    generated_text = "".join(part.get("text", "") for part in parts).strip()
                    if generated_text:
                        elapsed = time.time() - start_time
                        logger.info("Gemini generation completed with %s in %.2fs (%d chars)", m_name, elapsed, len(generated_text))
                        return {
                            "stage": "Gemini Initial Content",
                            "model": m_name,
                            "content": generated_text,
                            "inference_time_sec": round(elapsed, 2),
                            "content_type": content_type,
                            "content_length": content_length,
                        }
                else:
                    logger.warning("Gemini %s returned status %d: %s", m_name, response.status_code, response.text[:120])
                    last_error = f"Gemini error ({response.status_code}): {response.text[:120]}"
            except Exception as e:
                logger.warning("Gemini %s request failed: %s", m_name, e)
                last_error = str(e)

        raise RuntimeError(f"All Gemini models failed. Last error: {last_error}")


gemini_service = GeminiService()
