import asyncio
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from backend.services.gemma_service import gemma_service
from backend.services.llama_manager import llama_manager

async def test_gemma():
    print("=== Testing Gemma 3 4B GGUF Inference ===")
    print(f"Model path: {gemma_service.model_path}")
    print(f"Model available: {gemma_service.is_model_available()}")
    print(f"Model size: {gemma_service.get_model_size_mb():.2f} MB")

    test_prompt = "Explain in three concise bullet points what makes NVIDIA CUDA powerful for local LLM inference."

    print("\n[Test] Sending prompt to Gemma 3 4B...")
    result = await gemma_service.generate(prompt=test_prompt, content_type="Blog", content_length="Short")
    
    print("\n=== Result ===")
    print(f"Stage: {result['stage']}")
    print(f"Model: {result['model']}")
    print(f"Inference time: {result['inference_time_sec']}s")
    print(f"Output Content:\n{result['content']}")

    print("\n[Test] Unloading Gemma from memory...")
    await llama_manager.unload()
    print("Unloaded successfully! Gemma test PASSED!")

if __name__ == "__main__":
    asyncio.run(test_gemma())

