import asyncio
import os
import sys

# Ensure project root is in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from backend.services.humanizer_service import humanizer_service
from backend.services.llama_manager import llama_manager

async def test_humanizer():
    print("=== Testing Humanizer 12B GGUF Inference ===")
    print(f"Model path: {humanizer_service.model_path}")
    print(f"Model available: {humanizer_service.is_model_available()}")
    print(f"Model size: {humanizer_service.get_model_size_mb():.2f} MB")

    sample_draft = (
        "Artificial intelligence systems are rapidly proliferating throughout contemporary software ecosystems. "
        "Consequently, organizational structures must adapt to automated communication workflows."
    )

    print("\n[Test] Sending draft text to Humanizer...")
    result = await humanizer_service.humanize(sample_draft)
    print("\n=== Result ===")
    print(f"Stage: {result['stage']}")
    print(f"Model: {result['model']}")
    print(f"Inference time: {result['inference_time_sec']}s")
    print(f"Output Content:\n{result['content']}")

    print("\n[Test] Unloading model to verify memory release...")
    await llama_manager.unload()
    print("Unloaded successfully! Humanizer test PASSED!")

if __name__ == "__main__":
    asyncio.run(test_humanizer())

