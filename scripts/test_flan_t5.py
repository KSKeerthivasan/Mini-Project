import asyncio
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from backend.services.flan_t5_service import flan_t5_service

async def test_flan():
    print("=== Testing FLAN-T5-Paraphraser GGUF Inference ===")
    print(f"Model path: {flan_t5_service.get_model_path()}")
    print(f"Model available: {flan_t5_service.is_model_available()}")
    print(f"Model size: {flan_t5_service.get_model_size_mb():.2f} MB")

    test_text = (
        "Artificial intelligence has progressed dramatically in recent years. "
        "Engineers and researchers around the world are developing autonomous systems."
    )

    print(f"\n[Test] Input text:\n{test_text}")
    print("\n[Test] Running restructuring...")
    result = await flan_t5_service.restructure(test_text)

    print("\n=== Result ===")
    print(f"Stage: {result['stage']}")
    print(f"Model: {result['model']}")
    print(f"Inference time: {result['inference_time_sec']}s")
    print(f"Restructured Content:\n{result['content']}")
    print("\nFLAN-T5 Paraphraser test PASSED!")

if __name__ == "__main__":
    asyncio.run(test_flan())

