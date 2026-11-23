import asyncio
import os
import sys
import time

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from backend.services.llama_manager import llama_manager
from backend.services.gemma_service import gemma_service
from backend.services.flan_t5_service import flan_t5_service
from backend.services.humanizer_service import humanizer_service

async def test_full_pipeline():
    print("=====================================================================")
    print("           TESTING FULL SEQUENTIAL PIPELINE (ALL 3 MODELS)            ")
    print("=====================================================================")
    
    user_prompt = "Create a blog on the recent rise of AI."
    print(f"\nUser Prompt: \"{user_prompt}\"")

    total_start = time.time()

    # Step 1: Gemma 3 4B
    print("\n---------------------------------------------------------------------")
    print("[STAGE 1] Gemma 3 4B (Initial Generation)")
    print("---------------------------------------------------------------------")
    gemma_res = await gemma_service.generate(prompt=user_prompt, content_type="Blog", content_length="Short")
    print(f"Model: {gemma_res['model']}")
    print(f"Time:  {gemma_res['inference_time_sec']}s")
    print(f"Gemma Draft:\n{gemma_res['content'][:400]}...\n")

    # Verify model is in memory, then unload it
    hw_before = llama_manager.get_hardware_info()
    print(f"[Memory Check] Loaded model before unload: {hw_before['current_loaded_model']}")
    print("Unloading Gemma 3 4B from memory...")
    await llama_manager.unload()
    hw_after = llama_manager.get_hardware_info()
    print(f"[Memory Check] Loaded model after unload:  {hw_after['current_loaded_model']} (VRAM freed)")

    # Step 2: FLAN-T5 Paraphraser
    print("\n---------------------------------------------------------------------")
    print("[STAGE 2] FLAN-T5 Paraphraser (Restructuring)")
    print("---------------------------------------------------------------------")
    flan_res = await flan_t5_service.restructure(gemma_res['content'])
    print(f"Model: {flan_res['model']}")
    print(f"Time:  {flan_res['inference_time_sec']}s")
    print(f"FLAN-T5 Restructured Content:\n{flan_res['content'][:400]}...\n")

    # Step 3: Humanizer 12B
    print("\n---------------------------------------------------------------------")
    print("[STAGE 3] Humanizer 12B (Natural Refinement)")
    print("---------------------------------------------------------------------")
    humanizer_res = await humanizer_service.humanize(flan_res['content'])
    print(f"Model: {humanizer_res['model']}")
    print(f"Time:  {humanizer_res['inference_time_sec']}s")
    print(f"Final Humanized Content:\n{humanizer_res['content']}")

    # Unload Humanizer
    print("\nUnloading Humanizer 12B from memory...")
    await llama_manager.unload()
    hw_final = llama_manager.get_hardware_info()
    print(f"[Memory Check] Final state: {hw_final['current_loaded_model']} (All models clean)")

    total_time = round(time.time() - total_start, 2)
    print("\n=====================================================================")
    print(f"Pipeline Completed Successfully in {total_time}s!")
    print("=====================================================================")

if __name__ == "__main__":
    asyncio.run(test_full_pipeline())

