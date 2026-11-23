import os
import sys
import time
import asyncio
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.services.llama_manager import llama_manager
from backend.services.gemma_service import gemma_service
from backend.services.gemini_service import gemini_service, model1_uses_gemini
from backend.services.flan_t5_service import flan_t5_service
from backend.services.humanizer_service import humanizer_service

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("main_api")
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FRONTEND_DIST_DIR = os.path.join(ROOT_DIR, "frontend", "dist")

# The workstation intentionally fits only one model in VRAM at a time.  This
# lock prevents two browser requests from unloading/replacing one another's
# active model mid-generation.
generation_lock = asyncio.Lock()

app = FastAPI(
    title="LocalAIContentStudio API",
    description="Multi-model local AI content generation and refinement pipeline powered exclusively by llama.cpp",
    version="1.0.0"
)

# Enable CORS for React Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_stage1_service():
    return gemini_service if model1_uses_gemini() else gemma_service


def get_stage1_status() -> dict:
    if model1_uses_gemini():
        return {
            "provider": "gemini",
            "name": gemini_service.model_name,
            "cloud": True,
            "configured": gemini_service.is_configured(),
        }
    return {
        "provider": "gemma",
        "name": "Gemma 3 4B-it",
        "cloud": False,
        "configured": gemma_service.is_model_available(),
    }

class GemmaRequest(BaseModel):
    prompt: str
    content_type: str = "Chat"
    content_length: str = "Medium"

class FlanT5Request(BaseModel):
    content: str

class HumanizerRequest(BaseModel):
    content: str

class PipelineRequest(BaseModel):
    prompt: str
    content_type: str = "Chat"
    content_length: str = "Medium"

@app.get("/")
def read_root():
    index_file = os.path.join(FRONTEND_DIST_DIR, "index.html")
    if os.path.isfile(index_file):
        return FileResponse(index_file)
    return {
        "status": "online",
        "app": "LocalAIContentStudio",
        "engine": "llama.cpp (CUDA enabled)"
    }

@app.get("/api/status")
def get_status():
    hw = llama_manager.get_hardware_info()
    return {
        "hardware": hw,
        "models": {
            "gemma": {
                "name": "Gemma 3 4B-it",
                "available": gemma_service.is_model_available(),
                "size_mb": round(gemma_service.get_model_size_mb(), 2),
                "path": gemma_service.model_path
            },
            "flan_t5": {
                "name": "FLAN-T5-Large-Paraphraser",
                "available": flan_t5_service.is_model_available(),
                "size_mb": round(flan_t5_service.get_model_size_mb(), 2),
                "path": flan_t5_service.get_model_path()
            },
            "humanizer": {
                "name": "Humanizer 12B",
                "available": humanizer_service.is_model_available(),
                "size_mb": round(humanizer_service.get_model_size_mb(), 2),
                "path": humanizer_service.model_path
            }
        },
        "model1": get_stage1_status(),
        "currently_loaded_model": llama_manager.current_model
    }

@app.post("/api/unload")
async def unload_model():
    async with generation_lock:
        await llama_manager.unload()
    return {"status": "success", "message": "All models unloaded from memory and GPU VRAM freed."}

@app.post("/api/generate/gemma")
async def generate_gemma(req: GemmaRequest):
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
    try:
        async with generation_lock:
            try:
                try:
                    res = await get_stage1_service().generate(
                        prompt=req.prompt,
                        content_type=req.content_type,
                        content_length=req.content_length
                    )
                except Exception as e:
                    if model1_uses_gemini():
                        logger.warning(f"Gemini cloud failed ({e}), falling back to local Gemma 3 4B via llama.cpp...")
                        res = await gemma_service.generate(
                            prompt=req.prompt,
                            content_type=req.content_type,
                            content_length=req.content_length
                        )
                    else:
                        raise
            finally:
                # Keep cleanup inside the same lock: another request must not
                # have its freshly loaded model terminated by this request.
                await llama_manager.unload()
        return res
    except Exception as e:
        logger.error(f"Stage 1 error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate/flan-t5")
async def generate_flan_t5(req: FlanT5Request):
    if not req.content.strip():
        raise HTTPException(status_code=400, detail="Content cannot be empty.")
    try:
        async with generation_lock:
            try:
                # Before running FLAN-T5 CLI, unload any running server model
                await llama_manager.unload()
                res = await flan_t5_service.restructure(text=req.content)
            finally:
                await llama_manager.unload()
        return res
    except Exception as e:
        logger.error(f"FLAN-T5 stage error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate/humanizer")
async def generate_humanizer(req: HumanizerRequest):
    if not req.content.strip():
        raise HTTPException(status_code=400, detail="Content cannot be empty.")
    try:
        async with generation_lock:
            try:
                res = await humanizer_service.humanize(draft_text=req.content)
            finally:
                await llama_manager.unload()
        return res
    except Exception as e:
        logger.error(f"Humanizer stage error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/pipeline")
async def run_pipeline(req: PipelineRequest):
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    async with generation_lock:
        try:
            total_start = time.time()
            logger.info(f"=== Starting 3-Stage Pipeline for prompt: '{req.prompt[:50]}...' ===")

            # Stage 1: Gemini cloud or local Gemma (Content Generation)
            stage1_service = get_stage1_service()
            logger.info(">>> Stage 1: %s Generation >>>", "Gemini cloud" if model1_uses_gemini() else "Gemma 3 4B")
            try:
                gemma_res = await stage1_service.generate(
                    prompt=req.prompt,
                    content_type=req.content_type,
                    content_length=req.content_length
                )
            except Exception as e:
                if model1_uses_gemini():
                    logger.warning(f"Gemini cloud generation failed ({e}), automatically falling back to local Gemma 3 4B via llama.cpp...")
                    gemma_res = await gemma_service.generate(
                        prompt=req.prompt,
                        content_type=req.content_type,
                        content_length=req.content_length
                    )
                else:
                    raise
            gemma_text = gemma_res["content"]

            # Unload Gemma to free VRAM for Stage 2
            logger.info("Unloading Gemma from memory...")
            await llama_manager.unload()

            # Stage 2: FLAN-T5 Paraphraser (Content Restructuring)
            logger.info(">>> Stage 2: FLAN-T5 Paraphraser Restructuring >>>")
            flan_res = await flan_t5_service.restructure(text=gemma_text)
            flan_text = flan_res["content"]

            # Unload FLAN-T5 (CLI already clean, but ensure lock release & memory clear)
            await llama_manager.unload()

            # Stage 3: Humanizer 12B (Natural Writing Refinement)
            logger.info(">>> Stage 3: Humanizer 12B Refinement >>>")
            humanizer_res = await humanizer_service.humanize(draft_text=flan_text)
            final_text = humanizer_res["content"]

            total_elapsed = time.time() - total_start
            logger.info(f"=== Complete Pipeline finished in {total_elapsed:.2f}s ===")

            return {
                "status": "success",
                "prompt": req.prompt,
                "content_type": req.content_type,
                "content_length": req.content_length,
                "gemma": gemma_res,
                "flan_t5": flan_res,
                "humanizer": humanizer_res,
                "final_content": final_text,
                "total_inference_time_sec": round(total_elapsed, 2)
            }
        finally:
            # The final model is expensive in both RAM and VRAM; always release
            # it before a waiting request is allowed to start.
            await llama_manager.unload()

# The Docker image builds the React client into frontend/dist. Mounting it
# after all API routes preserves /api/* while serving its static assets.
if os.path.isdir(FRONTEND_DIST_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIST_DIR), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)

