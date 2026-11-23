import os
import sys
import time
import asyncio
import subprocess
import logging
import requests
import shutil

logger = logging.getLogger("llama_manager")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s][LlamaManager] %(message)s"))
    logger.addHandler(ch)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUNTIME_DIR = os.path.join(ROOT_DIR, "runtime", "llama.cpp")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
LOGS_DIR = os.path.join(ROOT_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

EXECUTABLE_SUFFIX = ".exe" if os.name == "nt" else ""
LLAMA_SERVER_EXE = os.path.join(RUNTIME_DIR, f"llama-server{EXECUTABLE_SUFFIX}")
LLAMA_CLI_EXE = os.path.join(RUNTIME_DIR, f"llama-cli{EXECUTABLE_SUFFIX}")
LLAMA_COMPLETION_EXE = os.path.join(RUNTIME_DIR, f"llama-completion{EXECUTABLE_SUFFIX}")

class LlamaManager:
    def __init__(self):
        self._lock = asyncio.Lock()
        self._server_process: subprocess.Popen | None = None
        self._current_model_path: str | None = None
        self._current_model_name: str | None = None
        self._server_port: int = 8080
        self._server_host: str = "127.0.0.1"

    @property
    def current_model(self) -> str | None:
        if self._server_process and self._server_process.poll() is None:
            return self._current_model_name
        return None

    def is_server_ready(self) -> bool:
        if not self._server_process or self._server_process.poll() is not None:
            return False
        try:
            r = requests.get(f"http://{self._server_host}:{self._server_port}/health", timeout=1.5)
            return r.status_code == 200
        except Exception:
            return False

    async def unload(self):
        async with self._lock:
            self._stop_server_sync()

    def _stop_server_sync(self):
        if self._server_process:
            logger.info(f"Unloading model: {self._current_model_name} (PID: {self._server_process.pid})")
            try:
                self._server_process.terminate()
                try:
                    self._server_process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self._server_process.kill()
                    self._server_process.wait(timeout=2)
            except Exception as e:
                logger.warning(f"Error terminating llama-server: {e}")
            finally:
                self._server_process = None
                self._current_model_path = None
                self._current_model_name = None

        # Clean up any lingering or orphaned llama processes on Windows.
        # Linux containers terminate the child process above directly.
        if os.name == "nt":
            try:
                subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                subprocess.run(["taskkill", "/F", "/IM", "llama-completion.exe"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass

        self._server_process = None
        self._current_model_path = None
        self._current_model_name = None
        time.sleep(1.0) # Grace period for GPU driver to reclaim memory
        logger.info("Model unloaded successfully. VRAM freed.")

    async def ensure_server(self, model_name: str, model_path: str, ngl: int = 99, ctx: int = 4096) -> str:
        """
        Ensures llama-server is running with the specified model.
        If another model is loaded, unloads it first.
        """
        async with self._lock:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            if not os.path.exists(LLAMA_SERVER_EXE):
                raise FileNotFoundError(f"llama-server executable not found: {LLAMA_SERVER_EXE}")

            # If already running the requested model and responding
            if (self._server_process and 
                self._server_process.poll() is None and 
                self._current_model_path == model_path and 
                self.is_server_ready()):
                logger.info(f"Reusing already running server for model: {model_name}")
                return f"http://{self._server_host}:{self._server_port}"

            # Stop existing server if any
            self._stop_server_sync()

            logger.info(f"Starting llama-server for [{model_name}]...")
            logger.info(f"Model path: {model_path} | GPU layers (ngl): {ngl} | Ctx: {ctx}")

            log_file_path = os.path.join(LOGS_DIR, f"llama-server-{model_name.lower()}.log")
            log_out = open(log_file_path, "w", encoding="utf-8")

            # Environment with runtime directory prepended to PATH so CUDA DLLs are found
            env = os.environ.copy()
            env["PATH"] = RUNTIME_DIR + os.pathsep + env.get("PATH", "")

            cmd = [
                LLAMA_SERVER_EXE,
                "-m", model_path,
                "-c", str(ctx),
                "-ngl", str(ngl),
                "--port", str(self._server_port),
                "--host", self._server_host,
                "--log-disable"
            ]

            self._server_process = subprocess.Popen(
                cmd,
                stdout=log_out,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=RUNTIME_DIR
            )
            self._current_model_path = model_path
            self._current_model_name = model_name

            # Wait for server readiness
            start_time = time.time()
            timeout = 90.0
            ready = False

            while time.time() - start_time < timeout:
                if self._server_process.poll() is not None:
                    log_out.flush()
                    log_out.close()
                    with open(log_file_path, "r", encoding="utf-8", errors="ignore") as lf:
                        err_content = lf.read()
                    self._stop_server_sync()
                    raise RuntimeError(f"llama-server exited unexpectedly (code {self._server_process.returncode}). Logs:\n{err_content[-1000:]}")

                if self.is_server_ready():
                    ready = True
                    break
                await asyncio.sleep(0.5)

            if not ready:
                self._stop_server_sync()
                raise TimeoutError(f"llama-server failed to become ready within {timeout}s.")

            logger.info(f"llama-server is ready on http://{self._server_host}:{self._server_port} for [{model_name}]")
            return f"http://{self._server_host}:{self._server_port}"

    async def run_cli_inference(self, model_name: str, model_path: str, prompt: str, max_tokens: int = 512, temp: float = 0.7, ngl: int = 0) -> str:
        """
        Runs inference using llama-completion/llama-cli for encoder-decoder models (like FLAN-T5).
        Unloads any running server first to ensure single-model memory constraint.
        """
        async with self._lock:
            # Enforce mutual exclusion: unload any active server before running CLI
            self._stop_server_sync()

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            bin_exe = LLAMA_COMPLETION_EXE if os.path.exists(LLAMA_COMPLETION_EXE) else LLAMA_CLI_EXE
            if not os.path.exists(bin_exe):
                raise FileNotFoundError(f"llama inference executable not found: {bin_exe}")

            logger.info(f"Running inference on [{model_name}] via {os.path.basename(bin_exe)}...")
            start_time = time.time()

            env = os.environ.copy()
            env["PATH"] = RUNTIME_DIR + os.pathsep + env.get("PATH", "")

            if bin_exe == LLAMA_COMPLETION_EXE:
                cmd = [
                    bin_exe,
                    "-m", model_path,
                    "-p", prompt,
                    "-n", str(max_tokens),
                    "--temp", str(temp),
                    "-ngl", str(ngl)
                ]
            else:
                cmd = [
                    bin_exe,
                    "-m", model_path,
                    "-p", prompt,
                    "-n", str(max_tokens),
                    "--temp", str(temp),
                    "-ngl", str(ngl),
                    "--single-turn",
                    "--simple-io"
                ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=RUNTIME_DIR
            )

            stdout_data, stderr_data = await asyncio.wait_for(process.communicate(), timeout=180.0)
            elapsed = time.time() - start_time
            raw_out = stdout_data.decode("utf-8", errors="ignore")

            # With llama-completion, stdout contains the pure generated text
            out_text = raw_out.replace("[end of text]", "").strip()

            if process.returncode != 0 and not out_text:
                err_text = stderr_data.decode("utf-8", errors="ignore")
                logger.error(f"llama inference failed with code {process.returncode}: {err_text}")
                raise RuntimeError(f"llama inference failed: {err_text[-500:]}")

            logger.info(f"[{model_name}] inference completed in {elapsed:.2f}s -> '{out_text}'")
            return out_text

    def get_hardware_info(self) -> dict:
        vram_info = {"total_mb": 6144, "used_mb": 0, "free_mb": 6144, "gpu_name": "NVIDIA GeForce RTX 3050 Laptop GPU"}
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free,name", "--format=csv,nounits,noheader"],
                capture_output=True, text=True, timeout=3
            )
            if res.returncode == 0 and res.stdout.strip():
                parts = [p.strip() for p in res.stdout.strip().split(",")]
                if len(parts) >= 4:
                    vram_info = {
                        "total_mb": int(parts[0]),
                        "used_mb": int(parts[1]),
                        "free_mb": int(parts[2]),
                        "gpu_name": parts[3]
                    }
        except Exception:
            pass

        return {
            "vram": vram_info,
            "current_loaded_model": self.current_model
        }

llama_manager = LlamaManager()
