"""Run: python -m uvicorn api:app --host 127.0.0.1 --port 8000 --workers 1."""
import gc
import logging
import threading
import time
from typing import Annotated
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
import torch

from gps_sdxl_inference import DemoSettings, GPSDiffusionXLRunner, InputValidationError, prepare_inputs

logger = logging.getLogger("uvicorn.error")
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


def create_app(settings=None, runner_factory=GPSDiffusionXLRunner):
    settings = settings or DemoSettings()
    application = FastAPI(
        title="GPSDiffusion SDXL Demo", version="1.0.0",
        description="Upload a composite image and its object mask (white object, black background). "
                    "Raw output: 512x512. Optional refined output: 256x256. One GPU request at a time.",
    )
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    application.mount("/outputs", StaticFiles(directory=settings.output_dir), name="outputs")
    lock = threading.Lock()
    runner = None
    last_error = None

    @application.get("/health")
    def health():
        missing = settings.missing_files()
        cuda_available = torch.cuda.is_available()
        status = "busy" if lock.locked() else (
            "error" if last_error else "ok" if runner is not None else
            "not_ready" if missing or not cuda_available else "ready_to_load"
        )
        return {"status": status, "model_loaded": runner is not None,
                "cuda_available": cuda_available, "device": settings.device,
                "weights_ready": not missing, "missing_files": missing,
                "load_postprocess": settings.load_postprocess,
                "low_vram": settings.low_vram, "error": last_error}

    def predict_sync(image, mask, num_samples, num_steps, seed, apply_postprocess, base_url):
        nonlocal runner, last_error
        if not lock.acquire(blocking=False):
            raise HTTPException(409, "GPU is busy. Wait for the current request to finish, then retry.")
        request_id = uuid4().hex
        request_started = time.perf_counter()
        logger.info("[request %s] Accepted: samples=%d, steps=%d, seed=%d, postprocess=%s",
                    request_id, num_samples, num_steps, seed, apply_postprocess)
        try:
            prepared = prepare_inputs(image, mask)
            if apply_postprocess and not settings.load_postprocess:
                raise InputValidationError("Postprocessing is disabled; set apply_postprocess=false.")
            if runner is None:
                runner = runner_factory(settings)
            result = runner.generate(prepared, num_samples=num_samples, num_steps=num_steps,
                                     seed=seed, apply_postprocess=apply_postprocess)
            request_dir = settings.output_dir / request_id
            request_dir.mkdir()
            response = {"request_id": request_id, "seeds": result.seeds,
                        "num_samples": len(result.generated), "num_steps": num_steps,
                        "peak_vram_gb": result.peak_vram_gb}
            for group in ("generated", "postprocessed", "shadow_masks"):
                response[group] = []
                for index, output in enumerate(getattr(result, group)):
                    filename = f"{group}_{index}.png"
                    output.save(request_dir / filename, format="PNG")
                    response[group].append({"filename": filename,
                                            "url": f"{base_url}outputs/{request_id}/{filename}"})
            logger.info("[request %s] Response ready in %.1f seconds; outputs saved to %s",
                        request_id, time.perf_counter() - request_started, request_dir)
            last_error = None
            return response
        except InputValidationError as exc:
            raise HTTPException(422, str(exc)) from exc
        except (FileNotFoundError, RuntimeError, ValueError, ImportError, OSError, KeyError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            logger.exception("GPSDiffusion inference failed")
            detail = last_error
            if "out of memory" in str(exc).lower():
                detail += ". Restart with GPSXL_LOW_VRAM=true and num_samples=1; check nvidia-smi for other processes."
            raise HTTPException(503, detail) from exc
        finally:
            # Release failures from model initialization too; never leave the
            # API permanently busy after an invalid upload or CUDA exception.
            try:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            finally:
                lock.release()

    @application.post("/predict", responses={409: {"description": "GPU busy"},
                                             422: {"description": "Invalid image, mask, or settings"},
                                             503: {"description": "Weights, GPU, or model unavailable"}})
    async def predict(
        request: Request,
        image: Annotated[UploadFile, File(description="Composite RGB image (PNG/JPEG), max 20 MiB.")],
        mask: Annotated[UploadFile, File(description="Same dimensions as image. White = foreground object; black = background.")],
        num_samples: Annotated[int, Form(ge=1, le=4, description="Generated sequentially; start with 1.")] = 1,
        num_steps: Annotated[int, Form(ge=1, le=100)] = 50,
        seed: Annotated[int, Form(ge=0, le=4294967292)] = 42,
        apply_postprocess: Annotated[bool, Form(description="Also produce refined image and shadow mask at 256x256.")] = True,
    ):
        try:
            image_bytes = await image.read(MAX_UPLOAD_BYTES + 1)
            mask_bytes = await mask.read(MAX_UPLOAD_BYTES + 1)
        finally:
            await image.close()
            await mask.close()
        if len(image_bytes) > MAX_UPLOAD_BYTES or len(mask_bytes) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Image and mask must each be at most 20 MiB.")
        return await run_in_threadpool(predict_sync, image_bytes, mask_bytes, num_samples,
                                      num_steps, seed, apply_postprocess, str(request.base_url))

    return application


app = create_app()
