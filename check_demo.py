"""Check files, dependencies and CUDA without downloading/loading checkpoints."""
from importlib.metadata import version
import sys

import torch
from gps_sdxl_inference import DemoSettings


def main():
    settings = DemoSettings()
    errors = []
    print(f"Python: {sys.version.split()[0]}")
    for package in ("torch", "torchvision", "diffusers", "transformers", "accelerate", "fastapi", "pytorch-lightning"):
        try:
            print(f"{package}: {version(package)}")
        except Exception as exc:
            errors.append(f"Dependency {package}: {exc}")
    try:
        from diffusers import ControlNetModel, UNet2DConditionModel, AutoencoderKL
        from attention_processor import IPAttnProcessor2_0
        from base_network import MaskCls
        if settings.load_postprocess:
            from train_post_process_predictor import PostProcess
    except Exception as exc:
        errors.append(f"Model imports: {type(exc).__name__}: {exc}")
    print(f"Weights: {settings.weights_dir}")
    print(f"Base model: {settings.base_model}")
    print(f"Postprocess: {settings.load_postprocess} ({settings.post_checkpoint})")
    print(f"Low VRAM mode: {settings.low_vram}")
    errors.extend(f"Missing: {path}" for path in settings.missing_files())
    try:
        device = torch.device(settings.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable. Check NVIDIA driver and CUDA-enabled PyTorch.")
        with torch.cuda.device(device):
            free, total = torch.cuda.mem_get_info()
            print(f"GPU: {torch.cuda.get_device_name(device)}")
            print(f"VRAM free/total: {free / 1024**3:.2f}/{total / 1024**3:.2f} GiB")
            # Check an actual CUDA operation, not just driver visibility.
            print(f"CUDA calculation: {(torch.ones(1, device=device) + 1).item()}")
    except Exception as exc:
        errors.append(f"GPU: {exc}")
    if errors:
        print("\nNOT READY:\n" + "\n".join(errors))
        return 1
    print("\nPreflight passed. Checkpoint contents, inference, and peak VRAM still need a real /predict request.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
