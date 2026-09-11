"""Single-image GPSDiffusion SDXL inference, independent of the training dataset.

The denoiser follows test_GPSDiffusion_sdxl.py, including its four IP attention
tokens and fixed prompt. Models are loaded once, on CPU, then staged on CUDA.
"""
from dataclasses import dataclass, field
import gc
import io
import os
from pathlib import Path
import pickle

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
RESOLUTION = 512
PROMPT = "foreground object with shadow"


def _env_path(name, default):
    path = Path(os.environ.get(name, str(default))).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass
class DemoSettings:
    weights_dir: Path = field(default_factory=lambda: _env_path("GPSXL_WEIGHTS_DIR", PROJECT_ROOT / "pretrained_models"))
    post_checkpoint: Path = field(default_factory=lambda: _env_path("GPSXL_POST_CHECKPOINT", PROJECT_ROOT / "models/pretrained_models/Shadow_ppp.ckpt"))
    output_dir: Path = field(default_factory=lambda: _env_path("GPSXL_OUTPUT_DIR", PROJECT_ROOT / "outputs"))
    base_model: str = field(default_factory=lambda: os.getenv("GPSXL_BASE_MODEL", "stabilityai/stable-diffusion-xl-base-1.0"))
    device: str = field(default_factory=lambda: os.getenv("GPSXL_DEVICE", "cuda:0"))
    load_postprocess: bool = field(default_factory=lambda: os.getenv("GPSXL_LOAD_POSTPROCESS", "true").lower() in {"true", "1", "yes"})
    low_vram: bool = field(default_factory=lambda: os.getenv("GPSXL_LOW_VRAM", "false").lower() in {"true", "1", "yes"})

    def missing_files(self):
        required = [self.weights_dir / name for name in (
            "controlnet/config.json", "ip_adapter.ckpt", "Shadow_cls.pth",
            "Shadow_reg.pth", "Shadow_cls_label.pkl",
        )]
        if self.load_postprocess:
            required.append(self.post_checkpoint)
        missing = [str(path) for path in required if not path.is_file()]
        control_dir = self.weights_dir / "controlnet"
        if not any(control_dir.glob("diffusion_pytorch_model*.safetensors")) and not any(control_dir.glob("diffusion_pytorch_model*.bin")):
            missing.append(str(control_dir / "diffusion_pytorch_model.safetensors (or .bin)"))
        return missing


class InputValidationError(ValueError):
    pass


@dataclass
class PreparedInputs:
    image: np.ndarray
    mask: np.ndarray
    control: np.ndarray
    pixels: np.ndarray
    bbox: np.ndarray


@dataclass
class InferenceResult:
    generated: list
    postprocessed: list
    shadow_masks: list
    seeds: list
    peak_vram_gb: float


def _open_image(source):
    try:
        if isinstance(source, Image.Image):
            image = source.copy()
        else:
            with Image.open(io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source) as opened:
                if opened.width * opened.height > 25_000_000:
                    raise InputValidationError("Image and mask must each contain at most 25 million pixels.")
                image = opened.copy()
        return ImageOps.exif_transpose(image)
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise InputValidationError("Could not decode image or mask; use a valid PNG or JPEG.") from exc


def prepare_inputs(image, mask):
    image, mask = _open_image(image), _open_image(mask)
    if image.size != mask.size:
        raise InputValidationError("Image and object mask must have the same dimensions.")
    image = np.asarray(image.convert("RGB").resize((RESOLUTION, RESOLUTION), Image.Resampling.BILINEAR))
    mask = np.asarray(mask.convert("L").resize((RESOLUTION, RESOLUTION), Image.Resampling.NEAREST))
    mask = (mask >= 128).astype(np.uint8) * 255
    if not mask.any():
        raise InputValidationError("Object mask must contain white foreground pixels (value >= 128).")
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    (x, y), (w, h), theta = cv2.minAreaRect(np.concatenate(contours))
    if w < h:
        w, h, theta = h, w, theta + 90
    # Match the integer foreground boxes used by the original dataset.
    bbox = np.array([x, y, w + 1, h + 1, theta]).astype(np.int32)
    control = np.concatenate([image, mask[..., None]], axis=-1).astype(np.float32) / 255
    pixels = image.astype(np.float32) / 127.5 - 1
    return PreparedInputs(image, mask, control.transpose(2, 0, 1)[None].copy(),
                          pixels.transpose(2, 0, 1)[None].copy(), bbox)


def rasterize_shadow_box(delta, foreground, *, size=RESOLUTION, rng, jitter=5):
    """Decode the rotated box and fill the actual returned buffer, on CPU."""
    dx, dy, dw, dh, angle = np.asarray(delta, dtype=np.float32)
    x, y, w, h, theta = np.asarray(foreground, dtype=np.float32)
    x, y = dx * w + x, dy * h + y
    w, h = w * np.exp(np.clip(dw, -10, 10)), h * np.exp(np.clip(dh, -10, 10))
    theta = angle * 180 / np.pi + theta
    if theta > 0:
        w, h, theta = h, w, theta - 90
    if not np.isfinite([x, y, w, h, theta]).all():
        raise RuntimeError("Geometry predictor returned non-finite bounding box coordinates.")
    points = cv2.boxPoints(((float(x), float(y)), (float(w), float(h)), float(theta)))
    points += rng.uniform(-jitter, jitter, points.shape).astype(np.float32)
    points = np.clip(points, 0, size - 1).astype(np.int32)
    region = np.zeros((size, size), dtype=np.uint8)
    cv2.fillPoly(region, [points], 1)
    return region[None, None].astype(np.float32)


def select_mask_embeddings(scores, centroids):
    labels = scores.topk(64, largest=True, sorted=False).indices.cpu().tolist()
    # Build once per image. Repeated += inside the denoising loop would amplify
    # the embeddings at every step when the source tensor is already on CUDA.
    embeddings = np.zeros((len(labels), 64, 2048), dtype=np.float32)
    for row, values in enumerate(labels):
        for col, label in enumerate(values):
            if label not in centroids:
                raise ValueError(f"Shadow_cls_label.pkl has no centroid for class {label}.")
            embeddings[row, col] = np.asarray(centroids[label]).reshape(2048)
    return torch.from_numpy(embeddings).to(device=scores.device, dtype=scores.dtype)


def merge_postprocess(original, generated, prediction):
    # Match post_processing.py: use its predicted shadow mask to composite the
    # diffusion RGB output over the original, not the unused RGB prediction.
    shadow = prediction[..., 3] >= 0
    merged = np.where(shadow[..., None], generated, original).astype(np.uint8)
    return Image.fromarray(merged), Image.fromarray(shadow.astype(np.uint8) * 255)


class ImageProjModel(torch.nn.Module):
    def __init__(self, cross_attention_dim):
        super().__init__()
        self.proj = torch.nn.Linear(2048, cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, embeddings):
        return self.norm(self.proj(embeddings))


def install_ip_adapter(unet, checkpoint):
    from attention_processor import AttnProcessor2_0, IPAttnProcessor2_0
    processors = {}
    for name in unet.attn_processors:
        if name.endswith("attn1.processor"):
            processors[name] = AttnProcessor2_0()
            continue
        if name.startswith("mid_block"):
            hidden = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            hidden = list(reversed(unet.config.block_out_channels))[int(name.split(".")[1])]
        else:
            hidden = unet.config.block_out_channels[int(name.split(".")[1])]
        # Keep the original training/inference setting of 4, even though the
        # projection receives 64 centroids. Changing it changes the checkpoint's
        # text/IP attention split and needs a separate model-quality experiment.
        processors[name] = IPAttnProcessor2_0(hidden, unet.config.cross_attention_dim, num_tokens=4)
    unet.set_attn_processor(processors)
    projection = ImageProjModel(unet.config.cross_attention_dim)
    projection.load_state_dict(checkpoint["image_proj"], strict=True)
    torch.nn.ModuleList(unet.attn_processors.values()).load_state_dict(checkpoint["ip_adapter"], strict=True)
    return projection


class GPSDiffusionXLRunner:
    def __init__(self, settings):
        self.settings = settings
        missing = settings.missing_files()
        if missing:
            raise FileNotFoundError("Missing model files:\n" + "\n".join(missing))
        self.device = torch.device(settings.device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Inference requires an NVIDIA CUDA GPU. Check nvidia-smi and your PyTorch CUDA installation.")
        torch.cuda.set_device(self.device)
        self.dtype = torch.float16
        self.models = []
        try:
            self._load()
        except Exception:
            self._offload_all()
            raise

    def _keep(self, model):
        model.eval().requires_grad_(False)
        self.models.append(model)
        return model

    @torch.inference_mode()
    def _load(self):
        from diffusers import AutoencoderKL, ControlNetModel, DDPMScheduler, UNet2DConditionModel
        from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection
        from base_network import MaskCls, RegNetwork

        # Encode the fixed prompt with one text encoder at a time, then release
        # both encoders permanently. No dataset or on-disk embedding cache.
        prompt_parts = []
        for index, cls in enumerate((CLIPTextModel, CLIPTextModelWithProjection)):
            suffix = "" if index == 0 else "_2"
            tokenizer = AutoTokenizer.from_pretrained(self.settings.base_model, subfolder="tokenizer" + suffix, use_fast=False)
            encoder = cls.from_pretrained(self.settings.base_model, subfolder="text_encoder" + suffix, torch_dtype=self.dtype)
            try:
                encoder.to(self.device).eval()
                tokens = tokenizer(PROMPT, padding="max_length", max_length=tokenizer.model_max_length,
                                   truncation=True, return_tensors="pt").input_ids.to(self.device)
                encoded = encoder(tokens, output_hidden_states=True)
                prompt_parts.append(encoded.hidden_states[-2].cpu())
                if index == 1:
                    self.text_embeds = encoded.text_embeds.cpu()
                del encoded, tokens
            finally:
                encoder.cpu()
                del encoder
                self._empty_cache()
        self.prompt_embeds = torch.cat(prompt_parts, dim=-1)
        self.scheduler = DDPMScheduler.from_pretrained(self.settings.base_model, subfolder="scheduler")
        self.vae = self._keep(AutoencoderKL.from_pretrained(self.settings.base_model, subfolder="vae", torch_dtype=torch.float32))
        self.vae.enable_slicing()
        self.vae.enable_tiling()
        self.unet = self._keep(UNet2DConditionModel.from_pretrained(self.settings.base_model, subfolder="unet", torch_dtype=self.dtype))
        checkpoint = torch.load(self.settings.weights_dir / "ip_adapter.ckpt", map_location="cpu", weights_only=True)
        self.projection = self._keep(install_ip_adapter(self.unet, checkpoint).to(dtype=self.dtype))
        # New attention modules are initially FP32; cast them along with UNet.
        self.unet.to(dtype=self.dtype)
        del checkpoint
        self.controlnet = self._keep(ControlNetModel.from_pretrained(self.settings.weights_dir / "controlnet", torch_dtype=self.dtype))
        if self.controlnet.config.conditioning_channels != 5:
            raise ValueError("Expected GPSDiffusion SDXL ControlNet with 5 conditioning channels (RGB + object mask + geometry).")
        self.classifier = self._keep(MaskCls(num_classes=256, pretrained=False))
        self.regressor = self._keep(RegNetwork())
        for model, filename in ((self.classifier, "Shadow_cls.pth"), (self.regressor, "Shadow_reg.pth")):
            state = torch.load(self.settings.weights_dir / filename, map_location="cpu", weights_only=True)
            model.load_state_dict(state["net"], strict=True)
            model.to(dtype=self.dtype)
        with (self.settings.weights_dir / "Shadow_cls_label.pkl").open("rb") as handle:
            # Only load the checkpoint/centroid files obtained from the author.
            self.centroids = pickle.load(handle)
        self.postprocess = None
        self._empty_cache()

    def _empty_cache(self):
        gc.collect()
        with torch.cuda.device(self.device):
            torch.cuda.empty_cache()

    def _offload_all(self):
        for model in self.models:
            model.cpu()
        self._empty_cache()

    def _load_postprocess(self):
        if self.postprocess is None:
            from train_post_process_predictor import PostProcess
            model = PostProcess()
            # Lightning checkpoints can contain non-tensor metadata. These are
            # local, trusted author checkpoints, never uploaded API inputs.
            state = torch.load(self.settings.post_checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(state.get("state_dict", state), strict=True)
            self.postprocess = self._keep(model.post_process_net)

    @torch.inference_mode()
    def _generate_one(self, prepared, num_steps, seed, apply_postprocess):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        rng = np.random.default_rng(seed)
        geometry = torch.from_numpy(prepared.control).to(self.device, dtype=self.dtype)
        self.regressor.to(self.device)
        delta = self.regressor(geometry)[0].float().cpu().numpy()
        self.regressor.cpu()
        self.classifier.to(self.device)
        embeddings = select_mask_embeddings(self.classifier(geometry), self.centroids)
        self.classifier.cpu()
        region = torch.from_numpy(rasterize_shadow_box(delta, prepared.bbox, rng=rng)).to(self.device, dtype=self.dtype)
        control = torch.cat([geometry, region], dim=1)
        self._empty_cache()

        self.vae.to(self.device)
        pixels = torch.from_numpy(prepared.pixels).to(self.device)
        latents = self.vae.encode(pixels).latent_dist.sample(generator=generator) * self.vae.config.scaling_factor
        latents = latents.to(self.dtype)
        self.vae.cpu()
        del pixels
        self._empty_cache()
        noise = torch.randn(latents.shape, device=self.device, dtype=self.dtype, generator=generator)
        initial_t = torch.full((1,), 999, device=self.device, dtype=torch.long)
        latents = self.scheduler.add_noise(latents.float(), noise.float(), initial_t).to(self.dtype)
        self.scheduler.set_timesteps(num_steps, device=self.device)
        prompt = self.prompt_embeds.to(self.device, dtype=self.dtype)
        added = {"text_embeds": self.text_embeds.to(self.device, dtype=self.dtype),
                 "time_ids": torch.tensor([[512, 512, 0, 0, 512, 512]], device=self.device, dtype=self.dtype)}
        self.projection.to(self.device)
        ip_prompt = torch.cat([prompt, self.projection(embeddings)], dim=1)
        self.projection.cpu()

        if not self.settings.low_vram:
            self.controlnet.to(self.device)
            self.unet.to(self.device)
        for timestep in self.scheduler.timesteps:
            if self.settings.low_vram:
                self.controlnet.to(self.device)
            down, mid = self.controlnet(latents, timestep, encoder_hidden_states=prompt,
                                        added_cond_kwargs=added, controlnet_cond=control, return_dict=False)
            if self.settings.low_vram:
                self.controlnet.cpu()
                self._empty_cache()
                self.unet.to(self.device)
            prediction = self.unet(latents, timestep, encoder_hidden_states=ip_prompt,
                                   added_cond_kwargs=added, down_block_additional_residuals=down,
                                   mid_block_additional_residual=mid, return_dict=False)[0]
            if self.settings.low_vram:
                self.unet.cpu()
                self._empty_cache()
            latents = self.scheduler.step(prediction, timestep, latents, generator=generator).prev_sample
            del down, mid, prediction
        self.unet.cpu()
        self.controlnet.cpu()
        del control, geometry, embeddings, ip_prompt
        self._empty_cache()

        self.vae.to(self.device)
        decoded = self.vae.decode(latents.float() / self.vae.config.scaling_factor).sample
        if not torch.isfinite(decoded).all():
            raise RuntimeError("VAE decoded non-finite values; check checkpoint compatibility.")
        rgb = ((decoded[0].clamp(-1, 1) + 1) * 127.5).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        self.vae.cpu()
        del decoded, latents
        self._empty_cache()
        generated = Image.fromarray(rgb)
        if not apply_postprocess:
            return generated, None, None

        self._load_postprocess()
        self.postprocess.to(self.device)
        original = np.asarray(Image.fromarray(prepared.image).resize((256, 256), Image.Resampling.BILINEAR))
        resized = np.asarray(generated.resize((256, 256), Image.Resampling.NEAREST))
        mask = np.asarray(Image.fromarray(prepared.mask).resize((256, 256), Image.Resampling.NEAREST))
        post_input = np.concatenate([resized.astype(np.float32) / 127.5 - 1,
                                     original.astype(np.float32) / 127.5 - 1,
                                     mask[..., None].astype(np.float32) / 255], axis=-1)
        tensor = torch.from_numpy(post_input.transpose(2, 0, 1).copy()).unsqueeze(0).to(self.device)
        prediction = self.postprocess(tensor, timesteps=torch.zeros(1, device=self.device))
        prediction = prediction[0].permute(1, 2, 0).cpu().numpy()
        postprocessed, shadow_mask = merge_postprocess(original, resized, prediction)
        return generated, postprocessed, shadow_mask

    @torch.inference_mode()
    def generate(self, prepared, *, num_samples=1, num_steps=50, seed=42, apply_postprocess=True):
        if not 1 <= num_samples <= 4 or not 1 <= num_steps <= 100 or not 0 <= seed <= 2**32 - 4:
            raise InputValidationError("Invalid samples (1-4), steps (1-100), or seed (0-4294967292).")
        if apply_postprocess and not self.settings.load_postprocess:
            raise InputValidationError("Postprocessing is disabled. Set apply_postprocess=false or enable GPSXL_LOAD_POSTPROCESS.")
        track_vram = self.device.type == "cuda"
        if track_vram:
            torch.cuda.reset_peak_memory_stats(self.device)
        result = InferenceResult([], [], [], [], 0)
        try:
            for index in range(num_samples):
                try:
                    generated, refined, mask = self._generate_one(prepared, num_steps, seed + index, apply_postprocess)
                    result.generated.append(generated)
                    result.seeds.append(seed + index)
                    if refined is not None:
                        result.postprocessed.append(refined)
                        result.shadow_masks.append(mask)
                finally:
                    self._offload_all()
            if track_vram:
                result.peak_vram_gb = round(torch.cuda.max_memory_reserved(self.device) / 1024**3, 3)
            return result
        finally:
            self._offload_all()
