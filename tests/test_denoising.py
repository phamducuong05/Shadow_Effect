"""Exercise the real sampling loop with tiny random Diffusers models on CPU."""
import unittest

import numpy as np
from PIL import Image
import torch
from diffusers import AutoencoderKL, ControlNetModel, DDPMScheduler, UNet2DConditionModel

from attention_processor import AttnProcessor2_0, IPAttnProcessor2_0
from gps_sdxl_inference import DemoSettings, GPSDiffusionXLRunner, ImageProjModel, install_ip_adapter, prepare_inputs


class CpuHarness(GPSDiffusionXLRunner):
    """Replace only external weights/CUDA; retain the production sampling code."""
    def __init__(self):
        self.settings = DemoSettings(load_postprocess=False)
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.models = []
        self.vae = self._keep(AutoencoderKL(
            in_channels=3, out_channels=3, latent_channels=4,
            down_block_types=("DownEncoderBlock2D",) * 4,
            up_block_types=("UpDecoderBlock2D",) * 4,
            block_out_channels=(8, 8, 8, 8), layers_per_block=1,
            norm_num_groups=4, sample_size=512,
        ))
        self.unet = self._keep(UNet2DConditionModel(
            sample_size=64, in_channels=4, out_channels=4, layers_per_block=1,
            block_out_channels=(8, 16), norm_num_groups=4,
            down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
            cross_attention_dim=16, attention_head_dim=(2, 2),
            addition_embed_type="text_time", addition_time_embed_dim=2,
            projection_class_embeddings_input_dim=20,
        ))
        self.controlnet = self._keep(ControlNetModel.from_unet(
            self.unet, conditioning_channels=5, conditioning_embedding_out_channels=(4, 8, 8, 8),
        ))
        processors = []
        for name in self.unet.attn_processors:
            attn = self.unet.get_submodule(name.rsplit(".processor", 1)[0])
            processors.append(AttnProcessor2_0() if name.endswith("attn1.processor") else
                              IPAttnProcessor2_0(attn.to_q.out_features, 16, num_tokens=4))
        checkpoint = {"image_proj": ImageProjModel(16).state_dict(),
                      "ip_adapter": torch.nn.ModuleList(processors).state_dict()}
        self.projection = self._keep(install_ip_adapter(self.unet, checkpoint))
        self.classifier = self._keep(torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1),
                                                       torch.nn.Flatten(), torch.nn.Linear(4, 256)))
        self.regressor = self._keep(torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1),
                                                      torch.nn.Flatten(), torch.nn.Linear(4, 5)))
        torch.nn.init.zeros_(self.regressor[-1].weight)
        torch.nn.init.zeros_(self.regressor[-1].bias)
        self.centroids = {i: np.full(2048, i / 255, dtype=np.float32) for i in range(256)}
        self.scheduler = DDPMScheduler()
        self.prompt_embeds = torch.randn(1, 77, 16)
        self.text_embeds = torch.randn(1, 8)
        # Default random initialization in this deliberately tiny VAE can
        # overflow after a diffusion step on older CPU kernels. Scaling keeps
        # the fixture finite while preserving observable seed differences.
        with torch.no_grad():
            for model in (self.vae, self.unet, self.controlnet, self.projection):
                for parameter in model.parameters():
                    parameter.mul_(0.25)

    def _empty_cache(self):
        pass


class DenoisingTests(unittest.TestCase):
    def test_sampling_is_seeded_and_offload_modes_agree(self):
        torch.set_num_threads(1)
        torch.manual_seed(123)
        runner = CpuHarness()
        prepared = prepare_inputs(Image.new("RGB", (32, 32), "white"), Image.new("L", (32, 32), 255))
        with self.assertLogs("uvicorn.error", level="INFO") as captured:
            first = runner.generate(prepared, num_samples=2, num_steps=2, seed=42, apply_postprocess=False)
        messages = "\n".join(captured.output)
        self.assertIn("Denoising 2/2 (100%)", messages)
        self.assertIn("[sample 2/2] Complete", messages)
        self.assertIn("[inference] Complete", messages)
        runner.settings.low_vram = True
        repeat = runner.generate(prepared, num_steps=2, seed=42, apply_postprocess=False)
        self.assertEqual(first.seeds, [42, 43])
        self.assertEqual(first.generated[0].size, (512, 512))
        self.assertEqual(first.postprocessed, [])
        np.testing.assert_array_equal(np.asarray(first.generated[0]), np.asarray(repeat.generated[0]))
        self.assertFalse(np.array_equal(np.asarray(first.generated[0]), np.asarray(first.generated[1])))


if __name__ == "__main__":
    unittest.main()
