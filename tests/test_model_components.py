"""Small real neural components on CPU; this is not a trained model quality test."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch


class ModelTests(unittest.TestCase):
    def test_tensor_checkpoint_accepts_numpy_scalar_metadata(self):
        from gps_sdxl_inference import load_tensor_checkpoint

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "weights.pth"
            torch.save({"net": {"weight": torch.ones(1)}, "score": np.float64(1.5)}, checkpoint)
            loaded = load_tensor_checkpoint(checkpoint)

        self.assertEqual(loaded["score"], np.float64(1.5))
        torch.testing.assert_close(loaded["net"]["weight"], torch.ones(1))

    def test_classifier_can_load_checkpoint_without_resnet_download(self):
        from base_network import MaskCls
        with patch("torch.hub.download_url_to_file", side_effect=AssertionError("Unexpected download")):
            classifier = MaskCls(num_classes=256, pretrained=False)
        self.assertEqual(classifier.resnet.conv1.in_channels, 4)
        self.assertEqual(classifier.resnet.fc.out_features, 256)

    def test_postprocess_network_runs_at_its_trained_resolution(self):
        from train_post_process_predictor import PostProcess
        torch.set_num_threads(2)
        model = PostProcess().eval()
        with torch.inference_mode():
            output = model.post_process_net(torch.zeros(1, 7, 256, 256), timesteps=torch.zeros(1))
        self.assertEqual(tuple(output.shape), (1, 4, 256, 256))
        self.assertTrue(torch.isfinite(output).all())

    def test_native_attention_runs_without_xformers(self):
        from attention_processor import IPAttnProcessor2_0
        from diffusers.models.attention_processor import Attention
        torch.set_num_threads(2)
        attn = Attention(query_dim=32, cross_attention_dim=32, heads=4, dim_head=8)
        processor = IPAttnProcessor2_0(32, 32, num_tokens=4)
        attn.set_processor(processor)
        with torch.no_grad():
            output = attn(torch.randn(1, 16, 32), encoder_hidden_states=torch.randn(1, 141, 32))
        self.assertEqual(tuple(output.shape), (1, 16, 32))
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
