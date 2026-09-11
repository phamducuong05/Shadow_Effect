"""CPU contract tests; no checkpoints or Hugging Face downloads required."""
import io
import tempfile
import unittest
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from gps_sdxl_inference import (
    DemoSettings, InputValidationError, InferenceResult, prepare_inputs,
    rasterize_shadow_box, select_mask_embeddings, merge_postprocess,
)
from api import create_app


def png(mode="RGB", size=(32, 32), color=120):
    image = Image.new(mode, size, color)
    stream = io.BytesIO()
    image.save(stream, "PNG")
    return stream.getvalue()


def uploads(mask=None):
    return {"image": ("photo.png", png(), "image/png"),
            "mask": ("object.png", mask or png("L", color=255), "image/png")}


class InputTests(unittest.TestCase):
    def test_image_mask_channels_and_binary_mask(self):
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[8:24, 8:24] = 200
        prepared = prepare_inputs(Image.new("RGB", (32, 32), (255, 0, 0)), Image.fromarray(mask))
        self.assertEqual(prepared.control.shape, (1, 4, 512, 512))
        self.assertEqual(prepared.pixels.shape, (1, 3, 512, 512))
        np.testing.assert_array_equal(np.unique(prepared.control[:, 3]), [0, 1])
        np.testing.assert_array_equal(prepared.pixels[0, :, 100, 100], [1, -1, -1])
        self.assertGreater(prepared.bbox[2], 0)

    def test_rejects_invalid_empty_and_mismatched_mask(self):
        for image, mask in [(b"bad", png("L")), (png(), png("L", color=0)),
                            (png(), png("L", size=(16, 16)))]:
            with self.subTest(image_type=type(image), mask_size=len(mask)):
                with self.assertRaises(InputValidationError):
                    prepare_inputs(image, mask)

    def test_shadow_box_actually_fills_mask_without_changing_input(self):
        box = np.array([32, 32, 20, 10, 0], dtype=np.float32)
        region = rasterize_shadow_box(np.zeros(5), box, size=64, rng=np.random.default_rng(0), jitter=0)
        self.assertEqual(region.shape, (1, 1, 64, 64))
        self.assertEqual(region[0, 0, 32, 32], 1)
        self.assertEqual(region[0, 0, 0, 0], 0)
        self.assertGreater(region.sum(), 100)
        np.testing.assert_array_equal(box, [32, 32, 20, 10, 0])

    def test_rotated_box_keeps_width_and_height_distinct(self):
        region = rasterize_shadow_box(np.zeros(5), np.array([32, 32, 24, 6, 90]),
                                      size=64, rng=np.random.default_rng(0), jitter=0)
        self.assertEqual(region[0, 0, 42, 32], 1)
        self.assertEqual(region[0, 0, 32, 42], 0)

    def test_embeddings_do_not_accumulate_between_calls(self):
        import torch
        scores = torch.arange(256).float().unsqueeze(0)
        table = {i: np.full(2048, i, dtype=np.float32) for i in range(256)}
        first = select_mask_embeddings(scores, table)
        second = select_mask_embeddings(scores, table)
        self.assertEqual(tuple(first.shape), (1, 64, 2048))
        self.assertEqual(set(first[0, :, 0].tolist()), set(range(192, 256)))
        torch.testing.assert_close(first, second)

    def test_postprocess_preserves_background(self):
        original = np.full((4, 4, 3), 200, dtype=np.uint8)
        generated = np.full((4, 4, 3), 60, dtype=np.uint8)
        prediction = np.full((4, 4, 4), -1, dtype=np.float32)
        prediction[1:3, 1:3, 3] = 1
        merged, mask = merge_postprocess(original, generated, prediction)
        np.testing.assert_array_equal(np.asarray(merged)[0, 0], [200, 200, 200])
        np.testing.assert_array_equal(np.asarray(merged)[1, 1], [60, 60, 60])
        self.assertEqual(np.asarray(mask)[1, 1], 255)


class FakeRunner:
    def generate(self, prepared, *, num_samples, num_steps, seed, apply_postprocess):
        return InferenceResult(
            generated=[Image.new("RGB", (512, 512), "red") for _ in range(num_samples)],
            postprocessed=[Image.new("RGB", (256, 256), "blue") for _ in range(num_samples)] if apply_postprocess else [],
            shadow_masks=[Image.new("L", (256, 256), 255) for _ in range(num_samples)] if apply_postprocess else [],
            seeds=[seed + i for i in range(num_samples)], peak_vram_gb=0,
        )


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.settings = DemoSettings(weights_dir=self.root / "weights", output_dir=self.root / "outputs")

    def tearDown(self):
        self.temp.cleanup()

    def test_docs_and_health_start_without_weights(self):
        with TestClient(create_app(self.settings)) as client:
            self.assertEqual(client.get("/docs").status_code, 200)
            health = client.get("/health").json()
            self.assertFalse(health["model_loaded"])
            self.assertFalse(health["weights_ready"])
            self.assertTrue(any("ip_adapter.ckpt" in name for name in health["missing_files"]))
            schema = client.get("/openapi.json").json()
            self.assertIn("multipart/form-data", schema["paths"]["/predict"]["post"]["requestBody"]["content"])

    def test_predict_saves_images_and_returns_downloadable_links(self):
        with TestClient(create_app(self.settings, runner_factory=lambda _: FakeRunner())) as client:
            response = client.post("/predict", files=uploads(), data={"seed": "42", "num_samples": "2"})
            self.assertEqual(response.status_code, 200, response.text)
            body = response.json()
            self.assertEqual(body["seeds"], [42, 43])
            self.assertEqual(len(body["generated"]), 2)
            image_response = client.get(body["generated"][0]["url"])
            self.assertEqual(image_response.status_code, 200)
            self.assertEqual(Image.open(io.BytesIO(image_response.content)).size, (512, 512))
            self.assertEqual(len(list(self.settings.output_dir.glob("*/*.png"))), 6)
            self.assertTrue(client.get("/health").json()["model_loaded"])

    def test_raw_only_does_not_return_postprocess_files(self):
        with TestClient(create_app(self.settings, runner_factory=lambda _: FakeRunner())) as client:
            response = client.post("/predict", files=uploads(), data={"apply_postprocess": "false"})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["postprocessed"], [])

    def test_invalid_input_rejected_before_loading_model(self):
        def unavailable(_):
            raise AssertionError("Invalid input must not load GPU models")
        with TestClient(create_app(self.settings, runner_factory=unavailable)) as client:
            response = client.post("/predict", files=uploads(png("L", color=0)))
            self.assertEqual(response.status_code, 422)
            self.assertIn("foreground", response.json()["detail"])
            self.assertEqual(client.post("/predict", files=uploads(), data={"num_samples": "0"}).status_code, 422)

    def test_missing_weights_are_actionable_and_retryable(self):
        with TestClient(create_app(self.settings)) as client:
            response = client.post("/predict", files=uploads())
            self.assertEqual(response.status_code, 503)
            self.assertIn("ip_adapter.ckpt", response.json()["detail"])
            self.assertFalse(client.get("/health").json()["model_loaded"])

    def test_gpu_failure_releases_request_slot(self):
        class FailingRunner(FakeRunner):
            def generate(self, *args, **kwargs):
                raise RuntimeError("CUDA out of memory")
        with TestClient(create_app(self.settings, runner_factory=lambda _: FailingRunner())) as client:
            for _ in range(2):
                response = client.post("/predict", files=uploads())
                self.assertEqual(response.status_code, 503)
                self.assertIn("memory", response.json()["detail"])

    def test_simultaneous_gpu_requests_return_busy(self):
        entered, release = Event(), Event()
        class BlockingRunner(FakeRunner):
            def generate(self, *args, **kwargs):
                entered.set()
                if not release.wait(10):
                    raise RuntimeError("test timeout")
                return super().generate(*args, **kwargs)
        with TestClient(create_app(self.settings, runner_factory=lambda _: BlockingRunner())) as client:
            with ThreadPoolExecutor(1) as pool:
                future = pool.submit(client.post, "/predict", files=uploads())
                try:
                    self.assertTrue(entered.wait(5))
                    self.assertEqual(client.post("/predict", files=uploads()).status_code, 409)
                finally:
                    release.set()
                self.assertEqual(future.result().status_code, 200)


if __name__ == "__main__":
    unittest.main()
