"""
FAST-PLATE-OCR-VIETNAMESE: Unit Test for ALPRService
Kiểm thử lớp dịch vụ backend ALPRService phục vụ WebUI và các cổng API.
"""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.service import ALPRService


class TestALPRService(unittest.TestCase):
    """Kiểm thử ALPRService."""

    @classmethod
    def setUpClass(cls):
        cls.service = ALPRService.get_instance(device="cpu", use_onnx=True)

    def test_01_process_image(self):
        """Kiểm tra nhận diện ảnh đơn lẻ và trả về Dataframe chuẩn."""
        dummy_img = np.full((384, 384, 3), 128, dtype=np.uint8)
        res = self.service.process_image(dummy_img, conf_threshold=0.25)

        self.assertIn("annotated_image", res)
        self.assertIn("detections", res)
        self.assertIn("dataframe", res)
        self.assertIn("latency_ms", res)
        self.assertGreater(res["latency_ms"], 0.0)

    def test_02_inspect_pipeline(self):
        """Kiểm tra bóc tách 5 công đoạn trung gian cho Tab Inspector."""
        dummy_img = np.full((384, 384, 3), 128, dtype=np.uint8)
        insp = self.service.inspect_pipeline(dummy_img, conf_threshold=0.25)

        self.assertIn("step1_crop", insp)
        self.assertIn("step2_corners", insp)
        self.assertIn("step3_warped", insp)
        self.assertIn("step4_virtual_strip", insp)
        self.assertIn("explanation", insp)
        self.assertEqual(insp["step4_virtual_strip"].shape, (64, 256, 3))

    def test_03_process_live_frame(self):
        """Kiểm tra xử lý frame camera trực tiếp và đo đạc FPS."""
        dummy_frame = np.full((480, 640, 3), 120, dtype=np.uint8)
        annotated, fps, stats = self.service.process_live_frame(dummy_frame)

        self.assertEqual(annotated.shape, (480, 640, 3))
        self.assertGreater(fps, 0.0)
        self.assertIn("Độ trễ toàn trình", stats)

    def test_04_edge_cases_robustness(self):
        """Kiểm tra xử lý các tình huống biên: ảnh đen, ảnh trắng, ảnh nhiễu."""
        black = np.zeros((200, 200, 3), dtype=np.uint8)
        res1 = self.service.process_image(black, conf_threshold=0.5)
        self.assertIn("annotated_image", res1)

        white = np.full((200, 200, 3), 255, dtype=np.uint8)
        res2 = self.service.process_image(white, conf_threshold=0.5)
        self.assertIn("annotated_image", res2)

        noise = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        insp = self.service.inspect_pipeline(noise, conf_threshold=0.5)
        self.assertIn("step1_crop", insp)
        self.assertIn("explanation", insp)


if __name__ == "__main__":
    unittest.main()
