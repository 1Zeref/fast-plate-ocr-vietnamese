"""
FAST-PLATE-OCR-VIETNAMESE: Unit Test for Dual-Dataset End-to-End Pipeline
Kiểm thử tích hợp song song 2 bộ dữ liệu:
1. bomaich/vnlicenseplate: Phát hiện xe và biển số toàn cảnh.
2. topkek69/vietnamese-license-plate-ocr: Nhận dạng ký tự và giải mã ràng buộc ngữ pháp.
"""

import os
import sys
import unittest
import numpy as np

# Đảm bảo import được module từ thư mục gốc
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.pipeline import VietnamesePlateOCRPipeline


class TestDualDatasetPipeline(unittest.TestCase):

    def setUp(self):
        self.pipeline = VietnamesePlateOCRPipeline(
            detector_weights="weights/best_detector.pt",
            ocr_weights="weights/fast_plate_ocr_vn.pt",
            conf_threshold=0.30,
        )

    def test_pipeline_on_full_scene_bomaich(self):
        """Kiểm tra quy trình trên ảnh toàn cảnh thực tế từ dataset bomaich/vnlicenseplate."""
        scene_dir = "data/detection_dataset/valid/images"
        sample_img = None
        if os.path.exists(scene_dir) and len(os.listdir(scene_dir)) > 0:
            preferred = os.path.join(scene_dir, "clip4_new_59.jpg")
            if os.path.exists(preferred):
                sample_img = preferred
            else:
                sample_img = os.path.join(scene_dir, os.listdir(scene_dir)[0])
        elif os.path.exists("samples/traffic_scene_sample.jpg"):
            sample_img = "samples/traffic_scene_sample.jpg"

        if sample_img:
            results = self.pipeline.predict(sample_img, beam_width=3, conf_threshold=0.25)
            self.assertIsInstance(results, list)
            self.assertGreater(len(results), 0)
            r = results[0]
            self.assertIn("plate_text", r)
            self.assertIn("bbox", r)
            self.assertIn("corners", r)
            self.assertTrue(r["is_valid_grammar"])
            print(f"[PASS] test_pipeline_on_full_scene_bomaich (Phát hiện biển: {r['plate_text']}, Conf: {r['overall_confidence']:.2f})")
        else:
            print("[SKIP] Chưa tìm thấy ảnh scene bomaich")

    def test_pipeline_on_cropped_plate_topkek69(self):
        """Kiểm tra quy trình trên ảnh cắt sát mép từ dataset topkek69/vietnamese-license-plate-ocr."""
        cropped_dir = "data/kaggle_dataset/images"
        if os.path.exists(cropped_dir) and len(os.listdir(cropped_dir)) > 0:
            sample_img = os.path.join(cropped_dir, os.listdir(cropped_dir)[0])
            results = self.pipeline.predict(sample_img, beam_width=3)
            self.assertIsInstance(results, list)
            self.assertGreater(len(results), 0)
            r = results[0]
            self.assertIn("plate_text", r)
            self.assertTrue(r["is_valid_grammar"])
            print(f"[PASS] test_pipeline_on_cropped_plate_topkek69 (Nhận diện thành công: {r['plate_text']})")
        else:
            print("[SKIP] Chưa tìm thấy ảnh crop topkek69")

    def test_draw_predictions(self):
        """Kiểm tra hàm vẽ kết quả trực quan."""
        dummy_img = np.full((400, 600, 3), 100, dtype=np.uint8)
        mock_results = [{
            "plate_text": "30A12345",
            "overall_confidence": 0.95,
            "bbox": [100, 100, 250, 160],
            "corners": [[100, 100], [250, 100], [250, 160], [100, 160]],
        }]
        vis = self.pipeline.draw_predictions(dummy_img, mock_results)
        self.assertEqual(vis.shape, dummy_img.shape)
        # Ảnh phải có pixel thay đổi do vẽ bbox và text
        self.assertTrue(np.any(vis != dummy_img))
        print("[PASS] test_draw_predictions (Vẽ bounding box và nhãn trực quan thành công)")


if __name__ == "__main__":
    print("=" * 60)
    print("CHẠY BỘ KIỂM THỬ PIPELINE SONG SONG 2 DATASET (DETECTION + OCR)")
    print("=" * 60)
    unittest.main()
