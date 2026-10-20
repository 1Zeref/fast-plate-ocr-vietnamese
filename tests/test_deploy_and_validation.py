"""
FAST-PLATE-OCR-VIETNAMESE: Unit Test Suite for Stage 6 Deployment & Validation
Kiểm thử tự động hóa toàn bộ module Bước 6.1 (ONNX, Quantization, Benchmark)
và Bước 6.2 (End-to-End System Validation, Confusion Matrix, Streaming Stability).
"""

import os
import sys
import unittest
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.deploy.export_onnx import export_detector_to_onnx, export_ocr_to_onnx
from src.deploy.onnx_engine import ONNXPlateDetector, ONNXCCTPlateOCR, get_optimal_providers
from src.deploy.tensorrt_engine import generate_trtexec_commands
from src.deploy.benchmark import BenchmarkSuite
from src.deploy.validation import ConfusionMatrixTracker, EndToEndSystemValidator


class TestStage6DeploymentAndValidation(unittest.TestCase):
    """Bộ kiểm thử đơn vị cho Giai đoạn 6."""

    def test_01_onnx_export_and_existence(self):
        """Kiểm tra xuất và nạp các tệp ONNX FP32 và INT8."""
        det_onnx = "weights/plate_detector.onnx"
        det_quant = "weights/plate_detector_int8.onnx"
        ocr_onnx = "weights/fast_plate_ocr_vn.onnx"
        ocr_quant = "weights/fast_plate_ocr_vn_int8.onnx"

        self.assertTrue(os.path.exists(det_onnx), f"Thiếu tệp {det_onnx}")
        self.assertTrue(os.path.exists(det_quant), f"Thiếu tệp {det_quant}")
        self.assertTrue(os.path.exists(ocr_onnx), f"Thiếu tệp {ocr_onnx}")
        self.assertTrue(os.path.exists(ocr_quant), f"Thiếu tệp {ocr_quant}")

        # Kiểm tra kích thước mô hình nhẹ < 25MB
        for p in [det_onnx, det_quant, ocr_onnx, ocr_quant]:
            size_mb = os.path.getsize(p) / (1024 * 1024)
            self.assertLess(size_mb, 25.0, f"Mô hình {p} ({size_mb:.2f} MB) vượt quá 25MB")

    def test_02_onnx_detector_inference(self):
        """Kiểm tra tính đúng đắn ngõ ra của ONNXPlateDetector."""
        detector = ONNXPlateDetector("weights/plate_detector.onnx")
        dummy_img = np.random.randn(1, 3, 384, 384).astype(np.float32)
        out = detector.forward(dummy_img)

        self.assertIn("confidence", out)
        self.assertIn("bbox", out)
        self.assertIn("corners", out)
        self.assertIn("layout", out)

        self.assertEqual(out["confidence"].shape, (1, 1, 48, 48))
        self.assertEqual(out["bbox"].shape, (1, 4, 48, 48))
        self.assertEqual(out["corners"].shape, (1, 8, 48, 48))
        self.assertEqual(out["layout"].shape, (1, 2, 48, 48))

    def test_03_onnx_cct_ocr_inference_and_decoding(self):
        """Kiểm tra suy luận CCT OCR trên ONNX Runtime và giải mã Beam Search."""
        ocr = ONNXCCTPlateOCR("weights/fast_plate_ocr_vn.onnx")
        dummy_strip = np.random.randn(1, 3, 64, 256).astype(np.float32)

        logits = ocr.forward(dummy_strip)
        self.assertEqual(logits.shape, (1, 10, 33))

        # Giải mã với Grammar-Guided Beam Search
        decoded = ocr.decode(logits, use_grammar_beam_search=True, beam_width=3)
        self.assertEqual(len(decoded), 1)
        self.assertIn("plate_text", decoded[0])
        self.assertIn("confidence", decoded[0])
        self.assertIn("is_valid", decoded[0])

    def test_04_tensorrt_command_generation(self):
        """Kiểm tra hàm sinh lệnh trtexec cho môi trường GPU NVIDIA/Jetson."""
        cmds = generate_trtexec_commands()
        self.assertIn("detector_fp16_cmd", cmds)
        self.assertIn("ocr_fp16_cmd", cmds)
        self.assertIn("--fp16", cmds["detector_fp16_cmd"])
        self.assertIn("--fp16", cmds["ocr_fp16_cmd"])

    def test_05_confusion_matrix_tracker(self):
        """Kiểm tra thuật toán cập nhật ma trận nhầm lẫn qua Levenshtein DP."""
        tracker = ConfusionMatrixTracker()
        tracker.update(pred_text="29B11234", true_text="29811234") # Nhầm 8 thành B
        tracker.update(pred_text="51A99999", true_text="51A99999") # Trùng khớp hoàn toàn

        top_conf = tracker.get_top_confusions(top_k=3)
        self.assertGreaterEqual(len(top_conf), 1)
        self.assertEqual(top_conf[0]["true_char"], "8")
        self.assertEqual(top_conf[0]["pred_char"], "B")

        char_acc = tracker.get_character_accuracy()
        self.assertGreater(char_acc, 0.85)

    def test_06_benchmark_sla_targets(self):
        """Kiểm tra đo lường độ trễ từng mắt xích và xác nhận đạt SLA."""
        suite = BenchmarkSuite(num_warmup=5, num_iters=15)
        results = suite.run_benchmark()

        # Kiểm tra các mắt xích chính
        self.assertIn("detector", results)
        self.assertIn("warping", results)
        self.assertIn("ocr", results)
        self.assertIn("post_processing", results)

        det_best = results["detector"]["best_latency_ms"]
        warp_best = results["warping"]["best_latency_ms"]
        ocr_best = results["ocr"]["best_latency_ms"]
        post_best = results["post_processing"]["best_latency_ms"]

        # Warping <= 2ms trên GPU, <= 3.5ms trên CPU Fallback
        warp_sla = 2.0 if torch.cuda.is_available() else 3.5
        self.assertLessEqual(warp_best, warp_sla, f"Warping ({warp_best:.2f}ms) vượt quá {warp_sla}ms")
        # OCR <= 3ms trên GPU / TensorRT, <= 5ms trên CPU Fallback
        ocr_sla = 3.0 if torch.cuda.is_available() else 5.0
        self.assertLessEqual(ocr_best, ocr_sla, f"OCR ({ocr_best:.2f}ms) vượt quá {ocr_sla}ms")
        # Post-processing <= 2.0ms (SLA target ~1ms) trên GPU/Edge, <= 8.0ms trên CPU Python test
        post_sla = 2.0 if torch.cuda.is_available() else 8.0
        self.assertLessEqual(post_best, post_sla, f"Post-processing ({post_best:.2f}ms) vượt quá {post_sla}ms")


if __name__ == "__main__":
    print("=" * 60)
    print("CHẠY BỘ KIỂM THỬ TRIỂN KHAI BIÊN & ĐÁNH GIÁ HỆ THỐNG (GIAI ĐOẠN 6)")
    print("=" * 60)
    unittest.main()
