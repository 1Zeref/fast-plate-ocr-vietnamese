"""
FAST-PLATE-OCR-VIETNAMESE: Unit Test Suite for Step 5.2 (Temporal Consensus Engine)
Kiểm thử bộ hợp nhất quyết định đa khung hình, thuật toán bỏ phiếu có trọng số
và cơ chế triệt tiêu lỗi chớp đèn / bóng lóa tức thời.
"""

import os
import sys
import unittest
import numpy as np

# Đảm bảo import được module từ thư mục gốc
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.tracking.consensus import TemporalConsensusEngine


class TestTemporalConsensusStep52(unittest.TestCase):

    def setUp(self):
        self.alphabet = "0123456789ABCDEFGHKLMNPQRSTUVXYZ_"
        self.char2idx = {c: i for i, c in enumerate(self.alphabet)}
        self.engine = TemporalConsensusEngine(alphabet=self.alphabet)

    def _create_mock_logits(self, text: str, max_slots: int = 10, noise_level: float = 0.1) -> np.ndarray:
        padded = text.ljust(max_slots, "_")[:max_slots]
        logits = np.full((max_slots, len(self.alphabet)), -2.0, dtype=np.float32)
        for i, char in enumerate(padded):
            idx = self.char2idx[char]
            logits[i, idx] = 5.0
        if noise_level > 0:
            logits += np.random.normal(0, noise_level, logits.shape).astype(np.float32)
        return logits

    def test_glare_and_flicker_suppression(self):
        """
        Kiểm tra triệt tiêu ngoại lai do chớp đèn hoặc bóng lóa tức thời:
        - 4 frame rõ nét (Q >= 0.80): '30A12345'
        - 1 frame bị đèn pha chói lóa (Q = 0.30): OCR bị nhầm số 3 thành số 8 -> '30A12845'
        - Consensus Engine phải phát hiện ngoại lai, bỏ phiếu có trọng số và xuất ra chuẩn '30A12345'.
        """
        frames_data = [
            {"frame_idx": 10, "text": "30A12345", "quality_score": 0.85, "confidence": 0.95},
            {"frame_idx": 11, "text": "30A12345", "quality_score": 0.82, "confidence": 0.92},
            {"frame_idx": 12, "text": "30A12845", "quality_score": 0.30, "confidence": 0.55}, # Lóa đèn pha!
            {"frame_idx": 13, "text": "30A12345", "quality_score": 0.88, "confidence": 0.96},
            {"frame_idx": 14, "text": "30A12345", "quality_score": 0.80, "confidence": 0.91},
        ]

        result = self.engine.consolidate_tracklet(track_id=1, frames_data=frames_data)

        # 1. Kết quả cuối cùng phải là chuỗi chính xác duy nhất
        self.assertEqual(result["plate_text"], "30A12345")
        self.assertTrue(result["is_valid"])
        self.assertGreater(result["agreement_rate"], 0.85)

        # 2. Frame 12 (lóa đèn) phải bị nhận diện và đưa vào danh sách ngoại lai bị triệt tiêu
        self.assertEqual(len(result["outliers_suppressed"]), 1)
        outlier = result["outliers_suppressed"][0]
        self.assertEqual(outlier["frame_idx"], 12)
        self.assertEqual(outlier["frame_pred"], "30A12845")

        print(f"[PASS] test_glare_and_flicker_suppression (Triệt tiêu thành công frame lóa={outlier['frame_idx']}, Đồng thuận: {result['agreement_percent']}%)")

    def test_soft_probability_matrix_fusion(self):
        """Kiểm tra hợp nhất ma trận xác suất mềm đa khung hình."""
        # 3 frame ma trận logits của biển '29B11234'
        frames_data = [
            {
                "frame_idx": 1,
                "logits": self._create_mock_logits("29B11234", noise_level=0.2),
                "quality_score": 0.90,
                "confidence": 0.95,
            },
            {
                "frame_idx": 2,
                "logits": self._create_mock_logits("29B11234", noise_level=0.3),
                "quality_score": 0.85,
                "confidence": 0.90,
            },
            {
                "frame_idx": 3,
                "logits": self._create_mock_logits("29B11234", noise_level=0.2),
                "quality_score": 0.88,
                "confidence": 0.93,
            },
        ]

        result = self.engine.consolidate_tracklet(track_id=2, frames_data=frames_data)
        self.assertEqual(result["plate_text"], "29B11234")
        self.assertTrue(result["is_valid"])
        self.assertGreater(result["consensus_confidence"], 0.80)
        print(f"[PASS] test_soft_probability_matrix_fusion (Hợp nhất Softmax xác suất thành công: {result['plate_text']})")

    def test_grammar_enforcement_in_consensus(self):
        """Kiểm tra kết quả hợp nhất luôn đảm bảo 100% quy chuẩn pháp lý DFA."""
        frames_data = [
            {"frame_idx": 5, "text": "51K99999", "quality_score": 0.90, "confidence": 0.95},
            {"frame_idx": 6, "text": "51K99999", "quality_score": 0.85, "confidence": 0.90},
        ]
        result = self.engine.consolidate_tracklet(track_id=3, frames_data=frames_data)
        self.assertEqual(result["plate_text"], "51K99999")
        self.assertTrue(result["is_valid"])
        self.assertEqual(result["dfa_meta"]["province_name"], "TP. Hồ Chí Minh")
        print("[PASS] test_grammar_enforcement_in_consensus (Kiểm định cú pháp DFA thành công)")


if __name__ == "__main__":
    print("=" * 60)
    print("CHẠY BỘ KIỂM THỬ HỢP NHẤT ĐA KHUNG HÌNH (BƯỚC 5.2)")
    print("=" * 60)
    unittest.main()
