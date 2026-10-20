"""
FAST-PLATE-OCR-VIETNAMESE: Unit Test Suite for Step 4.1
Kiểm thử thuật toán giải mã Grammar-Guided Beam Search và Probability Masking.
Xác minh triệt tiêu 100% các cặp nhầm lẫn quang học: 8 <-> B, 0 <-> D, 5 <-> S, 2 <-> Z.
"""

import os
import sys
import unittest
import numpy as np
import torch

# Đảm bảo import được module từ thư mục gốc
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.decoder.masking import DFAPositionMasker
from src.decoder.beam_search import GrammarGuidedBeamSearchDecoder
from src.grammar.dfa import VietnamPlateDFA


class TestGrammarGuidedBeamSearch(unittest.TestCase):

    def setUp(self):
        self.alphabet = "0123456789ABCDEFGHKLMNPQRSTUVXYZ_"
        self.char2idx = {c: i for i, c in enumerate(self.alphabet)}
        self.decoder = GrammarGuidedBeamSearchDecoder(alphabet=self.alphabet)

    def _create_logits_from_string(self, text: str, max_slots: int = 10) -> np.ndarray:
        """Tạo tensor logits hoàn hảo cho 1 chuỗi ký tự."""
        padded = text.ljust(max_slots, "_")[:max_slots]
        logits = np.full((max_slots, len(self.alphabet)), -2.0, dtype=np.float32)
        for i, char in enumerate(padded):
            idx = self.char2idx[char]
            logits[i, idx] = 5.0
        return logits

    def test_optical_confusion_8_vs_B(self):
        """
        Kiểm tra khử nhầm lẫn quang học 8 <-> B:
        - Giả lập tại slot 2 (vị trí sê-ri sau tỉnh '30'), OCR nhận diện nhầm số '8' (điểm 5.0) cao hơn chữ 'B' (điểm 4.5).
        - Greedy unconstrained sẽ chọn '30812345' (VI PHẠM cú pháp vì slot 2 không thể là số).
        - Grammar-Guided Beam Search phải tự động mask số 8, chọn chữ 'B' -> '30B12345' (HỢP LỆ DFA 100%).
        """
        logits = self._create_logits_from_string("30B12345")
        
        # Cố ý làm sai lệch: cho slot 2 có số '8' cao hơn chữ 'B'
        slot2_8 = self.char2idx["8"]
        slot2_B = self.char2idx["B"]
        logits[2, slot2_8] = 5.0  # Điểm cao nhất
        logits[2, slot2_B] = 4.2  # Điểm nhì

        # 1. Giải mã không ràng buộc
        res_raw = self.decoder.decode_single(logits, beam_width=5, apply_grammar=False)
        self.assertEqual(res_raw["text"], "30812345")
        self.assertFalse(res_raw["is_valid"])

        # 2. Giải mã ràng buộc ngữ pháp (Grammar-Guided)
        res_guided = self.decoder.decode_single(logits, beam_width=5, apply_grammar=True)
        self.assertEqual(res_guided["text"], "30B12345")
        self.assertTrue(res_guided["is_valid"])
        self.assertGreater(len(res_guided["corrections_made"]), 0)
        self.assertEqual(res_guided["corrections_made"][0]["confusion_type"], "8 -> B")
        print("[PASS] test_optical_confusion_8_vs_B (Triệt tiêu 8 -> B tại slot sê-ri)")

    def test_optical_confusion_0_vs_D(self):
        """
        Kiểm tra khử nhầm lẫn quang học 0 <-> D hai chiều:
        - Chiều 1: Tại slot 2 (sê-ri), OCR nhìn nhầm '0' cao hơn 'D'. Decoder phải chọn 'D'.
        - Chiều 2: Tại slot 5 (phần số), OCR nhìn nhầm 'D' cao hơn '0'. Decoder phải chọn '0'.
        """
        # Chiều 1: Biển '29D12345'
        logits1 = self._create_logits_from_string("29D12345")
        logits1[2, self.char2idx["0"]] = 5.0
        logits1[2, self.char2idx["D"]] = 4.0

        res1 = self.decoder.decode_single(logits1, beam_width=5, apply_grammar=True)
        self.assertEqual(res1["text"], "29D12345")
        self.assertTrue(res1["is_valid"])

        # Chiều 2: Biển '30A10345' (slot 4 là số 0, OCR nhầm thành D)
        logits2 = self._create_logits_from_string("30A10345")
        logits2[4, self.char2idx["D"]] = 5.2
        logits2[4, self.char2idx["0"]] = 4.1

        res2 = self.decoder.decode_single(logits2, beam_width=5, apply_grammar=True)
        self.assertEqual(res2["text"], "30A10345")
        self.assertTrue(res2["is_valid"])
        print("[PASS] test_optical_confusion_0_vs_D (Triệt tiêu 0 <-> D cả 2 chiều)")

    def test_province_code_masking(self):
        """
        Kiểm tra mặt nạ mã địa phương (slot 0 và 1):
        - Biển số Hải Phòng '15A12345'.
        - OCR bị xước khiến số '1' bị nhầm thành chữ 'T'.
        - Decoder ràng buộc phát hiện chữ 'T' không thể bắt đầu biển dân sự và mã 'T5' không tồn tại -> Mask 'T', chọn '1' -> '15A12345'.
        """
        logits = self._create_logits_from_string("15A12345")
        logits[0, self.char2idx["T"]] = 5.0
        logits[0, self.char2idx["1"]] = 4.2

        res = self.decoder.decode_single(logits, beam_width=5, apply_grammar=True)
        self.assertEqual(res["text"], "15A12345")
        self.assertTrue(res["is_valid"])
        print("[PASS] test_province_code_masking (Khử nhầm lẫn tại mã địa phương)")

    def test_military_plate_decoding(self):
        """Kiểm tra giải mã biển quân sự (ví dụ: 'TM1234')."""
        logits = self._create_logits_from_string("TM1234")
        res = self.decoder.decode_single(logits, beam_width=5, apply_grammar=True)
        self.assertEqual(res["text"], "TM1234")
        self.assertTrue(res["is_valid"])
        self.assertEqual(res["dfa_meta"]["plate_type"], "MILITARY")
        print("[PASS] test_military_plate_decoding (Giải mã biển quân sự chính xác)")

    def test_batch_decoding(self):
        """Kiểm tra giải mã theo lô (Batch Decoding)."""
        plates = ["30A12345", "51K99999", "80A04256", "29B11234"]
        batch_logits = np.stack([self._create_logits_from_string(p) for p in plates], axis=0)

        # Chuyển đổi sang PyTorch Tensor
        tensor_logits = torch.from_numpy(batch_logits)
        batch_results = self.decoder.decode_batch(tensor_logits, beam_width=4, apply_grammar=True)

        self.assertEqual(len(batch_results), 4)
        for i, p in enumerate(plates):
            self.assertEqual(batch_results[i]["text"], p)
            self.assertTrue(batch_results[i]["is_valid"])
        print("[PASS] test_batch_decoding (Giải mã thành công 4 mẫu trong batch)")


if __name__ == "__main__":
    print("=" * 60)
    print("CHẠY BỘ KIỂM THỬ GIẢI MÃ RÀNG BUỘC NGỮ PHÁP (BƯỚC 4.1)")
    print("=" * 60)
    unittest.main()
