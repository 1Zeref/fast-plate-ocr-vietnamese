"""
FAST-PLATE-OCR-VIETNAMESE: Grammar Probability Masking (Bước 4.1)
Tạo mặt nạ xác suất (Probability Masking) dựa trên Máy trạng thái hữu hạn DFA
nhằm giải quyết triệt để các cặp ký tự dễ nhầm lẫn quang học:
(8 <-> B, 0 <-> D, 5 <-> S, 2 <-> Z, 1 <-> T) tại từng vị trí ký tự.
"""

import numpy as np
from typing import Set, Dict, Any, Optional

from src.grammar.constants import (
    DIGITS,
    CIVILIAN_LETTERS,
    VALID_LETTERS,
    VALID_PROVINCE_CODES,
    MILITARY_UNIT_CODES,
    DIPLOMATIC_SERIES,
    SPECIAL_SERIES,
)
from src.grammar.dfa import VietnamPlateDFA


# Danh mục các cặp ký tự dễ nhầm lẫn quang học thường gặp trong nhận dạng biển số
OPTICAL_CONFUSION_PAIRS = {
    "8": "B", "B": "8",
    "0": "D", "D": "0",
    "5": "S", "S": "5",
    "2": "Z", "Z": "2",
    "1": "T", "T": "1",
}


class DFAPositionMasker:
    """
    Bộ sinh mặt nạ xác suất ngữ pháp (Grammar Masker):
    Xác định tập ký tự hợp lệ cho bước $t$ dựa trên chuỗi tiền tố (prefix) đã sinh.
    """

    def __init__(
        self,
        alphabet: str = "0123456789ABCDEFGHKLMNPQRSTUVXYZ_",
        pad_char: str = "_",
        max_slots: int = 10,
    ):
        self.alphabet = alphabet
        self.pad_char = pad_char
        self.max_slots = max_slots
        self.vocab_size = len(alphabet)
        self.char2idx = {c: i for i, c in enumerate(alphabet)}
        self.idx2char = {i: c for i, c in enumerate(alphabet)}

        self.valid_provinces = set(VALID_PROVINCE_CODES.keys())
        self.military_codes = set(MILITARY_UNIT_CODES.keys())
        self.diplomatic_codes = set(DIPLOMATIC_SERIES.keys())
        self.special_codes = set(SPECIAL_SERIES.keys())

        # Tiền tính toán các tập ký tự khởi đầu
        self.prov_first_digits = set(k[0] for k in self.valid_provinces)
        self.mil_first_letters = set(k[0] for k in self.military_codes)
        self.all_digits = set(DIGITS)
        self.all_letters = set(VALID_LETTERS)

        self.dfa = VietnamPlateDFA()

    def get_allowed_characters(self, prefix: str, step: int) -> Set[str]:
        """
        Trả về tập hợp các ký tự hợp lệ được phép xuất hiện tại vị trí `step`
        dựa trên chuỗi tiền tố `prefix`.
        """
        # Quy tắc 1: Nếu đã xuất hiện ký tự padding '_', các vị trí còn lại BẮT BUỘC là '_'
        if self.pad_char in prefix:
            return {self.pad_char}

        # Vị trí 0: Chỉ có thể là số đầu mã tỉnh (1-9) hoặc chữ cái đầu mã quân đội
        if step == 0:
            return self.prov_first_digits | self.mil_first_letters

        # Vị trí 1:
        if step == 1:
            if prefix[0].isdigit():
                # Dân sự: Chữ số thứ 2 phải ghép thành mã tỉnh hợp lệ 11-99
                return set(k[1] for k in self.valid_provinces if k[0] == prefix[0])
            else:
                # Quân sự: Chữ cái thứ 2 phải ghép thành mã quân sự hợp lệ (TM, TC, QA...)
                return set(k[1] for k in self.military_codes if k[0] == prefix[0])

        # NHÁNH QUÂN SỰ: prefix[:2] là mã đơn vị BQP
        if len(prefix) >= 2 and prefix[:2] in self.military_codes:
            if step in (2, 3, 4, 5):
                # 4 chữ số bắt buộc
                return self.all_digits
            elif step == 6:
                # Chữ số thứ 5 hoặc kết thúc bằng padding '_'
                return self.all_digits | {self.pad_char}
            else:
                # Đã đủ 5 số, các slot còn lại là padding
                return {self.pad_char}

        # NHÁNH DÂN SỰ / NGOẠI GIAO: prefix[:2] là mã tỉnh
        if step == 2:
            # Vị trí 2 BẮT BUỘC là chữ cái (Sê-ri bắt đầu bằng chữ, KHÔNG THỂ LÀ SỐ)
            # Khử hoàn toàn lỗi nhầm 8 -> B, 0 -> D tại slot này!
            return self.all_letters

        if step == 3:
            # Vị trí 3: Có thể là chữ cái thứ 2 (30AA, 51NG, 29LD...) HOẶC chữ số (29B1, 30A-1...)
            return self.all_letters | self.all_digits

        if step == 4:
            # Vị trí 4: Chắc chắn là chữ số (bắt đầu dãy số hoặc số xe)
            return self.all_digits

        if step in (5, 6):
            # Các chữ số tiếp theo trong biển số
            return self.all_digits

        if step in (7, 8):
            # Có thể là chữ số thứ 4 hoặc 5, hoặc kết thúc biển bằng padding '_'
            return self.all_digits | {self.pad_char}

        if step == 9:
            # Slot cuối cùng: chữ số hoặc padding '_'
            return self.all_digits | {self.pad_char}

        return {self.pad_char}

    def compute_log_mask(self, prefix: str, step: int) -> np.ndarray:
        """
        Tạo mặt nạ log-xác suất kích thước [V]:
        - 0.0 cho các ký tự hợp lệ.
        - -inf (-1e9) cho các ký tự vi phạm ngữ pháp.
        """
        allowed = self.get_allowed_characters(prefix, step)
        mask = np.full(self.vocab_size, -1e9, dtype=np.float32)

        for char in allowed:
            if char in self.char2idx:
                mask[self.char2idx[char]] = 0.0

        return mask
