"""
FAST-PLATE-OCR-VIETNAMESE: Deterministic Finite Automaton (DFA)
Máy trạng thái hữu hạn kiểm định tính hợp lệ cú pháp của biển số xe Việt Nam.
"""

from typing import Tuple, Optional, Dict, Any
from .constants import (
    DIGITS,
    CIVILIAN_LETTERS,
    VALID_LETTERS,
    EXCLUDED_LETTERS,
    VALID_PROVINCE_CODES,
    MILITARY_UNIT_CODES,
    DIPLOMATIC_SERIES,
    SPECIAL_SERIES,
)


class VietnamPlateDFA:
    """
    Máy trạng thái hữu hạn (DFA) xác thực cú pháp biển số xe Việt Nam.
    Tuân thủ quy định:
    - Biển ô tô dân sự & nhà nước (5 số & 4 số)
    - Biển mô tô, xe máy (5 số & 4 số, sê-ri 2 chữ cái hoặc 1 chữ cái + 1 số)
    - Biển xe chuyên dùng (LD, DA, R, RM, MK, CD, HC, MD)
    - Biển quân đội (2 chữ cái BQP + 4 hoặc 5 số)
    - Biển ngoại giao & tổ chức quốc tế (NG, QT, NN, CV + 3 số nước + 2 số thứ tự)
    """

    def __init__(self):
        self.valid_provinces = set(VALID_PROVINCE_CODES.keys())
        self.military_codes = set(MILITARY_UNIT_CODES.keys())
        self.diplomatic_codes = set(DIPLOMATIC_SERIES.keys())
        self.special_codes = set(SPECIAL_SERIES.keys())

    @staticmethod
    def clean_text(text: str) -> str:
        """Loại bỏ ký tự phân tách và chuẩn hóa chữ hoa."""
        return text.strip().upper().replace("-", "").replace(".", "").replace(" ", "").replace("_", "")

    def validate(self, plate_str: str) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Duyệt máy trạng thái cho chuỗi biển số đã được chuẩn hóa.
        Trả về: (is_valid, message, metadata)
        """
        raw_clean = self.clean_text(plate_str)
        metadata: Dict[str, Any] = {
            "raw_input": plate_str,
            "normalized": raw_clean,
            "plate_type": None,
            "province_code": None,
            "province_name": None,
            "series": None,
            "number": None,
            "military_unit": None,
            "diplomatic_org": None,
        }

        if not raw_clean:
            return False, "Chuỗi biển số rỗng.", metadata

        # Kiểm tra ký tự bị loại trừ tuyệt đối (I, O, J, W)
        for char in raw_clean:
            if char in EXCLUDED_LETTERS:
                return (
                    False,
                    f"Ký tự '{char}' không hợp lệ theo quy chuẩn biển số xe Việt Nam (loại trừ: I, O, J, W).",
                    metadata,
                )
            if char not in VALID_LETTERS and char not in DIGITS:
                return (
                    False,
                    f"Ký tự lạ '{char}' không nằm trong bảng chữ cái quy chuẩn.",
                    metadata,
                )

        # -------------------------------------------------------------
        # NHÁNH 1: BIỂN XE QUÂN SỰ (Bắt đầu bằng 2 chữ cái quân đội)
        # -------------------------------------------------------------
        if len(raw_clean) >= 2 and raw_clean[:2] in self.military_codes:
            unit_code = raw_clean[:2]
            digits_part = raw_clean[2:]
            
            # Kiểm tra phần số: phải là 4 hoặc 5 chữ số
            if not digits_part.isdigit():
                return False, f"Biển quân sự [{unit_code}] phải có đuôi là 4 hoặc 5 chữ số.", metadata
            if len(digits_part) not in (4, 5):
                return (
                    False,
                    f"Độ dài phần số của biển quân sự không hợp lệ ({len(digits_part)} số). Cần 4 hoặc 5 chữ số.",
                    metadata,
                )

            metadata["plate_type"] = "MILITARY"
            metadata["series"] = unit_code
            metadata["military_unit"] = MILITARY_UNIT_CODES[unit_code]
            metadata["number"] = digits_part
            return True, f"Biển quân sự hợp lệ: {MILITARY_UNIT_CODES[unit_code]}", metadata

        # -------------------------------------------------------------
        # NHÁNH 2: BIỂN DÂN SỰ / NGOẠI GIAO (Bắt đầu bằng 2 chữ số mã tỉnh)
        # -------------------------------------------------------------
        if len(raw_clean) < 4:
            return False, "Độ dài biển số quá ngắn.", metadata

        prov_candidate = raw_clean[:2]
        if not prov_candidate.isdigit():
            return False, f"Hai ký tự đầu '{prov_candidate}' không phải mã địa phương hợp lệ.", metadata

        if prov_candidate not in self.valid_provinces:
            return (
                False,
                f"Mã địa phương '{prov_candidate}' không tồn tại trong danh mục quy chuẩn 11-99.",
                metadata,
            )

        metadata["province_code"] = prov_candidate
        metadata["province_name"] = VALID_PROVINCE_CODES[prov_candidate]
        rest = raw_clean[2:]

        # -------------------------------------------------------------
        # NHÁNH 2.1: NGOẠI GIAO & TỔ CHỨC QUỐC TẾ (NG, QT, NN, CV)
        # -------------------------------------------------------------
        if len(rest) >= 2 and rest[:2] in self.diplomatic_codes:
            diplo_code = rest[:2]
            num_part = rest[2:]
            # Ngoại giao: 3 chữ số mã quốc gia + 2 chữ số thứ tự = 5 chữ số
            if not num_part.isdigit() or len(num_part) != 5:
                return (
                    False,
                    f"Biển ngoại giao/quốc tế [{diplo_code}] yêu cầu 5 chữ số (3 mã nước + 2 thứ tự). Nhận được: '{num_part}'.",
                    metadata,
                )
            
            metadata["plate_type"] = "DIPLOMATIC"
            metadata["series"] = diplo_code
            metadata["diplomatic_org"] = DIPLOMATIC_SERIES[diplo_code]
            metadata["number"] = num_part
            return True, f"Biển ngoại giao/quốc tế hợp lệ: {DIPLOMATIC_SERIES[diplo_code]}", metadata

        # -------------------------------------------------------------
        # NHÁNH 2.2: BIỂN RƠ-MOÓC / SƠ-MI RƠ-MOÓC (R hoặc RM)
        # -------------------------------------------------------------
        if rest.startswith("RM") and len(rest) > 2 and rest[2:].isdigit():
            metadata["plate_type"] = "TRAILER"
            metadata["series"] = "RM"
            metadata["number"] = rest[2:]
            return True, "Biển rơ-moóc / sơ-mi rơ-moóc hợp lệ (sê-ri RM).", metadata
        elif rest.startswith("R") and len(rest) > 1 and rest[1:].isdigit():
            num_digits = len(rest[1:])
            if num_digits in (4, 5):
                metadata["plate_type"] = "TRAILER"
                metadata["series"] = "R"
                metadata["number"] = rest[1:]
                return True, "Biển rơ-moóc / sơ-mi rơ-moóc hợp lệ (sê-ri R).", metadata

        # -------------------------------------------------------------
        # NHÁNH 2.3: BIỂN XE MÁY ĐIỆN (MD / MĐ)
        # Hỗ trợ cả MD + số phân khu + 5 số (ví dụ: 29MD1-12345) hoặc MD + 5 số (29MD-12345)
        # -------------------------------------------------------------
        if rest.startswith("MD"):
            tail = rest[2:]
            if tail.isdigit() and len(tail) in (4, 5, 6):
                metadata["plate_type"] = "ELECTRIC_MOTORCYCLE"
                metadata["series"] = "MD" if len(tail) in (4, 5) else f"MD{tail[0]}"
                metadata["number"] = tail if len(tail) in (4, 5) else tail[1:]
                return True, "Biển xe máy điện hợp lệ (sê-ri MD).", metadata

        # -------------------------------------------------------------
        # NHÁNH 2.4: BIỂN CHUYÊN DÙNG KHÁC (LD, DA, MK, CD, HC)
        # -------------------------------------------------------------
        if len(rest) >= 2 and rest[:2] in self.special_codes:
            spec_code = rest[:2]
            num_part = rest[2:]
            if num_part.isdigit() and len(num_part) in (4, 5):
                metadata["plate_type"] = "SPECIAL"
                metadata["series"] = spec_code
                metadata["number"] = num_part
                return True, f"Biển chuyên dùng hợp lệ ({SPECIAL_SERIES[spec_code]}).", metadata

        # -------------------------------------------------------------
        # NHÁNH 2.4: MÔ TÔ SÊ-RI 1 CHỮ CÁI + 1 SỐ (ví dụ: 29B1 12345, 59G2 1234)
        # HOẶC XE TẢI CON (ví dụ: 29C1 12345)
        # -------------------------------------------------------------
        if (
            len(rest) >= 6
            and rest[0] in CIVILIAN_LETTERS
            and rest[1].isdigit()
            and rest[2:].isdigit()
        ):
            num_digits = len(rest[2:])
            if num_digits in (4, 5):
                metadata["plate_type"] = "MOTORCYCLE_OR_LIGHT_TRUCK"
                metadata["series"] = rest[:2]
                metadata["number"] = rest[2:]
                return (
                    True,
                    f"Biển mô tô / xe chuyên trách hợp lệ ({rest[:2]}-{rest[2:]}).",
                    metadata,
                )

        # -------------------------------------------------------------
        # NHÁNH 2.5: Ô TÔ HOẶC MÔ TÔ SÊ-RI 2 CHỮ CÁI (TT 24/2023: 30AA 12345, 59BA 12345)
        # -------------------------------------------------------------
        if (
            len(rest) >= 6
            and rest[0] in CIVILIAN_LETTERS
            and rest[1] in CIVILIAN_LETTERS
            and rest[2:].isdigit()
        ):
            num_digits = len(rest[2:])
            if num_digits in (4, 5):
                metadata["plate_type"] = "CIVILIAN_2_LETTERS"
                metadata["series"] = rest[:2]
                metadata["number"] = rest[2:]
                return (
                    True,
                    f"Biển sê-ri 2 chữ cái hợp lệ ({rest[:2]}-{rest[2:]}).",
                    metadata,
                )

        # -------------------------------------------------------------
        # NHÁNH 2.6: Ô TÔ DÂN SỰ TIÊU CHUẨN SÊ-RI 1 CHỮ CÁI (30A 12345, 51K 99999, 29S 1234)
        # -------------------------------------------------------------
        if (
            len(rest) >= 5
            and rest[0] in CIVILIAN_LETTERS
            and rest[1:].isdigit()
        ):
            num_digits = len(rest[1:])
            if num_digits in (4, 5):
                metadata["plate_type"] = "CIVILIAN_CAR"
                metadata["series"] = rest[0]
                metadata["number"] = rest[1:]
                return (
                    True,
                    f"Biển ô tô tiêu chuẩn hợp lệ ({rest[0]}-{rest[1:]}).",
                    metadata,
                )

        return False, f"Không khớp với bất kỳ mẫu quy chuẩn nào của biển số Việt Nam (chuỗi: '{raw_clean}').", metadata

    def validate_double_line(self, line1: str, line2: str) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Kiểm định biển số 2 dòng (Biển vuông / Biển xe máy).
        Dòng 1: Ký hiệu địa phương + Sê-ri (hoặc Đơn vị quân sự)
        Dòng 2: Nhóm số thứ tự
        """
        l1 = self.clean_text(line1)
        l2 = self.clean_text(line2)
        combined = l1 + l2
        is_valid, msg, meta = self.validate(combined)
        meta["line_format"] = "DOUBLE_LINE"
        meta["line1_raw"] = line1
        meta["line2_raw"] = line2
        if is_valid:
            return True, f"[Biển 2 dòng] {msg}", meta
        return False, f"[Biển 2 dòng không hợp lệ] {msg}", meta
