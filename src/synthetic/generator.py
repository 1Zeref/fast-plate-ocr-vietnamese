"""
FAST-PLATE-OCR-VIETNAMESE: Synthetic String Generator
Sinh chuỗi biển số xe ngẫu nhiên tuân thủ 100% luật cú pháp và mã địa phương.
"""

import os
import sys
import random
from typing import Dict, Any, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.grammar.constants import (
    DIGITS,
    CIVILIAN_LETTERS,
    VALID_PROVINCE_CODES,
    MILITARY_UNIT_CODES,
    DIPLOMATIC_SERIES,
    SPECIAL_SERIES,
)

# Danh sách danh mục tra cứu nhanh
PROVINCE_LIST = list(VALID_PROVINCE_CODES.keys())
MILITARY_LIST = list(MILITARY_UNIT_CODES.keys())
DIPLOMATIC_LIST = list(DIPLOMATIC_SERIES.keys())
SPECIAL_LIST = ["LD", "DA", "R", "RM", "MK", "CD", "HC"]


def _random_digits(length: int) -> str:
    """Sinh chuỗi số ngẫu nhiên có độ dài length (từ 00..01 đến 99..99)."""
    return "".join(random.choice(DIGITS) for _ in range(length))


def generate_plate_specification() -> Dict[str, Any]:
    """
    Sinh ngẫu nhiên một mẫu biển số Việt Nam với đầy đủ cấu trúc và màu sắc:
    Tỷ lệ phân bổ:
    - 45% Ô tô cá nhân / doanh nghiệp (Trắng)
    - 25% Xe máy cá nhân (Trắng, 2 dòng)
    - 15% Xe kinh doanh vận tải (Vàng: taxi, xe tải, xe công nghệ)
    - 5%  Xe cơ quan công vụ (Xanh)
    - 5%  Xe quân sự (Đỏ)
    - 5%  Ngoại giao & xe chuyên dùng đặc thù
    """
    category = random.choices(
        ["CIVILIAN_CAR", "MOTORCYCLE", "COMMERCIAL", "STATE", "MILITARY", "DIPLOMATIC", "SPECIAL"],
        weights=[0.45, 0.25, 0.15, 0.05, 0.05, 0.025, 0.025],
        k=1,
    )[0]

    prov = random.choice(PROVINCE_LIST)
    spec: Dict[str, Any] = {}

    if category == "CIVILIAN_CAR":
        # Ô tô dân sự: Trắng, chữ đen
        is_square = random.random() < 0.25  # 75% biển dài, 25% biển vuông phía sau
        is_5_digits = random.random() < 0.90  # 90% biển 5 số, 10% biển 4 số cũ
        
        # Sê-ri: 70% 1 chữ cái (30A, 51K), 30% 2 chữ cái (30AA, 51AB)
        if random.random() < 0.70:
            series = random.choice(CIVILIAN_LETTERS)
        else:
            series = random.choice(CIVILIAN_LETTERS) + random.choice(CIVILIAN_LETTERS)

        if is_5_digits:
            p1, p2 = _random_digits(3), _random_digits(2)
            num_clean = f"{p1}{p2}"
            num_display = f"{p1}.{p2}"
        else:
            num_clean = _random_digits(4)
            num_display = num_clean

        spec["plate_type"] = "CIVILIAN_CAR"
        spec["bg_color_bgr"] = (250, 250, 250)      # Trắng phản quang
        spec["text_color_bgr"] = (20, 20, 20)        # Đen
        spec["is_two_line"] = is_square
        spec["layout"] = "SQUARE" if is_square else "LONG"
        spec["province"] = prov
        spec["series"] = series
        spec["number"] = num_clean
        spec["normalized"] = f"{prov}{series}{num_clean}"

        if is_square:
            spec["line1"] = f"{prov}{series}"
            spec["line2"] = num_display
            spec["display_text"] = f"{prov}{series}\n{num_display}"
        else:
            spec["line1"] = f"{prov}{series}-{num_display}"
            spec["line2"] = ""
            spec["display_text"] = f"{prov}{series}-{num_display}"

    elif category == "MOTORCYCLE":
        # Xe máy cá nhân: 99% biển 2 dòng (vuông), Nền trắng chữ đen
        is_5_digits = random.random() < 0.85
        # Sê-ri: 50% kiểu cũ 1 chữ + 1 số (29-B1), 40% kiểu mới 2 chữ cái (29-AA), 10% xe máy điện (29-MD1)
        sub_type = random.random()
        if sub_type < 0.50:
            series = f"{random.choice(CIVILIAN_LETTERS)}{random.choice(DIGITS)}"
        elif sub_type < 0.90:
            series = f"{random.choice(CIVILIAN_LETTERS)}{random.choice(CIVILIAN_LETTERS)}"
        else:
            series = f"MD{random.choice(DIGITS)}"

        if is_5_digits:
            p1, p2 = _random_digits(3), _random_digits(2)
            num_clean = f"{p1}{p2}"
            num_display = f"{p1}.{p2}"
        else:
            num_clean = _random_digits(4)
            num_display = num_clean

        spec["plate_type"] = "MOTORCYCLE"
        spec["bg_color_bgr"] = (250, 250, 250)
        spec["text_color_bgr"] = (20, 20, 20)
        spec["is_two_line"] = True
        spec["layout"] = "SQUARE"
        spec["province"] = prov
        spec["series"] = series
        spec["number"] = num_clean
        spec["normalized"] = f"{prov}{series}{num_clean}"
        spec["line1"] = f"{prov}-{series}"
        spec["line2"] = num_display
        spec["display_text"] = f"{prov}-{series}\n{num_display}"

    elif category == "COMMERCIAL":
        # Xe kinh doanh vận tải: Biển nền vàng, chữ đen
        is_square = random.random() < 0.35  # Xe tải/khách thường lắp biển vuông phía sau
        series = random.choice(CIVILIAN_LETTERS)
        p1, p2 = _random_digits(3), _random_digits(2)
        num_clean = f"{p1}{p2}"
        num_display = f"{p1}.{p2}"

        spec["plate_type"] = "COMMERCIAL_YELLOW"
        spec["bg_color_bgr"] = (20, 190, 240)        # Nền vàng giao thông (BGR)
        spec["text_color_bgr"] = (20, 20, 20)        # Đen
        spec["is_two_line"] = is_square
        spec["layout"] = "SQUARE" if is_square else "LONG"
        spec["province"] = prov
        spec["series"] = series
        spec["number"] = num_clean
        spec["normalized"] = f"{prov}{series}{num_clean}"

        if is_square:
            spec["line1"] = f"{prov}{series}"
            spec["line2"] = num_display
            spec["display_text"] = f"{prov}{series}\n{num_display}"
        else:
            spec["line1"] = f"{prov}{series}-{num_display}"
            spec["line2"] = ""
            spec["display_text"] = f"{prov}{series}-{num_display}"

    elif category == "STATE":
        # Xe cơ quan công vụ: Nền xanh dương, chữ trắng
        is_square = random.random() < 0.20
        # Thường dùng sê-ri A, B, C, D, E, M...
        series = random.choice(["A", "B", "C", "D", "E", "F", "G", "H", "K", "L", "M"])
        p1, p2 = _random_digits(3), _random_digits(2)
        num_clean = f"{p1}{p2}"
        num_display = f"{p1}.{p2}"

        spec["plate_type"] = "STATE_BLUE"
        spec["bg_color_bgr"] = (160, 60, 15)         # Xanh dương đậm (BGR)
        spec["text_color_bgr"] = (250, 250, 250)     # Chữ trắng
        spec["is_two_line"] = is_square
        spec["layout"] = "SQUARE" if is_square else "LONG"
        spec["province"] = prov
        spec["series"] = series
        spec["number"] = num_clean
        spec["normalized"] = f"{prov}{series}{num_clean}"

        if is_square:
            spec["line1"] = f"{prov}{series}"
            spec["line2"] = num_display
            spec["display_text"] = f"{prov}{series}\n{num_display}"
        else:
            spec["line1"] = f"{prov}{series}-{num_display}"
            spec["line2"] = ""
            spec["display_text"] = f"{prov}{series}-{num_display}"

    elif category == "MILITARY":
        # Xe quân đội: Nền đỏ tươi, chữ trắng
        unit = random.choice(MILITARY_LIST)
        is_square = random.random() < 0.30
        is_5_digits = random.random() < 0.40  # 40% 5 số, 60% 4 số
        
        if is_5_digits:
            p1, p2 = _random_digits(2), _random_digits(3)
            num_clean = f"{p1}{p2}"
            num_display = f"{p1}-{p2}"
        else:
            p1, p2 = _random_digits(2), _random_digits(2)
            num_clean = f"{p1}{p2}"
            num_display = f"{p1}-{p2}"

        spec["plate_type"] = "MILITARY_RED"
        spec["bg_color_bgr"] = (25, 25, 185)         # Đỏ quân sự (BGR)
        spec["text_color_bgr"] = (250, 250, 250)     # Trắng
        spec["is_two_line"] = is_square
        spec["layout"] = "SQUARE" if is_square else "LONG"
        spec["province"] = None
        spec["series"] = unit
        spec["number"] = num_clean
        spec["normalized"] = f"{unit}{num_clean}"

        if is_square:
            spec["line1"] = unit
            spec["line2"] = num_display
            spec["display_text"] = f"{unit}\n{num_display}"
        else:
            spec["line1"] = f"{unit}-{num_display}"
            spec["line2"] = ""
            spec["display_text"] = f"{unit}-{num_display}"

    elif category == "DIPLOMATIC":
        # Ngoại giao: Nền trắng, NG chữ đỏ hoặc toàn bộ đen
        diplo_code = random.choice(DIPLOMATIC_LIST)
        country_code = _random_digits(3)
        order_num = _random_digits(2)
        num_clean = f"{country_code}{order_num}"
        is_square = random.random() < 0.20

        spec["plate_type"] = "DIPLOMATIC"
        spec["bg_color_bgr"] = (250, 250, 250)
        spec["text_color_bgr"] = (20, 20, 20)
        spec["is_two_line"] = is_square
        spec["layout"] = "SQUARE" if is_square else "LONG"
        spec["province"] = prov
        spec["series"] = diplo_code
        spec["number"] = num_clean
        spec["normalized"] = f"{prov}{diplo_code}{num_clean}"

        if is_square:
            spec["line1"] = f"{prov}-{diplo_code}"
            spec["line2"] = f"{country_code}-{order_num}"
            spec["display_text"] = f"{prov}-{diplo_code}\n{country_code}-{order_num}"
        else:
            spec["line1"] = f"{prov}-{diplo_code}-{country_code}-{order_num}"
            spec["line2"] = ""
            spec["display_text"] = f"{prov}-{diplo_code}-{country_code}-{order_num}"

    else:  # SPECIAL
        # Rơ mooc, máy kéo, liên doanh
        series = random.choice(SPECIAL_LIST)
        is_square = random.random() < 0.40
        num_clean = _random_digits(5)
        p1, p2 = num_clean[:3], num_clean[3:]
        num_display = f"{p1}.{p2}"

        spec["plate_type"] = "SPECIAL"
        spec["bg_color_bgr"] = (250, 250, 250)
        spec["text_color_bgr"] = (20, 20, 20)
        spec["is_two_line"] = is_square
        spec["layout"] = "SQUARE" if is_square else "LONG"
        spec["province"] = prov
        spec["series"] = series
        spec["number"] = num_clean
        spec["normalized"] = f"{prov}{series}{num_clean}"

        if is_square:
            spec["line1"] = f"{prov}{series}"
            spec["line2"] = num_display
            spec["display_text"] = f"{prov}{series}\n{num_display}"
        else:
            spec["line1"] = f"{prov}{series}-{num_display}"
            spec["line2"] = ""
            spec["display_text"] = f"{prov}{series}-{num_display}"

    return spec
