"""
FAST-PLATE-OCR-VIETNAMESE: High-Fidelity Plate Renderer
Kết xuất phôi biển số xe chân thực dựa trên phông chữ và kích thước quy chuẩn.
"""

import os
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Any, Tuple, List

# Đường dẫn tới thư mục font
FONT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "assets", "fonts"))
FONTS = {
    "square_modern": os.path.join(FONT_DIR, "plate_square_modern.ttf"),
    "round_bold": os.path.join(FONT_DIR, "plate_round_bold.ttf"),
    "round_narrow": os.path.join(FONT_DIR, "plate_round_narrow.ttf"),
}


def get_random_font(size: int, style_preference: str = "random") -> ImageFont.FreeTypeFont:
    """Lấy ngẫu nhiên font nét vuông mới hoặc nét tròn cũ."""
    if style_preference == "square":
        fpath = FONTS["square_modern"]
    elif style_preference == "round":
        fpath = FONTS["round_bold"]
    else:
        fpath = random.choice(list(FONTS.values()))
    
    if os.path.exists(fpath):
        return ImageFont.truetype(fpath, size)
    return ImageFont.load_default()


def draw_plate_border(img: np.ndarray, color_bgr: Tuple[int, int, int], thickness: int = 4, radius: int = 15):
    """Vẽ đường viền bo góc quy chuẩn của biển số xe."""
    h, w = img.shape[:2]
    cv2.rectangle(
        img,
        (8, 8),
        (w - 8, h - 8),
        color_bgr,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )


def draw_screws(img: np.ndarray, is_square: bool):
    """Vẽ các đinh ốc / bù lông bắt biển vào xe (ngẫu nhiên 2 hoặc 4 đinh ốc)."""
    h, w = img.shape[:2]
    screws = []
    
    if is_square:
        # Biển vuông: Thường bắt 2 ốc ở 2 góc trên hoặc 2 bên sườn
        if random.random() < 0.7:
            screws = [(int(w * 0.2), int(h * 0.18)), (int(w * 0.8), int(h * 0.18))]
        else:
            screws = [
                (int(w * 0.12), int(h * 0.15)),
                (int(w * 0.88), int(h * 0.15)),
                (int(w * 0.12), int(h * 0.85)),
                (int(w * 0.88), int(h * 0.85)),
            ]
    else:
        # Biển dài: Thường bắt 2 ốc ở khoảng 1/3 và 2/3 chiều ngang
        screws = [(int(w * 0.28), int(h * 0.5)), (int(w * 0.72), int(h * 0.5))]

    for sx, sy in screws:
        radius = random.randint(7, 11)
        # Bóng đinh ốc
        cv2.circle(img, (sx + 2, sy + 2), radius, (50, 50, 50), -1, cv2.LINE_AA)
        # Thân kim loại tròn
        core_color = random.choice([(190, 190, 195), (140, 140, 145), (100, 100, 105)])
        cv2.circle(img, (sx, sy), radius, core_color, -1, cv2.LINE_AA)
        # Rãnh bắt ốc (+) hoặc (-)
        slot_len = int(radius * 0.7)
        angle = random.randint(0, 180)
        rad = np.deg2rad(angle)
        x1 = int(sx - slot_len * np.cos(rad))
        y1 = int(sy - slot_len * np.sin(rad))
        x2 = int(sx + slot_len * np.cos(rad))
        y2 = int(sy + slot_len * np.sin(rad))
        cv2.line(img, (x1, y1), (x2, y2), (40, 40, 40), 2, cv2.LINE_AA)


class LicensePlateRenderer:
    """Renderer phôi ảnh biển số tiêu chuẩn."""

    def __init__(self):
        # Kích thước canvas vẽ cơ sở (độ phân giải cao để đảm bảo độ nét trước nhiễu)
        self.canvas_long = (520, 120)     # (Width, Height) biển dài
        self.canvas_square = (340, 200)   # (Width, Height) biển vuông

    def render(self, spec: Dict[str, Any]) -> np.ndarray:
        """
        Kết xuất ảnh phôi biển số dựa trên thông số đặc tả `spec`.
        Trả về ảnh numpy array (BGR).
        """
        is_square = spec["is_two_line"]
        target_w, target_h = self.canvas_square if is_square else self.canvas_long
        bg_bgr = spec["bg_color_bgr"]
        text_bgr = spec["text_color_bgr"]

        # 1. Tạo ảnh nền với màu tương ứng
        base_img = np.full((target_h, target_w, 3), bg_bgr, dtype=np.uint8)

        # 2. Vẽ viền bo biển
        border_color = (text_bgr[0] // 2, text_bgr[1] // 2, text_bgr[2] // 2) if text_bgr[0] > 100 else (30, 30, 30)
        draw_plate_border(base_img, border_color, thickness=random.choice([3, 4, 5]))

        # Chuyển sang PIL để render text sắc nét
        pil_img = Image.fromarray(cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        text_rgb = (text_bgr[2], text_bgr[1], text_bgr[0])

        font_style = random.choice(["square", "round", "random"])

        if is_square:
            # Biển 2 dòng
            l1_text = spec.get("line1", "")
            l2_text = spec.get("line2", "")

            # Cân chỉnh kích thước font phù hợp
            font_size_l1 = int(target_h * 0.38)
            font_size_l2 = int(target_h * 0.42)
            font_l1 = get_random_font(font_size_l1, font_style)
            font_l2 = get_random_font(font_size_l2, font_style)

            # Tính tọa độ căn giữa dòng 1
            bbox1 = font_l1.getbbox(l1_text)
            w1 = bbox1[2] - bbox1[0]
            x1 = max(10, (target_w - w1) // 2)
            y1 = int(target_h * 0.12)

            # Tính tọa độ căn giữa dòng 2
            bbox2 = font_l2.getbbox(l2_text)
            w2 = bbox2[2] - bbox2[0]
            x2 = max(10, (target_w - w2) // 2)
            y2 = int(target_h * 0.50)

            draw.text((x1, y1), l1_text, font=font_l1, fill=text_rgb)
            draw.text((x2, y2), l2_text, font=font_l2, fill=text_rgb)

        else:
            # Biển 1 dòng
            full_text = spec.get("line1", "")
            font_size = int(target_h * 0.65)
            font = get_random_font(font_size, font_style)

            bbox = font.getbbox(full_text)
            w = bbox[2] - bbox[0]
            x = max(15, (target_w - w) // 2)
            y = int((target_h - (bbox[3] - bbox[1])) // 2) - 5

            draw.text((x, y), full_text, font=font, fill=text_rgb)

        # Chuyển ngược lại OpenCV BGR
        rendered_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # 3. Vẽ đinh ốc ngẫu nhiên (tỷ lệ 60% biển có đinh ốc)
        if random.random() < 0.60:
            draw_screws(rendered_cv, is_square)

        return rendered_cv
