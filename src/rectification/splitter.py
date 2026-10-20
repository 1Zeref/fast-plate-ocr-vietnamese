"""
FAST-PLATE-OCR-VIETNAMESE: Two-Line Splitter & Virtual 1-Line Strip Generator
Thuật toán phân tích hồ sơ chiếu ngang (Horizontal Projection Profile) để cắt tách 2 dòng
và nối không gian thành dải ảnh 1 dòng đơn giả lập (Virtual 1-Line Strip) chuẩn hóa cho OCR.
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional


class TwoLinePlateSplitter:
    """
    Bộ cắt tách biển số 2 dòng và ghép nối thành dải 1 dòng giả lập:
    - Tìm đường cắt giữa 2 dòng dựa trên Horizontal Projection Profile.
    - Chuẩn hóa tỷ lệ chiều cao của 2 dòng.
    - Nối không gian (Horizontal Concatenation) kèm padding phân cách.
    """

    def __init__(
        self,
        strip_height: int = 64,
        separator_width: int = 16,
        target_ocr_size: Optional[Tuple[int, int]] = (256, 64),
    ):
        self.strip_height = strip_height
        self.separator_width = separator_width
        self.target_ocr_size = target_ocr_size

    @staticmethod
    def find_split_boundary_y(rectified_square: np.ndarray) -> int:
        """
        Tìm ranh giới phân tách giữa dòng trên và dòng dưới (Split Y)
        bằng phân tích thung lũng trên biểu đồ chiếu ngang (Horizontal Projection Profile).
        """
        h, w = rectified_square.shape[:2]
        gray = cv2.cvtColor(rectified_square, cv2.COLOR_BGR2GRAY) if len(rectified_square.shape) == 3 else rectified_square

        # Áp dụng Otsu hoặc Adaptive Thresholding để bóc tách nét chữ
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Nếu nền ảnh là màu tối (ví dụ biển xanh hoặc biển đỏ chữ trắng), đảo ngược lại mask
        if np.mean(gray) < 100:
            binary = cv2.bitwise_not(binary)

        # Loại bỏ viền biển ngoài rìa (crop biên 10% mỗi phía)
        crop_y1, crop_y2 = int(h * 0.12), int(h * 0.88)
        crop_x1, crop_x2 = int(w * 0.08), int(w * 0.92)
        inner_bin = binary[crop_y1:crop_y2, crop_x1:crop_x2]

        # Tính biểu đồ chiếu ngang: tổng số pixel mực trên từng hàng
        proj = np.sum(inner_bin > 0, axis=1)

        # Khoảng tìm kiếm thung lũng (ở giữa biển số: 35% đến 65% chiều cao)
        search_start = int(len(proj) * 0.30)
        search_end = int(len(proj) * 0.70)
        search_region = proj[search_start:search_end]

        if len(search_region) > 0:
            local_min_idx = np.argmin(search_region)
            split_y = crop_y1 + search_start + local_min_idx
        else:
            split_y = int(h * 0.48) # Dự phòng theo tỷ lệ chuẩn dập khuôn

        # Giới hạn an toàn
        split_y = max(int(h * 0.35), min(int(h * 0.65), split_y))
        return split_y

    def create_virtual_1line_strip(
        self,
        rectified_square: np.ndarray,
        split_y: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Cắt tách dòng 1 và dòng 2, sau đó ghép nối thành dải 1 dòng đơn (Virtual 1-Line Strip).
        Trả về: (virtual_strip_image, metadata)
        """
        h, w = rectified_square.shape[:2]

        if split_y is None:
            split_y = self.find_split_boundary_y(rectified_square)

        # 1. Cắt dòng trên (Line 1) và dòng dưới (Line 2)
        line1 = rectified_square[0:split_y, :]
        line2 = rectified_square[split_y:h, :]

        # 2. Đồng bộ chiều cao về strip_height (64 px) trong khi giữ đúng tỷ lệ W/H
        h1, w1 = line1.shape[:2]
        h2, w2 = line2.shape[:2]

        target_h = self.strip_height
        new_w1 = max(10, int(w1 * (target_h / float(h1))))
        new_w2 = max(10, int(w2 * (target_h / float(h2))))

        resized_line1 = cv2.resize(line1, (new_w1, target_h), interpolation=cv2.INTER_LINEAR)
        resized_line2 = cv2.resize(line2, (new_w2, target_h), interpolation=cv2.INTER_LINEAR)

        # 3. Tạo đoạn đệm phân cách (Separator Padding) mang màu nền trung bình
        bg_sample = np.median(rectified_square[:10, :10], axis=(0, 1)).astype(np.uint8)
        separator = np.full((target_h, self.separator_width, 3), bg_sample, dtype=np.uint8)

        # 4. Ghép nối không gian theo chiều ngang: [Line 1 | Separator | Line 2]
        virtual_strip = np.hstack([resized_line1, separator, resized_line2])

        # 5. Nếu có chỉ định target_ocr_size, resize dải ảnh về kích thước mô hình
        final_strip = virtual_strip
        if self.target_ocr_size is not None:
            final_strip = cv2.resize(virtual_strip, self.target_ocr_size, interpolation=cv2.INTER_LINEAR)

        metadata = {
            "split_y": split_y,
            "split_ratio": round(split_y / float(h), 3),
            "line1_raw_shape": (h1, w1),
            "line2_raw_shape": (h2, w2),
            "virtual_strip_shape": final_strip.shape,
        }

        return final_strip, metadata
