"""
FAST-PLATE-OCR-VIETNAMESE: Unified Homography & Geometry Routing Pipeline
Pipeline tích hợp nắn phẳng xạ ảnh, định tuyến phân loại hình học và sinh Virtual 1-Line Strip.
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

from .homography import order_quad_corners, PerspectiveRectifier
from .router import GeometryRouter
from .splitter import TwoLinePlateSplitter


class PlateRectificationPipeline:
    """
    Pipeline hoàn chỉnh của Giai đoạn 2 (Bước 2.2):
    1. Tiếp nhận 4 điểm góc tứ giác và vùng ảnh.
    2. Định tuyến hình học (1-line vs 2-line).
    3. Nắn phối cảnh về hệ trục chuẩn tắc (Khử méo góc xiên).
    4. Đối với biển 2 dòng: Tách dòng và sinh dải 1 dòng giả lập (Virtual 1-Line Strip).
    """

    def __init__(
        self,
        target_ocr_size: Tuple[int, int] = (256, 64),
        aspect_ratio_threshold: float = 2.40,
    ):
        self.target_ocr_size = target_ocr_size
        self.rectifier = PerspectiveRectifier(
            target_size_1line=(512, 128),
            target_size_2line=(320, 240),
        )
        self.router = GeometryRouter(aspect_ratio_threshold=aspect_ratio_threshold)
        self.splitter = TwoLinePlateSplitter(
            strip_height=64,
            separator_width=16,
            target_ocr_size=target_ocr_size,
        )

    def process(
        self,
        image: np.ndarray,
        corners: List[List[float]],
        detector_layout_hint: Optional[str] = None,
        detector_confidence: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Xử lý toàn diện một phát hiện biển số từ Bước 2.1:
        Đầu vào:
        - image: Ảnh gốc (toàn cảnh hoặc crop xe)
        - corners: 4 điểm góc của biển số
        - detector_layout_hint: Gợi ý layout từ mô hình detector
        Đầu ra: Dict chứa:
        - 'layout': '1-line' hoặc '2-line'
        - 'rectified_raw': Ảnh biển số nắn phẳng chuẩn tắc
        - 'ocr_input_strip': Dải ảnh 1 dòng sẵn sàng đưa vào mô hình OCR (kích thước WxH cố định)
        - 'homography_matrix': Ma trận biến đổi 3x3
        - 'router_metadata': Thông tin phân luồng hình học
        - 'splitter_metadata': Thông tin tách dòng (nếu là biển 2 dòng)
        """
        # 1. Định tuyến hình học
        route_res = self.router.route(
            corners=corners,
            detector_layout_hint=detector_layout_hint,
            detector_confidence=detector_confidence,
        )
        is_two_line = route_res["is_two_line"]
        layout_name = route_res["layout"]

        # 2. Nắn phẳng xạ ảnh khử méo phối cảnh
        rectified_raw, H_matrix = self.rectifier.rectify(
            image=image,
            corners=corners,
            is_two_line=is_two_line,
        )

        splitter_meta = None

        if is_two_line:
            # 3. Biển 2 dòng: Tách dòng và tạo dải 1 dòng giả lập
            ocr_strip, splitter_meta = self.splitter.create_virtual_1line_strip(rectified_raw)
        else:
            # Biển 1 dòng: Resize trực tiếp về target_ocr_size
            ocr_strip = cv2.resize(rectified_raw, self.target_ocr_size, interpolation=cv2.INTER_LINEAR)

        return {
            "layout": layout_name,
            "is_two_line": is_two_line,
            "rectified_raw": rectified_raw,
            "ocr_input_strip": ocr_strip,
            "homography_matrix": H_matrix,
            "router_metadata": route_res,
            "splitter_metadata": splitter_meta,
        }
