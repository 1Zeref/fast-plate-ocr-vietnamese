"""
FAST-PLATE-OCR-VIETNAMESE: Geometry Router Module
Định tuyến và phân luồng biển số thành dạng 1 dòng (1-line) hoặc 2 dòng (2-line).
Kết hợp đặc trưng hình học tỷ lệ khung hình (Aspect Ratio) và xác suất của mô hình học sâu.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
from .homography import order_quad_corners, calculate_quad_aspect_ratio


class GeometryRouter:
    """
    Bộ định tuyến phân loại hình học biển số:
    - 1-line: Biển dài (ô tô trước/sau tiêu chuẩn, xe ngoại giao dài). Tỷ lệ quy chuẩn 520x110 mm (~4.7:1).
    - 2-line: Biển vuông (xe máy, ô tô vuông sau, xe tải/khách). Tỷ lệ quy chuẩn 330x165 mm (~2:1) hoặc 280x200 mm.
    """

    def __init__(self, aspect_ratio_threshold: float = 2.40):
        # Ngưỡng phân tách tỷ lệ hình học giữa biển dài và biển vuông
        self.ar_threshold = aspect_ratio_threshold

    def route(
        self,
        corners: list,
        detector_layout_hint: Optional[str] = None,
        detector_confidence: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Phân luồng biển số.
        Đầu vào:
        - corners: 4 điểm góc tứ giác
        - detector_layout_hint: Gợi ý từ detector ('1_LINE', '2_LINE' hoặc None)
        - detector_confidence: Độ tin cậy của detector
        Đầu ra:
        - 'layout': '1-line' hoặc '2-line'
        - 'aspect_ratio': Tỷ lệ W/H tính được từ 4 góc
        - 'confidence': Độ tin cậy của quyết định
        - 'method': Phương pháp ra quyết định ('GEOMETRY_ONLY', 'ENSEMBLE_AGREED', 'DETECTOR_DOMINANT')
        """
        ordered = order_quad_corners(corners)
        ar = calculate_quad_aspect_ratio(ordered)

        geom_choice = "1-line" if ar >= self.ar_threshold else "2-line"

        if detector_layout_hint is None:
            return {
                "layout": geom_choice,
                "aspect_ratio": round(ar, 3),
                "confidence": 0.90 if abs(ar - self.ar_threshold) > 0.4 else 0.75,
                "method": "GEOMETRY_ONLY",
                "is_two_line": (geom_choice == "2-line"),
            }

        # Chuẩn hóa detector hint
        det_choice = "2-line" if "2" in detector_layout_hint.lower() else "1-line"

        if geom_choice == det_choice:
            # Hai phương pháp đồng thuận tuyệt đối
            conf = min(0.99, max(0.85, detector_confidence))
            decision_method = "ENSEMBLE_AGREED"
            final_layout = geom_choice
        else:
            # Có mâu thuẫn (thường xảy ra khi camera góc xiên cực đoan làm co ngắn cạnh biển)
            # Nếu tỷ lệ góc rất xa ngưỡng (>3.0 hoặc <1.8), ưu tiên hình học
            if ar > 3.0:
                final_layout = "1-line"
                conf = 0.88
                decision_method = "GEOMETRY_OVERRIDE"
            elif ar < 1.7:
                final_layout = "2-line"
                conf = 0.88
                decision_method = "GEOMETRY_OVERRIDE"
            else:
                # Vùng biên tranh chấp (1.7 <= ar <= 3.0), ưu tiên nhãn detector
                final_layout = det_choice
                conf = detector_confidence * 0.85
                decision_method = "DETECTOR_DOMINANT"

        return {
            "layout": final_layout,
            "aspect_ratio": round(ar, 3),
            "confidence": round(conf, 3),
            "method": decision_method,
            "is_two_line": (final_layout == "2-line"),
        }
