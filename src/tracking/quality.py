"""
FAST-PLATE-OCR-VIETNAMESE: Optical Frame Quality Assessor (Bước 5.1)
Bộ lọc đánh giá chất lượng quang học của từng frame biển số trong tracklet:
- Biến thiên toán tử Laplace (Laplacian Variance): Đo độ sắc nét, phát hiện nhòe chuyển động.
- Góc xoay và độ méo hình học (Geometric Skew & Angle Penalty): Đánh giá góc nghiêng camera.
- Độ phân giải và kích thước vùng biển số (Resolution Score).
- Phân phối độ sáng quang học (Photometric Score).
"""

import math
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple


class FrameQualityAssessor:
    """
    Bộ đánh giá chất lượng khung hình quang học:
    Gán nhãn điểm chất lượng toàn diện Q in [0.0, 1.0] cho từng khung hình trích xuất.
    """

    def __init__(
        self,
        sharpness_threshold: float = 85.0,
        min_width: int = 60,
        min_height: int = 20,
    ):
        self.sharpness_threshold = sharpness_threshold
        self.min_width = min_width
        self.min_height = min_height

    def compute_laplacian_sharpness(self, image: np.ndarray) -> Tuple[float, float]:
        """
        Đo biến thiên toán tử Laplace để xác định độ sắc nét.
        Trả về: (laplacian_var, normalized_sharpness [0.0, 1.0])
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        lap = cv2.Laplacian(gray, cv2.CV_64F)
        var = float(lap.var())

        # Chuẩn hóa về [0.0, 1.0] qua hàm phi tuyến sigmoid-like
        # Tại var = 85, điểm khoảng 0.50. Tại var = 300+, điểm tiến sát 1.0.
        norm_sharpness = var / (var + self.sharpness_threshold + 1e-6)
        norm_sharpness = min(1.0, max(0.0, norm_sharpness))

        return round(var, 2), round(norm_sharpness, 4)

    @staticmethod
    def compute_geometric_skew_score(corners: Optional[List[List[float]]]) -> float:
        """
        Đánh giá góc nghiêng và độ méo xạ ảnh của tứ giác 4 góc:
        Tứ giác càng vuông vắn, ít góc xiên thì điểm càng cao (tiến sát 1.0).
        """
        if not corners or len(corners) != 4:
            return 0.80  # Điểm mặc định nếu chỉ có bounding box 2D

        pts = np.array(corners, dtype=np.float32)
        tl, tr, br, bl = pts[0], pts[1], pts[2], pts[3]

        # 1. Tính độ lệch góc cạnh trên so với phương ngang
        dx_top = tr[0] - tl[0]
        dy_top = tr[1] - tl[1]
        top_angle_deg = abs(math.degrees(math.atan2(dy_top, max(1e-4, dx_top))))

        # 2. Tính tỷ lệ cạnh đối xứng (Top / Bottom, Left / Right)
        len_top = np.linalg.norm(tr - tl)
        len_bottom = np.linalg.norm(br - bl)
        len_left = np.linalg.norm(bl - tl)
        len_right = np.linalg.norm(br - tr)

        ratio_tb = min(len_top, len_bottom) / max(1e-4, max(len_top, len_bottom))
        ratio_lr = min(len_left, len_right) / max(1e-4, max(len_left, len_right))
        symmetry_score = (ratio_tb + ratio_lr) / 2.0

        # Phạt góc nghiêng: góc nghiêng < 10 độ -> phạt nhẹ, > 35 độ -> phạt nặng
        angle_penalty = max(0.0, 1.0 - (top_angle_deg / 45.0) ** 1.5)

        geo_score = 0.6 * symmetry_score + 0.4 * angle_penalty
        return round(float(min(1.0, max(0.0, geo_score))), 4)

    def compute_resolution_score(self, width: int, height: int) -> float:
        """Đánh giá kích thước pixel vùng biển số."""
        if width < self.min_width or height < self.min_height:
            return 0.20
        # Điểm tăng dần tới độ phân giải lý tưởng (W >= 150, H >= 45)
        w_score = min(1.0, width / 160.0)
        h_score = min(1.0, height / 50.0)
        return round(float(0.5 * w_score + 0.5 * h_score), 4)

    @staticmethod
    def compute_photometric_score(image: np.ndarray) -> float:
        """Đánh giá độ tương phản và phân bổ ánh sáng."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        mean_val = float(np.mean(gray))
        std_val = float(np.std(gray))

        # Phạt cháy sáng (mean > 220) hoặc quá tối (mean < 35)
        if mean_val < 35.0:
            light_score = max(0.1, mean_val / 35.0)
        elif mean_val > 220.0:
            light_score = max(0.1, (255.0 - mean_val) / 35.0)
        else:
            light_score = 1.0

        # Độ tương phản (std_val >= 35 là tương phản tốt)
        contrast_score = min(1.0, std_val / 40.0)

        return round(float(0.6 * light_score + 0.4 * contrast_score), 4)

    def evaluate_quality(
        self,
        image_crop: np.ndarray,
        corners: Optional[List[List[float]]] = None,
        bbox: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Tính toán chỉ số chất lượng tổng thể Q cho một khung hình:
        Q = 0.45 * Sharpness + 0.25 * Geometry + 0.15 * Resolution + 0.15 * Photometric.
        """
        h, w = image_crop.shape[:2]
        lap_var, sharpness = self.compute_laplacian_sharpness(image_crop)
        geo_score = self.compute_geometric_skew_score(corners)
        res_score = self.compute_resolution_score(w, h)
        photo_score = self.compute_photometric_score(image_crop)

        overall_q = (
            0.45 * sharpness
            + 0.25 * geo_score
            + 0.15 * res_score
            + 0.15 * photo_score
        )
        overall_q = round(float(min(1.0, max(0.0, overall_q))), 4)

        is_sharp = lap_var >= self.sharpness_threshold
        is_acceptable = overall_q >= 0.35 and is_sharp

        return {
            "overall_quality": overall_q,
            "quality_percent": round(overall_q * 100, 2),
            "laplacian_var": lap_var,
            "sharpness": sharpness,
            "geometric_score": geo_score,
            "resolution_score": res_score,
            "photometric_score": photo_score,
            "is_sharp": is_sharp,
            "is_acceptable": is_acceptable,
            "crop_dimensions": (w, h),
        }
