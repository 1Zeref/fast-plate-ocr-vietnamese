"""
FAST-PLATE-OCR-VIETNAMESE: Perspective Rectification & Homography Module
Khử hoàn toàn méo phối cảnh của biển số xe bằng phép biến đổi xạ ảnh (Homography).
Tự động chuẩn hóa thứ tự 4 góc (TL, TR, BR, BL) và nắn phẳng về hệ trục chuẩn tắc.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


def order_quad_corners(corners: List[List[float]]) -> np.ndarray:
    """
    Sắp xếp 4 điểm góc bất kỳ theo thứ tự chuẩn hình học:
    0: Top-Left (TL)
    1: Top-Right (TR)
    2: Bottom-Right (BR)
    3: Bottom-Left (BL)
    
    Phương pháp:
    - Tổng (x + y): Điểm có tổng nhỏ nhất là Top-Left; lớn nhất là Bottom-Right.
    - Hiệu (y - x): Điểm có hiệu nhỏ nhất là Top-Right; lớn nhất là Bottom-Left.
    """
    pts = np.array(corners, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError(f"Yêu cầu mảng 4 điểm góc (4, 2), nhận được: {pts.shape}")

    ordered = np.zeros((4, 2), dtype=np.float32)

    # Tổng x + y
    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)] # Top-Left
    ordered[2] = pts[np.argmax(s)] # Bottom-Right

    # Hiệu y - x (hoặc x - y)
    diff = np.diff(pts, axis=1) # y - x
    ordered[1] = pts[np.argmin(diff)] # Top-Right
    ordered[3] = pts[np.argmax(diff)] # Bottom-Left

    return ordered


def calculate_quad_aspect_ratio(ordered_corners: np.ndarray) -> float:
    """
    Tính tỷ lệ khung hình (Aspect Ratio = Width / Height) của tứ giác lồi:
    Dựa trên trung bình độ dài Euclid của 2 cạnh ngang chia cho 2 cạnh dọc.
    """
    tl, tr, br, bl = ordered_corners

    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    avg_width = (width_top + width_bottom) / 2.0

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    avg_height = (height_left + height_right) / 2.0

    if avg_height <= 1e-4:
        return 1.0
    return float(avg_width / avg_height)


class PerspectiveRectifier:
    """
    Bộ nắn phẳng xạ ảnh (Homography Rectifier) đưa biển số về hệ trục chuẩn tắc.
    """

    def __init__(
        self,
        target_size_1line: Tuple[int, int] = (512, 128),  # (Width, Height) biển dài chuẩn
        target_size_2line: Tuple[int, int] = (320, 240),  # (Width, Height) biển vuông chuẩn
    ):
        self.target_size_1line = target_size_1line
        self.target_size_2line = target_size_2line

    def rectify(
        self,
        image: np.ndarray,
        corners: List[List[float]],
        is_two_line: bool = False,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Khử hoàn toàn góc nghiêng và warp biển số về kích thước chuẩn tắc.
        Trả về: (rectified_image, homography_matrix)
        """
        ordered = order_quad_corners(corners)

        if target_size is not None:
            dst_w, dst_h = target_size
        else:
            dst_w, dst_h = self.target_size_2line if is_two_line else self.target_size_1line

        dst_pts = np.float32([
            [0, 0],
            [dst_w - 1, 0],
            [dst_w - 1, dst_h - 1],
            [0, dst_h - 1],
        ])

        # Tính ma trận phối cảnh 3x3
        homography_matrix = cv2.getPerspectiveTransform(ordered, dst_pts)

        # Warp phối cảnh với phép nội suy song tuyến tính
        rectified = cv2.warpPerspective(
            image,
            homography_matrix,
            (dst_w, dst_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        return rectified, homography_matrix
