"""
FAST-PLATE-OCR-VIETNAMESE: Advanced Plate Image Augmentations
Bộ công cụ làm giàu dữ liệu nâng cao cho huấn luyện mô hình OCR:
- Simulated Screw / Bolt Cutout Occlusion (Mô phỏng ốc vít khoan biển số)
- Directional Motion Blur (Nhòe chuyển động)
- Photometric Distortions & Sensor Noise (Biến dạng quang học & nhiễu cảm biến)
"""

import math
import random
import numpy as np
import cv2
from typing import Tuple, Optional


def apply_screw_cutout(
    image: np.ndarray,
    num_screws_range: Tuple[int, int] = (1, 3),
    radius_range: Tuple[int, int] = (4, 8),
    p: float = 0.6,
) -> np.ndarray:
    """
    Mô phỏng ốc vít bắt biển số (Screw / Bolt Cutout Occlusion):
    Tại Việt Nam, biển số thường bị thợ khoan 2 hoặc 4 lỗ vít trực tiếp xuyên qua mặt biển
    để gắn vào cản xe, gây che khuất một phần nét chữ.
    Hàm này tạo các vết tròn / lục giác giả lập ốc vít kim loại kèm bóng đổ và mũ vít tối màu.
    """
    if random.random() > p:
        return image

    img = image.copy()
    h, w = img.shape[:2]
    num_screws = random.randint(num_screws_range[0], num_screws_range[1])

    for _ in range(num_screws):
        radius = random.randint(radius_range[0], radius_range[1])
        # Vị trí xuất hiện thường nằm rải rác ở dải giữa hoặc 1/3 trên dưới
        cx = random.randint(int(w * 0.1), int(w * 0.9))
        cy = random.randint(int(h * 0.2), int(h * 0.8))

        # Màu ốc vít: bạc kim loại hoặc đen rỉ sét
        is_metallic = random.random() > 0.4
        if is_metallic:
            outer_color = (random.randint(160, 210), random.randint(160, 210), random.randint(160, 210))
            inner_color = (random.randint(40, 90), random.randint(40, 90), random.randint(40, 90))
        else:
            outer_color = (random.randint(30, 60), random.randint(30, 60), random.randint(30, 60))
            inner_color = (random.randint(15, 35), random.randint(15, 35), random.randint(15, 35))

        # 1. Vẽ vòng đệm ngoài (washer)
        cv2.circle(img, (cx, cy), radius, outer_color, -1, lineType=cv2.LINE_AA)
        
        # 2. Vẽ mũ ốc trong hoặc rãnh chữ thập / lục giác
        inner_r = max(2, radius - 2)
        cv2.circle(img, (cx, cy), inner_r, inner_color, -1, lineType=cv2.LINE_AA)

        # 3. Vẽ rãnh vít (slot / cross head) ngẫu nhiên
        if random.random() > 0.5:
            angle = random.uniform(0, math.pi)
            dx = int(inner_r * math.cos(angle))
            dy = int(inner_r * math.sin(angle))
            cv2.line(img, (cx - dx, cy - dy), (cx + dx, cy + dy), (20, 20, 20), 1, lineType=cv2.LINE_AA)

    return img


def apply_motion_blur(
    image: np.ndarray,
    kernel_size_range: Tuple[int, int] = (3, 7),
    angle_range: Tuple[int, int] = (-25, 25),
    p: float = 0.5,
) -> np.ndarray:
    """
    Mô phỏng hiệu ứng nhòe chuyển động (Directional Motion Blur)
    do phương tiện di chuyển hoặc rung lắc camera hành trình.
    """
    if random.random() > p:
        return image

    ksize = random.randrange(kernel_size_range[0], kernel_size_range[1] + 1, 2)
    if ksize <= 1:
        return image

    angle = random.uniform(angle_range[0], angle_range[1])
    
    # Tạo ma trận quay cho kernel nhòe chuyển động
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    center = (ksize - 1) / 2.0
    rad = math.radians(angle)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)

    for i in range(ksize):
        offset = i - center
        x = int(round(center + offset * cos_a))
        y = int(round(center + offset * sin_a))
        if 0 <= x < ksize and 0 <= y < ksize:
            kernel[y, x] = 1.0

    k_sum = kernel.sum()
    if k_sum > 0:
        kernel /= k_sum
    else:
        kernel[int(center), int(center)] = 1.0

    return cv2.filter2D(image, -1, kernel)


def apply_photometric_distortions(
    image: np.ndarray,
    brightness_range: Tuple[float, float] = (0.75, 1.30),
    contrast_range: Tuple[float, float] = (0.75, 1.35),
    noise_sigma: float = 8.0,
    p: float = 0.7,
) -> np.ndarray:
    """
    Áp dụng biến dạng ánh sáng (độ sáng, độ tương phản) và nhiễu cảm biến ISO cao.
    """
    if random.random() > p:
        return image

    img = image.astype(np.float32)

    # 1. Độ sáng & Độ tương phản
    alpha = random.uniform(contrast_range[0], contrast_range[1])
    beta = (random.uniform(brightness_range[0], brightness_range[1]) - 1.0) * 128.0
    img = img * alpha + beta

    # 2. Nhiễu Gaussian cảm biến
    if random.random() > 0.4 and noise_sigma > 0:
        noise = np.random.normal(0, noise_sigma, img.shape).astype(np.float32)
        img = img + noise

    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


class PlateAugmentor:
    """
    Pipeline tăng cường dữ liệu toàn diện cho huấn luyện OCR.
    """

    def __init__(
        self,
        screw_prob: float = 0.5,
        blur_prob: float = 0.4,
        photo_prob: float = 0.6,
    ):
        self.screw_prob = screw_prob
        self.blur_prob = blur_prob
        self.photo_prob = photo_prob

    def __call__(self, image: np.ndarray, is_train: bool = True) -> np.ndarray:
        if not is_train:
            return image

        img = image.copy()
        # Áp dụng ngẫu nhiên các tầng tăng cường
        img = apply_screw_cutout(img, p=self.screw_prob)
        img = apply_motion_blur(img, p=self.blur_prob)
        img = apply_photometric_distortions(img, p=self.photo_prob)
        return img
