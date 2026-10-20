"""
FAST-PLATE-OCR-VIETNAMESE: Physical Texture & Weathering Simulation
Mô phỏng bề mặt vật lý thực địa: kim loại xước, bùn đất bám, trầy xước cơ học, ố rỉ.
"""

import os
import random
import cv2
import numpy as np


class PhysicalTextureGenerator:
    """Tạo và phủ các lớp texture vật lý lên bề mặt biển số xe."""

    @staticmethod
    def apply_brushed_metal(img: np.ndarray, intensity: float = 0.25) -> np.ndarray:
        """Mô phỏng sọc xước phay kim loại của phôi nhôm dập biển."""
        h, w = img.shape[:2]
        # Sinh nhiễu 1 chiều theo phương ngang để tạo sọc phay
        noise_1d = np.random.normal(0, 15, (h, 1)).astype(np.float32)
        noise_2d = np.repeat(noise_1d, w, axis=1)
        
        # Thêm nhiễu vi mô 2 chiều
        micro_noise = np.random.normal(0, 5, (h, w)).astype(np.float32)
        total_noise = noise_2d + micro_noise

        result = img.astype(np.float32)
        for c in range(3):
            result[:, :, c] += total_noise * intensity
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def apply_scratches(img: np.ndarray, num_scratches: int = 5) -> np.ndarray:
        """Vẽ các vết cào xước cơ học ngẫu nhiên trên mặt biển."""
        h, w = img.shape[:2]
        scratch_layer = np.zeros((h, w), dtype=np.uint8)
        
        for _ in range(num_scratches):
            x1, y1 = random.randint(0, w - 1), random.randint(0, h - 1)
            length = random.randint(20, int(w * 0.4))
            angle = random.uniform(0, 2 * np.pi)
            x2 = int(x1 + length * np.cos(angle))
            y2 = int(y1 + length * np.sin(angle))
            thickness = random.choice([1, 2])
            cv2.line(scratch_layer, (x1, y1), (x2, y2), 255, thickness=thickness, lineType=cv2.LINE_AA)

        # Xước làm sáng hoặc tối màu bề mặt
        mask = scratch_layer > 0
        result = img.copy()
        # Vết xước sáng bạc (lộ nhôm)
        result[mask] = np.clip(result[mask].astype(np.int32) + random.randint(30, 70), 0, 255).astype(np.uint8)
        return result

    @staticmethod
    def apply_mud_dirt(img: np.ndarray, coverage: float = 0.15) -> np.ndarray:
        """Mô phỏng các đốm bùn đất sẫm màu bắn lên từ mặt đường."""
        h, w = img.shape[:2]
        dirt_mask = np.zeros((h, w), dtype=np.float32)

        # Tạo nhiều đốm bùn ngẫu nhiên
        num_splatters = random.randint(5, 20)
        for _ in range(num_splatters):
            cx = random.randint(0, w)
            cy = random.randint(int(h * 0.3), h) # Tập trung nhiều ở mép dưới
            rad_x = random.randint(10, 45)
            rad_y = random.randint(5, 25)
            cv2.ellipse(dirt_mask, (cx, cy), (rad_x, rad_y), random.randint(0, 180), 0, 360, 1.0, -1)

        # Làm nhòe viền đốm bùn tự nhiên
        dirt_mask = cv2.GaussianBlur(dirt_mask, (25, 25), 0)
        dirt_mask = np.clip(dirt_mask * coverage, 0, 0.75)

        # Màu bùn đất (nâu xám: BGR)
        mud_color = np.array([random.randint(30, 60), random.randint(45, 80), random.randint(60, 110)], dtype=np.float32)

        result = img.astype(np.float32)
        for c in range(3):
            result[:, :, c] = result[:, :, c] * (1.0 - dirt_mask) + mud_color[c] * dirt_mask
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def apply_vignette_and_stains(img: np.ndarray) -> np.ndarray:
        """Mô phỏng bụi bặm và ố mờ quang viền biển."""
        h, w = img.shape[:2]
        kernel_x = cv2.getGaussianKernel(w, w * 0.45)
        kernel_y = cv2.getGaussianKernel(h, h * 0.45)
        kernel = kernel_y * kernel_x.T
        mask = kernel / kernel.max()
        
        # Mép viền tối hơn một chút
        factor = 0.85 + 0.15 * mask
        result = img.astype(np.float32)
        for c in range(3):
            result[:, :, c] *= factor
        return np.clip(result, 0, 255).astype(np.uint8)


def apply_random_weathering(img: np.ndarray) -> np.ndarray:
    """Áp dụng kết hợp ngẫu nhiên các loại texture vật lý."""
    res = img.copy()

    # 1. Phay kim loại nhôm (xác suất 70%)
    if random.random() < 0.70:
        res = PhysicalTextureGenerator.apply_brushed_metal(res, intensity=random.uniform(0.15, 0.35))

    # 2. Xước cơ học (xác suất 50%)
    if random.random() < 0.50:
        res = PhysicalTextureGenerator.apply_scratches(res, num_scratches=random.randint(2, 8))

    # 3. Bùn đất bẩn (xác suất 40%)
    if random.random() < 0.40:
        res = PhysicalTextureGenerator.apply_mud_dirt(res, coverage=random.uniform(0.1, 0.4))

    # 4. Ố mờ mép viền (xác suất 60%)
    if random.random() < 0.60:
        res = PhysicalTextureGenerator.apply_vignette_and_stains(res)

    return res
