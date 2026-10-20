"""
FAST-PLATE-OCR-VIETNAMESE: Realistic Augmentor & Perspective Warper
Mô phỏng góc chụp camera thực tế (Homography), điều kiện chiếu sáng và suy giảm cảm biến.
"""

import random
import cv2
import numpy as np


class RealisticAugmentor:
    """Bộ biến dạng hình học 3D và môi trường quang học thực tế."""

    @staticmethod
    def apply_perspective_transform(
        img: np.ndarray,
        max_tilt_ratio: float = 0.14,
    ) -> np.ndarray:
        """
        Mô phỏng góc chụp nghiêng của camera giám sát giao thông / camera hành trình
        bằng phép biến đổi xạ ảnh (Homography / Perspective Transform).
        """
        h, w = img.shape[:2]

        # 4 điểm góc ban đầu
        src_pts = np.float32([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1],
        ])

        # Biên độ xê dịch góc ngẫu nhiên
        max_dx = w * max_tilt_ratio
        max_dy = h * max_tilt_ratio

        dst_pts = np.float32([
            [random.uniform(0, max_dx), random.uniform(0, max_dy)],
            [w - 1 - random.uniform(0, max_dx), random.uniform(0, max_dy)],
            [w - 1 - random.uniform(0, max_dx), h - 1 - random.uniform(0, max_dy)],
            [random.uniform(0, max_dx), h - 1 - random.uniform(0, max_dy)],
        ])

        # Tính ma trận biến đổi và áp dụng
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(
            img,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return warped

    @staticmethod
    def apply_sun_glare(img: np.ndarray) -> np.ndarray:
        """Mô phỏng vệt chói lóa ánh nắng hoặc đèn pha xe phản xạ trên màng phản quang."""
        h, w = img.shape[:2]
        glare_mask = np.zeros((h, w), dtype=np.float32)

        center_x = random.randint(0, w)
        center_y = random.randint(0, h)
        radius = random.randint(int(min(h, w) * 0.3), int(max(h, w) * 0.8))

        cv2.circle(glare_mask, (center_x, center_y), radius, 1.0, -1)
        glare_mask = cv2.GaussianBlur(glare_mask, (51, 51), 0)

        glare_strength = random.uniform(0.3, 0.7)
        result = img.astype(np.float32) + glare_mask[:, :, np.newaxis] * 255.0 * glare_strength
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def apply_shadow(img: np.ndarray) -> np.ndarray:
        """Mô phỏng bóng đổ che khuất một phần biển số (bóng cây, gầm xe)."""
        h, w = img.shape[:2]
        shadow_mask = np.ones((h, w), dtype=np.float32)

        # Tạo đường bóng xiên
        x1, y1 = random.randint(0, w), 0
        x2, y2 = random.randint(0, w), h
        poly = np.array([[0, 0], [x1, y1], [x2, y2], [0, h]], dtype=np.int32)
        cv2.fillPoly(shadow_mask, [poly], random.uniform(0.4, 0.7))

        shadow_mask = cv2.GaussianBlur(shadow_mask, (21, 21), 0)
        result = img.astype(np.float32) * shadow_mask[:, :, np.newaxis]
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def apply_motion_blur(img: np.ndarray, max_kernel_size: int = 9) -> np.ndarray:
        """Mô phỏng nhòe chuyển động (xe chạy qua camera tốc độ cao)."""
        size = random.choice([3, 5, 7, max_kernel_size])
        kernel = np.zeros((size, size))
        
        # Góc nhòe (ngang hoặc hơi chéo theo hướng xe)
        angle = random.choice([0, 180, 45, 135])
        if angle in (0, 180):
            kernel[int((size - 1) / 2), :] = np.ones(size)
        else:
            np.fill_diagonal(kernel, 1)

        kernel /= size
        return cv2.filter2D(img, -1, kernel)

    @staticmethod
    def apply_sensor_noise(img: np.ndarray, noise_sigma: float = 12.0) -> np.ndarray:
        """Thêm nhiễu hạt cảm biến ISO (Gaussian + Salt & Pepper)."""
        h, w, c = img.shape
        gauss = np.random.normal(0, noise_sigma, (h, w, c)).astype(np.float32)
        noisy = img.astype(np.float32) + gauss
        return np.clip(noisy, 0, 255).astype(np.uint8)

    @staticmethod
    def apply_downscale_upscale(img: np.ndarray, scale_factor: float = 0.5) -> np.ndarray:
        """Mô phỏng xe ở khoảng cách xa (độ phân giải thấp, pixelation nhẹ)."""
        h, w = img.shape[:2]
        small_h = max(16, int(h * scale_factor))
        small_w = max(32, int(w * scale_factor))
        small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def apply_jpeg_compression(img: np.ndarray, quality: int = 40) -> np.ndarray:
        """Mô phỏng hiện tượng vỡ khối nén ảnh JPEG từ camera hành trình/RTSP stream."""
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encimg = cv2.imencode(".jpg", img, encode_param)
        return cv2.imdecode(encimg, 1)


def augment_plate_image(img: np.ndarray, is_square: bool) -> np.ndarray:
    """Áp dụng toàn bộ chuỗi suy giảm vật lý ngẫu nhiên lên ảnh biển số."""
    res = img.copy()

    # 1. Biến đổi góc nhìn 3D (Xạ ảnh / Homography) - 80% xác suất
    if random.random() < 0.80:
        res = RealisticAugmentor.apply_perspective_transform(
            res, max_tilt_ratio=random.uniform(0.06, 0.15)
        )

    # 2. Chiếu sáng: Bóng đổ hoặc Chói lóa
    light_rand = random.random()
    if light_rand < 0.35:
        res = RealisticAugmentor.apply_sun_glare(res)
    elif light_rand < 0.70:
        res = RealisticAugmentor.apply_shadow(res)

    # 3. Môi trường ánh sáng toàn cục (sáng tối / tương phản)
    alpha = random.uniform(0.75, 1.25)  # Contrast
    beta = random.randint(-30, 30)     # Brightness
    res = np.clip(res.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    # 4. Nhòe chuyển động hoặc làm mờ nhẹ (40% xác suất)
    if random.random() < 0.40:
        res = RealisticAugmentor.apply_motion_blur(res, max_kernel_size=random.choice([5, 7]))

    # 5. Độ phân giải thấp (xe ở xa - 30% xác suất)
    if random.random() < 0.30:
        res = RealisticAugmentor.apply_downscale_upscale(res, scale_factor=random.uniform(0.4, 0.7))

    # 6. Nhiễu hạt cảm biến ISO (60% xác suất)
    if random.random() < 0.60:
        res = RealisticAugmentor.apply_sensor_noise(res, noise_sigma=random.uniform(4.0, 16.0))

    # 7. Nén JPEG suy hao (70% xác suất)
    if random.random() < 0.70:
        res = RealisticAugmentor.apply_jpeg_compression(res, quality=random.randint(25, 80))

    return res
