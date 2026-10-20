"""
FAST-PLATE-OCR-VIETNAMESE: Video Frame Sampler & Quality Filter
Trích xuất khung hình thông minh từ video camera giám sát giao thông và bãi đỗ xe thực địa.
Lọc bỏ khung hình tĩnh, loại bỏ chuyển động nhòe mờ và phân loại kịch bản môi trường.
"""

import os
import sys
import cv2
import numpy as np
from typing import Dict, Any, List, Generator, Tuple, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


class VideoFrameSampler:
    """
    Bộ trích xuất khung hình tự động từ video thực địa:
    - Lọc bỏ frame trùng lặp (khi xe dừng đèn đỏ / đứng yên).
    - Lọc bỏ frame nhòe mờ không đạt tiêu chuẩn OCR (Laplacian Variance Filter).
    - Tự động nhận diện kịch bản môi trường chiếu sáng (Nắng gắt, Đêm ngược sáng, Mưa, Góc xiên).
    """

    def __init__(
        self,
        min_motion_diff: float = 12.0,
        blur_threshold: float = 85.0,
        sample_interval_ms: int = 250, # Lấy mẫu tối đa 4 frames/giây
    ):
        self.min_motion_diff = min_motion_diff
        self.blur_threshold = blur_threshold
        self.sample_interval_ms = sample_interval_ms

    @staticmethod
    def calculate_blur_score(image: np.ndarray) -> float:
        """Đo độ sắc nét của ảnh dựa trên phương sai của toán tử vi phân Laplacian."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def classify_lighting_scenario(image: np.ndarray) -> str:
        """
        Phân tích biểu đồ histogram để ước lượng kịch bản môi trường:
        - NIGHT_BACKLIGHT: Ban đêm, ánh sáng yếu hoặc bị chói đèn pha.
        - SUNNY_DAYLIGHT: Nắng gắt, tương phản cao, nhiều pixel bão hòa trắng.
        - RAINY_WET: Ánh sáng tán xạ, phản quang mặt đường ướt.
        - STANDARD: Điều kiện ánh sáng tiêu chuẩn.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        mean_lum = float(np.mean(gray))
        std_lum = float(np.std(gray))
        high_pixel_ratio = float(np.sum(gray > 240)) / gray.size

        if mean_lum < 55.0:
            return "NIGHT_BACKLIGHT"
        elif mean_lum > 175.0 or high_pixel_ratio > 0.15:
            return "SUNNY_DAYLIGHT"
        elif std_lum < 35.0 and 60.0 < mean_lum < 150.0:
            return "RAINY_WET"
        else:
            return "STANDARD_DAYLIGHT"

    def process_video(
        self,
        video_path: str,
        output_dir: str,
        scenario_tag: Optional[str] = None,
        max_frames: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Đọc và trích xuất các khung hình chất lượng cao từ một tệp video.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Không tìm thấy file video: {video_path}")

        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Không thể mở luồng video từ: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_step = max(1, int(fps * (self.sample_interval_ms / 1000.0)))
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        extracted_metadata: List[Dict[str, Any]] = []
        prev_gray: Optional[np.ndarray] = None
        frame_idx = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_step != 0:
                frame_idx += 1
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 1. Kiểm tra chuyển động (Frame Differencing) để tránh frame trùng lặp
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                mean_diff = float(np.mean(diff))
                if mean_diff < self.min_motion_diff:
                    # Chuyển động quá ít (xe đứng yên hoặc camera đóng băng)
                    frame_idx += 1
                    continue

            # 2. Kiểm tra độ sắc nét (Laplacian Blur Check)
            blur_score = self.calculate_blur_score(frame)
            if blur_score < self.blur_threshold:
                # Frame bị nhòe mờ nghiêm trọng do rung lắc hoặc tốc độ cao
                frame_idx += 1
                continue

            # 3. Phân loại kịch bản môi trường
            detected_scenario = self.classify_lighting_scenario(frame)
            final_scenario = scenario_tag if scenario_tag else detected_scenario

            # 4. Lưu ảnh frame đạt chuẩn
            saved_count += 1
            filename = f"{video_name}_f{frame_idx:06d}_{final_scenario}.jpg"
            save_path = os.path.join(output_dir, filename)
            cv2.imwrite(save_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            meta = {
                "frame_id": f"{video_name}_{frame_idx:06d}",
                "filename": filename,
                "relative_path": os.path.join("frames", filename),
                "absolute_path": os.path.abspath(save_path),
                "frame_index": frame_idx,
                "timestamp_sec": round(frame_idx / fps, 3),
                "resolution": [int(frame.shape[1]), int(frame.shape[0])],
                "blur_score": round(blur_score, 2),
                "scenario": final_scenario,
            }
            extracted_metadata.append(meta)

            prev_gray = gray
            frame_idx += 1

            if max_frames and saved_count >= max_frames:
                break

        cap.release()
        return extracted_metadata
