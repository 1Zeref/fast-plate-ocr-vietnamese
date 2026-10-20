"""
FAST-PLATE-OCR-VIETNAMESE: Best-K Frame Buffer & Quality Priority Queue (Bước 5.1)
Hàng đợi lưu trữ và chọn lọc K khung hình chất lượng cao nhất (K=3-5)
cho từng Tracklet ID để phục vụ tích lũy đa khung hình và chuyển giao cho module OCR.
"""

import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class BestKFrameBuffer:
    """
    Hàng đợi ưu tiên lưu giữ Top-K khung hình chất lượng cao nhất cho từng xe (Tracklet ID):
    - Tự động duy trì tối đa K khung hình có điểm chất lượng Q cao nhất.
    - Loại bỏ các khung hình mờ nhòe khi xuất hiện khung hình rõ nét hơn.
    - Sẵn sàng cung cấp dữ liệu cho module tích lũy và bỏ phiếu đa khung hình (Multi-frame Accumulation).
    """

    def __init__(
        self,
        k: int = 5,
        min_quality_threshold: float = 0.35,
        max_tracklets: int = 200,
    ):
        self.k = k
        self.min_quality_threshold = min_quality_threshold
        self.max_tracklets = max_tracklets

        # Cấu trúc lưu trữ: {track_id: list of frame_records sorted by quality descending}
        self.buffers: Dict[int, List[Dict[str, Any]]] = {}
        # Lịch sử quan sát: {track_id: {'total_seen': int, 'first_frame': int, 'last_frame': int}}
        self.history_stats: Dict[int, Dict[str, Any]] = {}

    def add_frame(
        self,
        track_id: int,
        frame_idx: int,
        quality_score: float,
        crop_image: np.ndarray,
        corners: Optional[List[List[float]]] = None,
        bbox: Optional[List[int]] = None,
        virtual_strip: Optional[np.ndarray] = None,
        quality_meta: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Tiếp nhận 1 frame mới thuộc về track_id:
        Trả về True nếu frame được đưa vào danh sách Top-K tốt nhất.
        """
        # Cập nhật lịch sử
        if track_id not in self.history_stats:
            self.history_stats[track_id] = {
                "total_seen": 1,
                "first_frame": frame_idx,
                "last_frame": frame_idx,
            }
        else:
            self.history_stats[track_id]["total_seen"] += 1
            self.history_stats[track_id]["last_frame"] = frame_idx

        # Khởi tạo buffer nếu chưa có
        if track_id not in self.buffers:
            self.buffers[track_id] = []

        record = {
            "track_id": track_id,
            "frame_idx": frame_idx,
            "timestamp": time.time(),
            "quality_score": round(float(quality_score), 4),
            "crop_image": crop_image,
            "corners": corners or [],
            "bbox": bbox or [],
            "virtual_strip": virtual_strip,
            "quality_meta": quality_meta or {},
        }

        buf = self.buffers[track_id]

        # Kiểm tra xem có đưa được vào Top-K không
        if len(buf) < self.k:
            buf.append(record)
            buf.sort(key=lambda x: x["quality_score"], reverse=True)
            return True
        else:
            # So sánh với phần tử có chất lượng thấp nhất hiện tại trong Top-K
            min_current_q = buf[-1]["quality_score"]
            if quality_score > min_current_q:
                buf[-1] = record
                buf.sort(key=lambda x: x["quality_score"], reverse=True)
                return True

        return False

    def get_best_frames(self, track_id: int) -> List[Dict[str, Any]]:
        """Lấy danh sách tối đa K khung hình tốt nhất của track_id hiện tại."""
        return self.buffers.get(track_id, [])

    def is_ready_for_recognition(
        self,
        track_id: int,
        min_frames: int = 3,
        min_avg_quality: float = 0.40,
    ) -> bool:
        """
        Kiểm tra xem tracklet đã tích lũy đủ số lượng và chất lượng khung hình để chạy OCR chưa.
        """
        frames = self.get_best_frames(track_id)
        if len(frames) < min_frames:
            return False

        avg_q = sum(f["quality_score"] for f in frames) / len(frames)
        return avg_q >= min_avg_quality

    def get_track_summary(self, track_id: int) -> Dict[str, Any]:
        """Tổng hợp thông số chất lượng của tracklet."""
        frames = self.get_best_frames(track_id)
        stats = self.history_stats.get(track_id, {})

        if not frames:
            return {
                "track_id": track_id,
                "buffered_frames": 0,
                "total_seen": stats.get("total_seen", 0),
                "best_quality": 0.0,
                "mean_quality": 0.0,
            }

        scores = [f["quality_score"] for f in frames]
        return {
            "track_id": track_id,
            "buffered_frames": len(frames),
            "total_seen": stats.get("total_seen", 0),
            "best_quality": max(scores),
            "mean_quality": round(float(np.mean(scores)), 4),
            "frame_indices": [f["frame_idx"] for f in frames],
        }

    def remove_track(self, track_id: int):
        """Xóa giải phóng bộ nhớ khi phương tiện đã rời khỏi vùng quan sát."""
        if track_id in self.buffers:
            del self.buffers[track_id]
        if track_id in self.history_stats:
            del self.history_stats[track_id]

    def cleanup_stale_tracks(self, active_track_ids: List[int], max_idle_frames: int = 60, current_frame: int = 0):
        """Thu hồi các tracklet không còn xuất hiện trong active list."""
        stale = []
        for tid, stats in self.history_stats.items():
            if tid not in active_track_ids:
                if current_frame - stats.get("last_frame", 0) > max_idle_frames:
                    stale.append(tid)

        for tid in stale:
            self.remove_track(tid)
