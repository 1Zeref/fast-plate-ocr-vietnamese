"""
FAST-PLATE-OCR-VIETNAMESE: Base Tracklet and STrack State
Cấu trúc dữ liệu Tracklet theo chuẩn ByteTrack.
"""

import numpy as np
from enum import Enum
from typing import List, Optional, Tuple

from src.tracking.kalman import KalmanFilter


class TrackState(Enum):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack:
    _count = 0

    track_id = 0
    is_activated = False
    state = TrackState.New

    history = []
    features = []
    curr_feature = None
    score = 0
    start_frame = 0
    frame_id = 0
    time_since_update = 0

    # Vị trí
    location = (np.inf, np.inf)

    @property
    def end_frame(self):
        return self.frame_id

    @staticmethod
    def next_id():
        BaseTrack._count += 1
        return BaseTrack._count

    @staticmethod
    def reset_id():
        BaseTrack._count = 0


class STrack(BaseTrack):
    """
    Tracklet đại diện cho một phương tiện/biển số xe đang được theo dõi:
    - Bounding Box [x1, y1, x2, y2]
    - Tọa độ 4 điểm góc biển số (Corners)
    - Chỉ số tin cậy (Confidence score)
    - Bộ lọc Kalman duy trì trạng thái vị trí và vận tốc
    """
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh: np.ndarray, score: float, corners: Optional[List[List[float]]] = None, layout: str = "1_LINE"):
        # tlwh: [top_left_x, top_left_y, width, height]
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = None
        self.mean = None
        self.covariance = None
        self.is_activated = False

        self.score = float(score)
        self.corners = corners or []
        self.layout = layout
        self.tracklet_len = 0

    def to_tlbr(self) -> np.ndarray:
        """Chuyển đổi [x, y, w, h] sang [x1, y1, x2, y2]."""
        ret = self._tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self) -> np.ndarray:
        """Chuyển đổi sang [center_x, center_y, aspect_ratio, height]."""
        ret = self._tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlbr_to_tlwh(tlbr: np.ndarray) -> np.ndarray:
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh: np.ndarray) -> np.ndarray:
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= max(1e-4, ret[3])
        return ret

    @staticmethod
    def xyah_to_tlwh(xyah: np.ndarray) -> np.ndarray:
        ret = np.asarray(xyah).copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.shared_kalman.predict(mean_state, self.covariance)

    def activate(self, kalman_filter: KalmanFilter, frame_id: int):
        """Kích hoạt một tracklet mới."""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.to_xyah())

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track: 'STrack', frame_id: int, new_id: bool = False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_track.to_xyah()
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        if new_track.corners:
            self.corners = new_track.corners
        self.layout = new_track.layout

    def update(self, new_track: 'STrack', frame_id: int):
        """Cập nhật tracklet với detection mới."""
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_xyah = new_track.to_xyah()
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_xyah
        )
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        if new_track.corners:
            self.corners = new_track.corners
        self.layout = new_track.layout

    @property
    def tlwh(self) -> np.ndarray:
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.xyah_to_tlwh(self.mean[:4])
        return ret

    @property
    def tlbr(self) -> np.ndarray:
        return self.to_tlbr()

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed
