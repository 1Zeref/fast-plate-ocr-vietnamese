"""
FAST-PLATE-OCR-VIETNAMESE: ByteTrack Multi-Object Tracker (Bước 5.1)
Thuật toán ByteTrack theo dõi phương tiện và biển số xe theo thời gian thực (25-30 FPS).
Liên kết hai pha phát hiện độ tin cậy cao (D_high) và độ tin cậy thấp (D_low)
giúp duy trì Tracklet ID liên tục ngay cả khi xe bị che khuất hoặc nhòe mờ.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from scipy.optimize import linear_sum_assignment

from src.tracking.kalman import KalmanFilter
from src.tracking.basetrack import STrack, TrackState, BaseTrack


def bbox_ious(atlbrs: np.ndarray, btlbrs: np.ndarray) -> np.ndarray:
    """Tính ma trận IoU giữa hai tập hợp bounding boxes [N, 4] và [M, 4]."""
    if len(atlbrs) == 0 or len(btlbrs) == 0:
        return np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)

    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    for i, a in enumerate(atlbrs):
        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        for j, b in enumerate(btlbrs):
            area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
            iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
            ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
            inter = iw * ih
            union = area_a + area_b - inter
            if union > 0:
                ious[i, j] = inter / union
    return ious


def linear_assignment(cost_matrix: np.ndarray, thresh: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ghép cặp tối ưu qua thuật toán Hungarian (scipy linear_sum_assignment)."""
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(cost_matrix.shape[0], dtype=int),
            np.arange(cost_matrix.shape[1], dtype=int),
        )

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = []
    unmatched_a = list(range(cost_matrix.shape[0]))
    unmatched_b = list(range(cost_matrix.shape[1]))

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= thresh:
            matches.append([r, c])
            if r in unmatched_a:
                unmatched_a.remove(r)
            if c in unmatched_b:
                unmatched_b.remove(c)

    return np.asarray(matches, dtype=int), np.asarray(unmatched_a, dtype=int), np.asarray(unmatched_b, dtype=int)


class ByteTracker:
    """
    Bộ theo dõi ByteTrack chuyên dụng cho biển số xe và phương tiện:
    - Nhận danh sách phát hiện thô từ Bước 2.1 trên mỗi frame video.
    - Duy trì Tracklet ID bất biến cho từng phương tiện.
    - Cập nhật chuyển động mượt mà với Bộ lọc Kalman.
    """

    def __init__(
        self,
        track_thresh: float = 0.40,
        match_thresh: float = 0.70,
        low_score_thresh: float = 0.10,
        max_time_lost: int = 30,
        frame_rate: int = 30,
    ):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.low_score_thresh = low_score_thresh
        self.max_time_lost = max_time_lost
        self.frame_rate = frame_rate

        self.kalman_filter = KalmanFilter()
        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        self.removed_stracks: List[STrack] = []
        self.frame_id = 0

    def reset(self):
        """Khởi động lại toàn bộ trạng thái tracking."""
        BaseTrack.reset_id()
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0

    def update(self, detections: List[Dict[str, Any]]) -> List[STrack]:
        """
        Cập nhật tracking trên frame kế tiếp:
        detections: Danh sách dict từ mô hình detector, mỗi phần tử chứa:
          - 'plate_bbox': [x1, y1, x2, y2]
          - 'confidence': float
          - 'corners': list 4 điểm góc
          - 'layout': '1_LINE' hoặc '2_LINE'
        """
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        # 1. Chuyển đổi detections thành các đối tượng STrack
        detections_high: List[STrack] = []
        detections_low: List[STrack] = []

        for det in detections:
            bbox = det["plate_bbox"]
            tlwh = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
            score = float(det.get("confidence", 1.0))
            corners = det.get("corners", [])
            layout = det.get("layout", "1_LINE")

            track = STrack(tlwh, score, corners=corners, layout=layout)
            if score >= self.track_thresh:
                detections_high.append(track)
            elif score >= self.low_score_thresh:
                detections_low.append(track)

        # 2. Dự đoán vị trí kế tiếp cho các tracklet đang hoạt động
        unconfirmed: List[STrack] = []
        tracked_stracks: List[STrack] = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        strack_pool = self._joint_stracks(tracked_stracks, self.lost_stracks)
        for strack in strack_pool:
            strack.predict()

        # -------------------------------------------------------------
        # BƯỚC 1: LIÊN KẾT PHA 1 VỚI D_HIGH (High-Score Detections)
        # -------------------------------------------------------------
        dists = self._iou_distance(strack_pool, detections_high)
        matches, u_track, u_detection = linear_assignment(dists, thresh=self.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections_high[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # -------------------------------------------------------------
        # BƯỚC 2: LIÊN KẾT PHA 2 VỚI D_LOW (Low-Score Detections)
        # -------------------------------------------------------------
        # Thu hồi các tracklet bị che khuất hoặc mờ nhòe bằng các detection điểm thấp
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = self._iou_distance(r_tracked_stracks, detections_low)
        matches, u_track_r, _ = linear_assignment(dists, thresh=0.50)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_low[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track_r:
            track = r_tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # -------------------------------------------------------------
        # BƯỚC 3: XỬ LÝ UNCONFIRMED TRACKS & KHỞI TẠO TRACKLET MỚI
        # -------------------------------------------------------------
        detections_rem = [detections_high[i] for i in u_detection]
        dists = self._iou_distance(unconfirmed, detections_rem)
        matches, u_unconfirmed, u_detection_rem = linear_assignment(dists, thresh=0.70)

        for itracked, idet in matches:
            unconfirmed[itracked].update(detections_rem[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])

        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        # Khởi tạo các tracklet mới từ detection high-score còn sót lại
        for inew in u_detection_rem:
            track = detections_rem[inew]
            if track.score >= self.track_thresh:
                track.activate(self.kalman_filter, self.frame_id)
                activated_starcks.append(track)

        # -------------------------------------------------------------
        # BƯỚC 4: QUẢN LÝ VÒNG ĐỜI TRACKLET (LOST / REMOVED)
        # -------------------------------------------------------------
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = self._joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = self._joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = self._sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = self._sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)

        # Xuất các tracklet đang hoạt động
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        return output_stracks

    @staticmethod
    def _joint_stracks(tlista: List[STrack], tlistb: List[STrack]) -> List[STrack]:
        exists = {}
        res = []
        for t in tlista:
            exists[t.track_id] = 1
            res.append(t)
        for t in tlistb:
            tid = t.track_id
            if not exists.get(tid, 0):
                exists[tid] = 1
                res.append(t)
        return res

    @staticmethod
    def _sub_stracks(tlista: List[STrack], tlistb: List[STrack]) -> List[STrack]:
        stracks = {t.track_id: t for t in tlista}
        for t in tlistb:
            tid = t.track_id
            if tid in stracks:
                del stracks[tid]
        return list(stracks.values())

    @staticmethod
    def _iou_distance(atracks: List[STrack], btracks: List[STrack]) -> np.ndarray:
        if len(atracks) == 0 or len(btracks) == 0:
            return np.zeros((len(atracks), len(btracks)), dtype=np.float32)
        atlbrs = np.array([track.tlbr for track in atracks], dtype=np.float32)
        btlbrs = np.array([track.tlbr for track in btracks], dtype=np.float32)
        ious = bbox_ious(atlbrs, btlbrs)
        return 1.0 - ious
