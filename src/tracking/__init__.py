"""
FAST-PLATE-OCR-VIETNAMESE: Realtime Tracking & Quality Filtering Module (Giai đoạn 5)
Hệ thống theo dõi ByteTrack, đánh giá chất lượng quang học, hàng đợi Best-K frames,
và động cơ hợp nhất quyết định đa khung hình Temporal Consensus Engine.
"""

from src.tracking.kalman import KalmanFilter
from src.tracking.basetrack import BaseTrack, STrack, TrackState
from src.tracking.byte_tracker import ByteTracker
from src.tracking.quality import FrameQualityAssessor
from src.tracking.buffer import BestKFrameBuffer
from src.tracking.consensus import TemporalConsensusEngine

__all__ = [
    "KalmanFilter",
    "BaseTrack",
    "STrack",
    "TrackState",
    "ByteTracker",
    "FrameQualityAssessor",
    "BestKFrameBuffer",
    "TemporalConsensusEngine",
]
