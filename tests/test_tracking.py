"""
FAST-PLATE-OCR-VIETNAMESE: Unit Test Suite for Step 5.1 (Realtime Tracking & Quality Filtering)
Kiểm thử bộ theo dõi ByteTrack, bộ lọc chất lượng quang học Laplacian & góc nghiêng,
và hàng đợi Best-K frames buffer.
"""

import os
import sys
import unittest
import numpy as np
import cv2

# Đảm bảo import được module từ thư mục gốc
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.tracking.kalman import KalmanFilter
from src.tracking.basetrack import STrack
from src.tracking.byte_tracker import ByteTracker
from src.tracking.quality import FrameQualityAssessor
from src.tracking.buffer import BestKFrameBuffer


class TestRealtimeTrackingStep51(unittest.TestCase):

    def test_kalman_filter(self):
        """Kiểm tra tính đúng đắn của Bộ lọc Kalman 8 chiều."""
        kf = KalmanFilter()
        measurement = np.array([100.0, 150.0, 2.5, 40.0], dtype=np.float32)
        mean, cov = kf.initiate(measurement)

        self.assertEqual(mean.shape, (8,))
        self.assertEqual(cov.shape, (8, 8))

        # Dự đoán bước kế tiếp
        next_mean, next_cov = kf.predict(mean, cov)
        self.assertFalse(np.isnan(next_mean).any())

        # Cập nhật quan sát mới
        new_obs = np.array([105.0, 152.0, 2.5, 40.0], dtype=np.float32)
        up_mean, up_cov = kf.update(next_mean, next_cov, new_obs)
        self.assertAlmostEqual(up_mean[0], 105.0, delta=3.0)
        print("[PASS] test_kalman_filter (Khởi tạo, Dự đoán & Cập nhật mượt mà)")

    def test_bytetracker_association_and_low_score_recovery(self):
        """
        Kiểm tra khả năng duy trì Tracklet ID và phục hồi đối tượng mờ nhòe bằng ByteTrack:
        - Frame 1: Phát hiện rõ (conf = 0.90) -> Khởi tạo Tracklet.
        - Frame 2: Xe di chuyển nhẹ (conf = 0.85) -> Cập nhật cùng ID.
        - Frame 3: Xe bị mờ nhòe/che khuất (conf = 0.25 - rơi vào D_low) -> ByteTrack Pha 2 cứu vớt, KHÔNG ĐỔI ID!
        - Frame 4: Xe rõ trở lại (conf = 0.88) -> Tiếp tục duy trì ID.
        """
        tracker = ByteTracker(track_thresh=0.40, low_score_thresh=0.15)

        # Frame 1
        dets_f1 = [{
            "plate_bbox": [100, 100, 180, 140],
            "confidence": 0.90,
            "corners": [[100, 100], [180, 100], [180, 140], [100, 140]],
        }]
        tracks_f1 = tracker.update(dets_f1)
        self.assertEqual(len(tracks_f1), 1)
        assigned_id = tracks_f1[0].track_id

        # Frame 2
        dets_f2 = [{
            "plate_bbox": [104, 102, 184, 142],
            "confidence": 0.85,
            "corners": [[104, 102], [184, 102], [184, 142], [104, 142]],
        }]
        tracks_f2 = tracker.update(dets_f2)
        self.assertEqual(len(tracks_f2), 1)
        self.assertEqual(tracks_f2[0].track_id, assigned_id)

        # Frame 3 (Detection điểm thấp: rơi vào D_low)
        dets_f3 = [{
            "plate_bbox": [108, 104, 188, 144],
            "confidence": 0.25,  # Dưới ngưỡng 0.40 của detector thông thường
            "corners": [[108, 104], [188, 104], [188, 144], [108, 144]],
        }]
        tracks_f3 = tracker.update(dets_f3)
        self.assertEqual(len(tracks_f3), 1)
        # ByteTrack liên kết pha 2 thành công và bảo toàn Tracklet ID
        self.assertEqual(tracks_f3[0].track_id, assigned_id)

        # Frame 4
        dets_f4 = [{
            "plate_bbox": [112, 106, 192, 146],
            "confidence": 0.88,
            "corners": [[112, 106], [192, 106], [192, 146], [112, 146]],
        }]
        tracks_f4 = tracker.update(dets_f4)
        self.assertEqual(len(tracks_f4), 1)
        self.assertEqual(tracks_f4[0].track_id, assigned_id)
        print(f"[PASS] test_bytetracker_association_and_low_score_recovery (Bảo toàn Tracklet ID={assigned_id} qua 4 frames)")

    def test_quality_assessor_sharp_vs_blurred(self):
        """Kiểm tra Bộ lọc chất lượng quang học phát hiện nhòe mờ và góc nghiêng."""
        assessor = FrameQualityAssessor(sharpness_threshold=85.0)

        # 1. Tạo ảnh nét (Text rõ ràng)
        sharp_img = np.full((64, 256, 3), 240, dtype=np.uint8)
        cv2.putText(sharp_img, "30A-123.45", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (10, 10, 10), 3)

        q_sharp = assessor.evaluate_quality(sharp_img)
        self.assertTrue(q_sharp["is_sharp"])
        self.assertGreater(q_sharp["laplacian_var"], 85.0)
        self.assertGreater(q_sharp["overall_quality"], 0.50)

        # 2. Tạo ảnh nhòe mờ do chuyển động cực mạnh (Motion blur)
        blurred_img = cv2.GaussianBlur(sharp_img, (25, 25), 10.0)
        q_blurred = assessor.evaluate_quality(blurred_img)
        self.assertFalse(q_blurred["is_sharp"])
        self.assertLess(q_blurred["laplacian_var"], 30.0)
        self.assertLess(q_blurred["overall_quality"], q_sharp["overall_quality"])

        # 3. Kiểm tra phạt góc nghiêng (Skew penalty)
        square_corners = [[10, 10], [100, 10], [100, 50], [10, 50]]
        skewed_corners = [[10, 10], [90, 45], [75, 75], [5, 40]] # Méo góc xiên ~30 độ
        score_sq = assessor.compute_geometric_skew_score(square_corners)
        score_skew = assessor.compute_geometric_skew_score(skewed_corners)
        self.assertGreater(score_sq, score_skew)

        print(f"[PASS] test_quality_assessor (Nét: Var={q_sharp['laplacian_var']} vs Mờ: Var={q_blurred['laplacian_var']})")

    def test_best_k_buffer(self):
        """Kiểm tra Hàng đợi lưu giữ Top-K khung hình tốt nhất."""
        buffer = BestKFrameBuffer(k=3, min_quality_threshold=0.35)
        track_id = 101

        # Nạp 6 frame có chất lượng biến động: [0.30, 0.85, 0.40, 0.95, 0.60, 0.90]
        scores = [0.30, 0.85, 0.40, 0.95, 0.60, 0.90]
        dummy_img = np.zeros((64, 256, 3), dtype=np.uint8)

        for idx, score in enumerate(scores):
            buffer.add_frame(
                track_id=track_id,
                frame_idx=idx + 1,
                quality_score=score,
                crop_image=dummy_img,
            )

        best_frames = buffer.get_best_frames(track_id)
        self.assertEqual(len(best_frames), 3)  # Giữ lại đúng K=3 frames
        saved_scores = [f["quality_score"] for f in best_frames]
        # Top 3 điểm cao nhất phải là 0.95, 0.90, 0.85 theo thứ tự giảm dần
        self.assertEqual(saved_scores, [0.95, 0.90, 0.85])

        # Kiểm tra điều kiện sẵn sàng chuyển giao cho OCR
        self.assertTrue(buffer.is_ready_for_recognition(track_id, min_frames=3, min_avg_quality=0.80))
        summary = buffer.get_track_summary(track_id)
        self.assertEqual(summary["total_seen"], 6)
        self.assertEqual(summary["buffered_frames"], 3)
        self.assertEqual(summary["best_quality"], 0.95)
        print(f"[PASS] test_best_k_buffer (Top-3 scores: {saved_scores}, Sẵn sàng cho OCR)")

    def test_multi_vehicle_tracking(self):
        """Kiểm tra theo dõi đồng thời 2 xe di chuyển song song."""
        tracker = ByteTracker(track_thresh=0.40)
        # Frame 1: 2 xe
        tracks_f1 = tracker.update([
            {"plate_bbox": [50, 100, 130, 140], "confidence": 0.90},
            {"plate_bbox": [300, 100, 380, 140], "confidence": 0.92},
        ])
        self.assertEqual(len(tracks_f1), 2)
        id1, id2 = tracks_f1[0].track_id, tracks_f1[1].track_id
        self.assertNotEqual(id1, id2)

        # Frame 2: Cả 2 xe cùng dịch chuyển
        tracks_f2 = tracker.update([
            {"plate_bbox": [55, 103, 135, 143], "confidence": 0.89},
            {"plate_bbox": [305, 102, 385, 142], "confidence": 0.91},
        ])
        self.assertEqual(len(tracks_f2), 2)
        f2_ids = {tracks_f2[0].track_id, tracks_f2[1].track_id}
        self.assertEqual(f2_ids, {id1, id2})
        print(f"[PASS] test_multi_vehicle_tracking (Phân tách chính xác 2 Tracklet ID: {id1} và {id2})")


if __name__ == "__main__":
    print("=" * 60)
    print("CHẠY BỘ KIỂM THỬ BYTETRACK, CHẤT LƯỢNG KHUNG HÌNH & BEST-K BUFFER (BƯỚC 5.1)")
    print("=" * 60)
    unittest.main()
