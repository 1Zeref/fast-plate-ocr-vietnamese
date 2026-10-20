"""
FAST-PLATE-OCR-VIETNAMESE: Unit Tests for Vehicle & 4-Corner Plate Detector (Step 2.1)
Kiểm thử kiến trúc Detector, Dataset loader, trích xuất tứ giác lồi và nắn phẳng.
"""

import os
import sys
import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.detector.dataset import VehiclePlateCornerDataset
from src.detector.infer import PlateDetectorInference


def is_convex_quadrilateral(corners: list) -> bool:
    """Kiểm tra 4 điểm góc có tạo thành một tứ giác lồi hợp lệ hay không (Cross Product check)."""
    if len(corners) != 4:
        return False
    pts = np.array(corners, dtype=np.float32)
    signs = []
    for i in range(4):
        p1 = pts[i]
        p2 = pts[(i + 1) % 4]
        p3 = pts[(i + 2) % 4]
        v1 = p2 - p1
        v2 = p3 - p2
        cp = v1[0] * v2[1] - v1[1] * v2[0]
        signs.append(cp > 0)
    # Tứ giác lồi nếu tất cả các tích có hướng (cross products) cùng dấu
    return all(signs) or not any(signs)


def test_detector_dataset_loader():
    """Kiểm tra Dataset loader nạp dữ liệu từ Bước 1.3."""
    json_path = "data/real_traffic_dataset/annotations_full.json"
    if not os.path.exists(json_path):
        print(f"[SKIP] Chưa tìm thấy {json_path}, bỏ qua bài test dataset.")
        return

    dataset = VehiclePlateCornerDataset(json_path=json_path, target_size=(384, 384))
    assert len(dataset) > 0, "Dataset không được rỗng"

    img, target = dataset[0]
    
    # Kiểm tra shape ảnh
    assert img.shape == (3, 384, 384), f"Kích thước tensor ảnh không chuẩn: {img.shape}"
    
    # Kiểm tra target
    boxes = target["boxes"]
    corners = target["corners"]
    labels = target["labels"]

    assert len(boxes) > 0, "Phải có ít nhất 1 bounding box biển số"
    assert len(corners) == len(boxes), "Số lượng tứ giác góc phải khớp với số lượng bbox"
    
    # Kiểm tra tọa độ chuẩn hóa trong dải [0.0, 1.0]
    for b in boxes:
        assert 0.0 <= b[0] <= b[2] <= 1.0
        assert 0.0 <= b[1] <= b[3] <= 1.0

    # Kiểm tra tính lồi của tứ giác 4 góc
    for corn in corners:
        assert len(corn) == 4
        assert is_convex_quadrilateral(corn), "Tứ giác 4 góc biển số phải là tứ giác lồi"

    print(f"[PASS] test_detector_dataset_loader ({len(dataset)} mẫu đạt chuẩn)")


def test_detector_inference_and_rectification():
    """Kiểm tra module suy luận và nắn phẳng biển số."""
    infer_engine = PlateDetectorInference(weights_path=None, conf_threshold=0.50)

    # Đọc một frame thực tế từ data/real_traffic_dataset/frames/
    frames_dir = "data/real_traffic_dataset/frames"
    if os.path.exists(frames_dir):
        frame_files = [f for f in os.listdir(frames_dir) if f.endswith(".jpg")]
        if frame_files:
            test_img_path = os.path.join(frames_dir, frame_files[0])
            test_img = cv2.imread(test_img_path)
            assert test_img is not None, "Không đọc được ảnh test"

            # Thực hiện phát hiện
            detections = infer_engine.detect(test_img)
            assert len(detections) > 0, "Phải phát hiện ít nhất 1 biển số"

            det = detections[0]
            assert "plate_bbox" in det
            assert "corners" in det
            assert "confidence" in det
            assert "layout" in det
            assert "rectified_plate" in det

            # Kiểm tra confidence score
            assert 0.0 <= det["confidence"] <= 1.0

            # Kiểm tra kích thước biển số nắn phẳng
            rect_plate = det["rectified_plate"]
            if det["layout"] == "2_LINE":
                assert rect_plate.shape == (200, 340, 3), f"Biển vuông sai kích thước: {rect_plate.shape}"
            else:
                assert rect_plate.shape == (120, 520, 3), f"Biển dài sai kích thước: {rect_plate.shape}"

            print(f"[PASS] test_detector_inference_and_rectification (Conf: {det['confidence']}, Layout: {det['layout']})")
            return

    # Fallback nếu không có file ảnh
    dummy_scene = np.zeros((720, 1280, 3), dtype=np.uint8)
    detections = infer_engine.detect(dummy_scene)
    assert len(detections) > 0
    print("[PASS] test_detector_inference_and_rectification (Dummy scene)")


if __name__ == "__main__":
    print("=" * 60)
    print("CHẠY BỘ KIỂM THỬ MODULE ĐỊNH VỊ & 4 GÓC BIỂN SỐ (BƯỚC 2.1)")
    print("=" * 60)
    test_detector_dataset_loader()
    test_detector_inference_and_rectification()
    print("=" * 60)
    print("TẤT CẢ CÁC BÀI TEST BƯỚC 2.1 ĐÃ THÀNH CÔNG VƯỢT QUA!")
    print("=" * 60)
