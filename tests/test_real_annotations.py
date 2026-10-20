"""
FAST-PLATE-OCR-VIETNAMESE: Unit Tests for Real-World Data Pipeline (Step 1.3)
Kiểm thử tính đúng đắn của Sampler, Nắn phẳng 4 điểm góc (Rectifier) và Định dạng nhãn chuẩn.
"""

import os
import sys
import json
import csv
import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.video_sampler import VideoFrameSampler
from src.data.annotator import (
    LicensePlateAnnotation,
    RealDatasetManager,
    rectify_plate_quad,
)


def test_video_sampler_metrics():
    """Kiểm tra các hàm đo lường chất lượng ảnh của VideoFrameSampler."""
    sampler = VideoFrameSampler()

    # 1. Ảnh sắc nét vs Ảnh bị làm nhòe mờ
    sharp_img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    blurred_img = cv2.GaussianBlur(sharp_img, (21, 21), 0)

    score_sharp = sampler.calculate_blur_score(sharp_img)
    score_blurred = sampler.calculate_blur_score(blurred_img)
    assert score_sharp > score_blurred, "Ảnh sắc nét phải có điểm blur score cao hơn ảnh nhòe"

    # 2. Phân loại kịch bản ánh sáng
    night_img = np.full((100, 100, 3), 30, dtype=np.uint8)
    sunny_img = np.full((100, 100, 3), 220, dtype=np.uint8)
    assert sampler.classify_lighting_scenario(night_img) == "NIGHT_BACKLIGHT"
    assert sampler.classify_lighting_scenario(sunny_img) == "SUNNY_DAYLIGHT"

    print("[PASS] test_video_sampler_metrics")


def test_perspective_rectification():
    """Kiểm tra thuật toán nắn phẳng 4 điểm góc (Perspective Rectification)."""
    # Tạo ảnh giả lập có kích thước 1280x720
    test_scene = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # 4 điểm góc nghiêng xiên 35 độ
    corners_skewed = [
        [400, 320], # Top-Left
        [720, 350], # Top-Right
        [690, 480], # Bottom-Right
        [380, 440], # Bottom-Left
    ]

    # Nắn phẳng về biển dài
    rectified_long = rectify_plate_quad(test_scene, corners_skewed, layout="1_LINE")
    assert rectified_long.shape == (120, 520, 3), f"Kích thước biển dài nắn phẳng sai: {rectified_long.shape}"

    # Nắn phẳng về biển vuông
    rectified_square = rectify_plate_quad(test_scene, corners_skewed, layout="2_LINE")
    assert rectified_square.shape == (200, 340, 3), f"Kích thước biển vuông nắn phẳng sai: {rectified_square.shape}"

    print("[PASS] test_perspective_rectification (Warp 4 điểm góc chính xác)")


def test_annotation_schema_export():
    """Kiểm tra cấu trúc và tính đầy đủ của 5 trường nhãn chuẩn theo Bước 1.3."""
    temp_dir = "data/test_annotation_temp"
    manager = RealDatasetManager(output_dir=temp_dir)

    # Tạo mẫu nhãn
    ann = LicensePlateAnnotation(
        vehicle_type="car",
        vehicle_bbox=[300, 200, 850, 600],
        plate_bbox=[480, 450, 670, 520],
        corners=[[480, 455], [670, 450], [665, 520], [482, 515]],
        text="30A-123.45",
        layout="1_LINE",
        scenario="OBLIQUE_ANGLE_30_45",
    )

    manager.add_frame_annotation(
        frame_id="frame_000001",
        image_path=os.path.join(temp_dir, "frames", "frame_000001.jpg"),
        image_size=(1280, 720),
        scenario="OBLIQUE_ANGLE_30_45",
        annotations=[ann],
    )

    json_path, csv_path, obb_dir = manager.export_annotations()

    # 1. Kiểm tra JSON
    assert os.path.exists(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        obj = data["frames"][0]["objects"][0]
        assert "vehicle_bbox" in obj
        assert "plate_bbox" in obj
        assert "corners" in obj
        assert len(obj["corners"]) == 4
        assert obj["text"] == "30A12345"
        assert obj["layout"] == "1_LINE"
        assert obj["grammar_valid"] is True

    # 2. Kiểm tra CSV
    assert os.path.exists(csv_path)
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["label"] == "30A12345"
        assert rows[0]["layout"] == "1_LINE"
        assert rows[0]["corners_4pt"] == "480,455;670,450;665,520;482,515"

    # 3. Kiểm tra YOLO-OBB txt file
    obb_file = os.path.join(obb_dir, "frame_000001.txt")
    assert os.path.exists(obb_file)
    with open(obb_file, "r", encoding="utf-8") as f:
        content = f.read().strip()
        parts = content.split()
        assert len(parts) == 9 # class_id + 8 tọa độ (x1 y1 x2 y2 x3 y3 x4 y4)
        assert parts[0] == "0" # 1_LINE tương ứng class 0

    print("[PASS] test_annotation_schema_export (Đầy đủ JSON, CSV, YOLO-OBB)")

    # Dọn dẹp thư mục test
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    print("=" * 60)
    print("CHẠY BỘ KIỂM THỬ THỰC ĐỊA & GÁN NHÃN CHUẨN (BƯỚC 1.3)")
    print("=" * 60)
    test_video_sampler_metrics()
    test_perspective_rectification()
    test_annotation_schema_export()
    print("=" * 60)
    print("TẤT CẢ CÁC BÀI TEST BƯỚC 1.3 ĐÃ THÀNH CÔNG VƯỢT QUA!")
    print("=" * 60)
