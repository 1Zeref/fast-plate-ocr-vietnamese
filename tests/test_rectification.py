"""
FAST-PLATE-OCR-VIETNAMESE: Unit Tests for Homography & Geometry Routing (Step 2.2)
Kiểm thử nắn phối cảnh, sắp xếp 4 góc, định tuyến 1-line/2-line và sinh Virtual 1-Line Strip.
"""

import os
import sys
import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rectification.homography import (
    order_quad_corners,
    calculate_quad_aspect_ratio,
    PerspectiveRectifier,
)
from src.rectification.router import GeometryRouter
from src.rectification.splitter import TwoLinePlateSplitter
from src.rectification.pipeline import PlateRectificationPipeline


def test_corner_ordering():
    """Kiểm tra thuật toán tự động sắp xếp 4 điểm góc theo thứ tự chuẩn: TL, TR, BR, BL."""
    # Định nghĩa 4 điểm góc chuẩn
    p_tl = [100.0, 100.0]
    p_tr = [500.0, 120.0]
    p_br = [480.0, 220.0]
    p_bl = [90.0, 200.0]

    # Xáo trộn thứ tự các điểm
    shuffled_corners = [p_br, p_tl, p_bl, p_tr]

    ordered = order_quad_corners(shuffled_corners)
    
    assert np.allclose(ordered[0], p_tl), f"Top-Left sai: {ordered[0]} vs {p_tl}"
    assert np.allclose(ordered[1], p_tr), f"Top-Right sai: {ordered[1]} vs {p_tr}"
    assert np.allclose(ordered[2], p_br), f"Bottom-Right sai: {ordered[2]} vs {p_br}"
    assert np.allclose(ordered[3], p_bl), f"Bottom-Left sai: {ordered[3]} vs {p_bl}"

    print("[PASS] test_corner_ordering (Sắp xếp 4 góc chuẩn xác)")


def test_geometry_router():
    """Kiểm tra bộ định tuyến phân luồng 1-line vs 2-line."""
    router = GeometryRouter(aspect_ratio_threshold=2.40)

    # 1. Biển dài (520x120 -> AR ~ 4.3)
    long_corners = [[100, 100], [520, 105], [515, 215], [105, 210]]
    res_long = router.route(long_corners)
    assert res_long["layout"] == "1-line", f"Kỳ vọng 1-line, thực tế: {res_long['layout']}"
    assert res_long["is_two_line"] is False

    # 2. Biển vuông (340x200 -> AR ~ 1.7)
    square_corners = [[100, 100], [300, 102], [298, 260], [102, 258]]
    res_square = router.route(square_corners)
    assert res_square["layout"] == "2-line", f"Kỳ vọng 2-line, thực tế: {res_square['layout']}"
    assert res_square["is_two_line"] is True

    # 3. Thử nghiệm Ensemble với Detector Hint
    res_ens = router.route(square_corners, detector_layout_hint="2_LINE", detector_confidence=0.92)
    assert res_ens["layout"] == "2-line"
    assert res_ens["method"] == "ENSEMBLE_AGREED"

    print("[PASS] test_geometry_router (Định tuyến hình học & Ensemble chính xác)")


def test_two_line_splitter_and_virtual_strip():
    """Kiểm tra cắt tách 2 dòng và sinh Virtual 1-Line Strip."""
    splitter = TwoLinePlateSplitter(strip_height=64, separator_width=16, target_ocr_size=(256, 64))

    # Tạo một ảnh biển vuông giả lập có 2 dòng chữ
    h, w = 200, 340
    dummy_square = np.full((h, w, 3), 250, dtype=np.uint8)
    
    # Vẽ chữ giả lập dòng 1 (y: 30..70)
    cv2.putText(dummy_square, "29-B1", (60, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20, 20, 20), 3)
    # Vẽ chữ giả lập dòng 2 (y: 120..170)
    cv2.putText(dummy_square, "123.45", (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20, 20, 20), 3)

    # 1. Tìm đường cắt
    split_y = splitter.find_split_boundary_y(dummy_square)
    assert 65 < split_y < 120, f"Đường cắt split_y={split_y} phải nằm giữa dòng 1 (y=65) và dòng 2 (y=120)"

    # 2. Sinh Virtual 1-Line Strip
    virtual_strip, meta = splitter.create_virtual_1line_strip(dummy_square, split_y=split_y)
    
    assert virtual_strip.shape == (64, 256, 3), f"Kích thước strip không chuẩn: {virtual_strip.shape}"
    assert "split_y" in meta
    assert "virtual_strip_shape" in meta

    print(f"[PASS] test_two_line_splitter_and_virtual_strip (Split Y={split_y}, Strip shape={virtual_strip.shape})")


def test_full_rectification_pipeline():
    """Kiểm tra toàn bộ pipeline: Ảnh góc xiên -> Nắn phẳng -> Router -> OCR Strip."""
    pipeline = PlateRectificationPipeline(target_ocr_size=(256, 64))

    # Tạo ảnh cảnh giả lập 1280x720 chứa biển số góc xiên 35 độ
    scene = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # 4 điểm góc xiên của biển vuông
    skewed_square_corners = [
        [500, 310], # TL
        [740, 345], # TR
        [720, 480], # BR
        [485, 435], # BL
    ]

    result = pipeline.process(scene, skewed_square_corners)

    assert result["is_two_line"] is True
    assert result["layout"] == "2-line"
    assert result["rectified_raw"].shape == (240, 320, 3), "Kích thước nắn phẳng biển vuông phải là (240, 320, 3)"
    assert result["ocr_input_strip"].shape == (64, 256, 3), "Kích thước dải ảnh OCR phải là (64, 256, 3)"
    assert result["homography_matrix"].shape == (3, 3), "Ma trận phối cảnh phải là 3x3"

    print("[PASS] test_full_rectification_pipeline (Pipeline hoàn chỉnh vượt qua toàn bộ kiểm thử)")


if __name__ == "__main__":
    print("=" * 60)
    print("CHẠY BỘ KIỂM THỬ NẮN PHỐI CẢNH & ĐỊNH TUYẾN HÌNH HỌC (BƯỚC 2.2)")
    print("=" * 60)
    test_corner_ordering()
    test_geometry_router()
    test_two_line_splitter_and_virtual_strip()
    test_full_rectification_pipeline()
    print("=" * 60)
    print("TẤT CẢ CÁC BÀI TEST BƯỚC 2.2 ĐÃ THÀNH CÔNG VƯỢT QUA!")
    print("=" * 60)
