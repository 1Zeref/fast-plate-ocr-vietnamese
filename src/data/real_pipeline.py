"""
FAST-PLATE-OCR-VIETNAMESE: Real-World Dataset Generation & Verification Pipeline
Tích hợp trích xuất frame, gán nhãn 5 trường dữ liệu chuẩn và kiểm thử nắn phẳng 4 góc.
"""

import os
import sys
import random
import cv2
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.data.annotator import LicensePlateAnnotation, RealDatasetManager, rectify_plate_quad
from src.synthetic.generator import generate_plate_specification
from src.synthetic.renderer import LicensePlateRenderer
from src.synthetic.textures import apply_random_weathering


def create_realistic_scene_frame(
    spec: Dict[str, Any],
    clean_plate: np.ndarray,
    scenario: str,
    canvas_size: Tuple[int, int] = (1280, 720), # Độ phân giải camera HD thực tế
) -> Tuple[np.ndarray, List[int], List[int], List[List[int]]]:
    """
    Giả lập đặt biển số và phương tiện vào khung hình camera giám sát thực địa (1280x720)
    theo từng kịch bản cụ thể:
    - OBLIQUE_ANGLE_30_45: Góc camera xiên 30 - 45 độ (gắn trên cột cao)
    - SUNNY_DAYLIGHT: Nắng gắt, bóng gắt
    - NIGHT_BACKLIGHT: Ban đêm, đèn xe rọi ngược sáng
    - RAINY_WET: Mưa ẩm, mặt đường tán xạ
    - PARKING_GATE: Cổng barrier bãi đỗ xe
    Trả về: (scene_img, vehicle_bbox, plate_bbox, plate_corners_4pt)
    """
    cw, ch = canvas_size
    is_two_line = spec["is_two_line"]
    pw, ph = (340, 200) if is_two_line else (520, 120)

    # 1. Tạo nền khung cảnh camera
    if scenario == "NIGHT_BACKLIGHT":
        scene = np.random.randint(15, 45, (ch, cw, 3), dtype=np.uint8)
        # Vệt sáng đèn đường / đèn pha
        cv2.circle(scene, (int(cw * 0.4), int(ch * 0.5)), 180, (60, 60, 80), -1)
    elif scenario == "SUNNY_DAYLIGHT":
        scene = np.random.randint(160, 220, (ch, cw, 3), dtype=np.uint8)
        # Đổ bóng nắng gắt
        cv2.line(scene, (0, 0), (cw, ch), (110, 110, 110), 120)
    elif scenario == "RAINY_WET":
        scene = np.random.randint(90, 140, (ch, cw, 3), dtype=np.uint8)
        # Vệt nước mưa
        for _ in range(15):
            rx = random.randint(0, cw - 50)
            cv2.ellipse(scene, (rx, ch - 80), (60, 15), 0, 0, 360, (140, 140, 150), -1)
    elif scenario == "OBLIQUE_ANGLE_30_45":
        scene = np.random.randint(110, 160, (ch, cw, 3), dtype=np.uint8)
    else:  # PARKING_GATE
        scene = np.random.randint(120, 170, (ch, cw, 3), dtype=np.uint8)

    # 2. Tạo Bounding Box Phương tiện (Vehicle Bounding Box)
    v_type = "motorcycle" if is_two_line and random.random() < 0.8 else "car"
    
    if v_type == "car":
        vw = random.randint(int(cw * 0.35), int(cw * 0.55))
        vh = random.randint(int(ch * 0.35), int(ch * 0.60))
    else:
        vw = random.randint(int(cw * 0.20), int(cw * 0.32))
        vh = random.randint(int(ch * 0.35), int(ch * 0.55))

    vx1 = random.randint(int(cw * 0.2), cw - vw - int(cw * 0.1))
    vy1 = random.randint(int(ch * 0.2), ch - vh - int(ch * 0.05))
    vx2, vy2 = vx1 + vw, vy1 + vh

    # Vẽ thân xe tượng trưng mờ
    v_color = (random.randint(40, 180), random.randint(40, 180), random.randint(40, 180))
    cv2.rectangle(scene, (vx1, vy1), (vx2, vy2), v_color, -1)
    cv2.rectangle(scene, (vx1, vy1), (vx2, vy2), (20, 20, 20), 2)

    # 3. Tính toán vị trí và góc nghiêng của Biển số (Plate Bounding Box & 4 Corners)
    # Tỷ lệ kích thước biển so với xe
    scale = (vw * 0.35) / pw
    target_pw = int(pw * scale)
    target_ph = int(ph * scale)

    # Đặt biển ở khoảng giữa phía dưới xe
    px_center = vx1 + vw // 2 + random.randint(-15, 15)
    py_center = vy1 + int(vh * 0.78)

    # 4 điểm góc hình chữ nhật phẳng tương đối
    hw, hh = target_pw // 2, target_ph // 2
    raw_corners = [
        [-hw, -hh], # Top-Left
        [hw, -hh],  # Top-Right
        [hw, hh],   # Bottom-Right
        [-hw, hh],  # Bottom-Left
    ]

    # Áp dụng góc xoay xạ ảnh (Perspective / Oblique tilt)
    if scenario == "OBLIQUE_ANGLE_30_45":
        # Góc xiên mạnh 30 - 45 độ: co ngắn một bên và kéo dài góc nhìn
        tilt_factor = random.uniform(0.25, 0.45)
        raw_corners[0][1] -= int(target_ph * tilt_factor) # TL nâng lên
        raw_corners[1][0] -= int(target_pw * tilt_factor * 0.6) # TR dịch vào
        raw_corners[2][0] -= int(target_pw * tilt_factor * 0.6) # BR dịch vào
        raw_corners[3][1] += int(target_ph * tilt_factor * 0.4) # BL kéo xuống
    else:
        # Góc nhìn camera giao thông / bãi xe thông thường (nghiêng nhẹ 5 - 20 độ)
        t_x = random.uniform(-0.10, 0.10)
        t_y = random.uniform(-0.10, 0.10)
        for i in range(4):
            raw_corners[i][0] += int(target_pw * t_x * (1 if i in (1, 2) else -1))
            raw_corners[i][1] += int(target_ph * t_y * (1 if i in (2, 3) else -1))

    # Tọa độ 4 góc tuyệt đối trên khung hình
    corners_abs = []
    for cx, cy in raw_corners:
        corners_abs.append([int(px_center + cx), int(py_center + cy)])

    # Warp phôi biển số vào 4 góc này
    src_pts = np.float32([
        [0, 0],
        [pw - 1, 0],
        [pw - 1, ph - 1],
        [0, ph - 1],
    ])
    dst_pts = np.float32(corners_abs)
    warp_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    # Tạo ảnh biển số đã warp
    warped_plate = cv2.warpPerspective(clean_plate, warp_matrix, (cw, ch), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full((ph, pw), 255, dtype=np.uint8), warp_matrix, (cw, ch), flags=cv2.INTER_NEAREST)

    # Đè biển số lên khung hình
    scene[mask > 0] = warped_plate[mask > 0]

    # Tính Bounding Box chữ nhật bao quanh 4 điểm góc biển số
    all_x = [pt[0] for pt in corners_abs]
    all_y = [pt[1] for pt in corners_abs]
    plate_bbox = [max(0, min(all_x)), max(0, min(all_y)), min(cw - 1, max(all_x)), min(ch - 1, max(all_y))]
    vehicle_bbox = [vx1, vy1, vx2, vy2]

    # Thêm nhiễu thời tiết theo kịch bản
    if scenario == "NIGHT_BACKLIGHT":
        # Chói đèn pha gần biển số
        glare = np.zeros((ch, cw), dtype=np.float32)
        cv2.circle(glare, (px_center - 80, py_center), 70, 0.8, -1)
        glare = cv2.GaussianBlur(glare, (51, 51), 0)
        scene = np.clip(scene.astype(np.float32) + glare[:, :, np.newaxis] * 255, 0, 255).astype(np.uint8)

    return scene, vehicle_bbox, plate_bbox, corners_abs


def generate_real_traffic_dataset(
    output_dir: str = "data/real_traffic_dataset",
    num_samples: int = 1000,
    rectify_verification: bool = True,
):
    """
    Sinh tập dữ liệu thực tế mẫu với đầy đủ 5 kịch bản môi trường và kiểm định nắn phẳng 4 góc.
    """
    manager = RealDatasetManager(output_dir=output_dir)
    renderer = LicensePlateRenderer()

    scenarios = [
        "SUNNY_DAYLIGHT",
        "NIGHT_BACKLIGHT",
        "RAINY_WET",
        "OBLIQUE_ANGLE_30_45",
        "PARKING_GATE",
    ]

    print(f"\n{'=' * 65}")
    print(f"BẮT ĐẦU TẠO TẬP DỮ LIỆU THỰC ĐỊA CAMERA: {num_samples:,} MẪU")
    print(f"Thư mục lưu trữ: {manager.output_dir}")
    print(f"Bao phủ 5 kịch bản: {', '.join(scenarios)}")
    print(f"{'=' * 65}\n")

    for i in tqdm(range(num_samples), desc="Tạo dữ liệu thực tế", unit="frame"):
        scen = random.choice(scenarios)
        spec = generate_plate_specification()
        clean = renderer.render(spec)
        weathered = apply_random_weathering(clean)

        frame_img, v_bbox, p_bbox, corners = create_realistic_scene_frame(
            spec=spec,
            clean_plate=weathered,
            scenario=scen,
            canvas_size=(1280, 720),
        )

        frame_filename = f"frame_{i:06d}_{scen}.jpg"
        frame_rel_path = os.path.join("frames", frame_filename)
        frame_abs_path = os.path.join(manager.frames_dir, frame_filename)

        cv2.imwrite(frame_abs_path, frame_img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

        ann = LicensePlateAnnotation(
            vehicle_type="motorcycle" if spec["is_two_line"] else "car",
            vehicle_bbox=v_bbox,
            plate_bbox=p_bbox,
            corners=corners,
            text=spec["normalized"],
            layout="2_LINE" if spec["is_two_line"] else "1_LINE",
            scenario=scen,
        )

        manager.add_frame_annotation(
            frame_id=f"frame_{i:06d}",
            image_path=frame_abs_path,
            image_size=(1280, 720),
            scenario=scen,
            annotations=[ann],
        )

        # Kiểm chứng nắn phẳng 4 góc (Perspective Rectification Verification) cho 50 mẫu đầu tiên
        if rectify_verification and i < 50:
            rectified = rectify_plate_quad(frame_img, corners, layout=ann.layout)
            crop_name = f"rectified_{i:06d}_{ann.layout}.jpg"
            cv2.imwrite(os.path.join(manager.crops_dir, crop_name), rectified)

    json_path, csv_path, obb_dir = manager.export_annotations()

    print(f"{'=' * 65}")
    print("HOÀN THÀNH BƯỚC 1.3: TẬP DỮ LIỆU THỰC ĐỊA ĐÃ ĐƯỢC THIẾT LẬP VÀ GÁN NHÃN!")
    print(f" -> Khung hình gốc:     {manager.frames_dir}")
    print(f" -> Biển đã nắn phẳng:  {manager.crops_dir}")
    print(f" -> Nhãn JSON:          {json_path}")
    print(f" -> Nhãn CSV cho OCR:   {csv_path}")
    print(f" -> Nhãn YOLO-OBB:      {obb_dir}")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    generate_real_traffic_dataset(num_samples=100)
