"""
FAST-PLATE-OCR-VIETNAMESE: Real-World Data Annotator & Quad Rectifier
Chuẩn hóa định dạng nhãn dữ liệu thực địa: Bbox phương tiện, Bbox biển số,
Tọa độ 4 điểm góc (x1,y1)..(x4,y4), Chuỗi ký tự OCR và Cờ phân loại (1 dòng / 2 dòng).
Tích hợp thuật toán nắn phẳng xạ ảnh 4 điểm góc (Perspective Rectification).
"""

import os
import sys
import json
import csv
import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.grammar.dfa import VietnamPlateDFA


class LicensePlateAnnotation:
    """Đặc tả nhãn chi tiết cho một phương tiện và biển số xe."""

    def __init__(
        self,
        vehicle_type: str,
        vehicle_bbox: List[int],  # [vx_min, vy_min, vx_max, vy_max]
        plate_bbox: List[int],    # [px_min, py_min, px_max, py_max]
        corners: List[List[int]], # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] (TL, TR, BR, BL)
        text: str,
        layout: str,              # "1_LINE" hoặc "2_LINE"
        scenario: str = "STANDARD",
        confidence: float = 1.0,
    ):
        self.vehicle_type = vehicle_type
        self.vehicle_bbox = vehicle_bbox
        self.plate_bbox = plate_bbox
        self.corners = corners
        self.text = text.strip().upper().replace("-", "").replace(".", "").replace(" ", "")
        self.layout = layout
        self.scenario = scenario
        self.confidence = confidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vehicle_type": self.vehicle_type,
            "vehicle_bbox": self.vehicle_bbox,
            "plate_bbox": self.plate_bbox,
            "corners": self.corners,
            "text": self.text,
            "layout": self.layout,
            "scenario": self.scenario,
            "confidence": self.confidence,
        }


def rectify_plate_quad(
    image: np.ndarray,
    corners: List[List[float]],
    target_size: Optional[Tuple[int, int]] = None,
    layout: str = "1_LINE",
) -> np.ndarray:
    """
    Nắn thẳng biển số xe bị nghiêng/chụp góc xiên 30-45 độ dựa vào 4 điểm góc (Perspective Warp).
    - corners: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] theo thứ tự Top-Left, Top-Right, Bottom-Right, Bottom-Left.
    - target_size: (Width, Height). Nếu None, tự động chọn kích thước chuẩn (dài: 520x120, vuông: 340x200).
    """
    if target_size is None:
        target_size = (340, 200) if layout == "2_LINE" else (520, 120)

    target_w, target_h = target_size
    src_pts = np.float32(corners)
    dst_pts = np.float32([
        [0, 0],
        [target_w - 1, 0],
        [target_w - 1, target_h - 1],
        [0, target_h - 1],
    ])

    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    rectified = cv2.warpPerspective(
        image,
        matrix,
        (target_w, target_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rectified


class RealDatasetManager:
    """Quản lý, xác thực và xuất tập dữ liệu thực tế theo các định dạng chuẩn quốc tế."""

    def __init__(self, output_dir: str = "data/real_traffic_dataset"):
        self.output_dir = os.path.abspath(output_dir)
        self.frames_dir = os.path.join(self.output_dir, "frames")
        self.crops_dir = os.path.join(self.output_dir, "rectified_crops")
        os.makedirs(self.frames_dir, exist_ok=True)
        os.makedirs(self.crops_dir, exist_ok=True)
        self.dfa = VietnamPlateDFA()
        self.dataset_records: List[Dict[str, Any]] = []

    def add_frame_annotation(
        self,
        frame_id: str,
        image_path: str,
        image_size: Tuple[int, int], # (Width, Height)
        scenario: str,
        annotations: List[LicensePlateAnnotation],
    ):
        """Thêm một frame đã gán nhãn vào tập dữ liệu và kiểm định DFA."""
        validated_objects = []
        for ann in annotations:
            ann_dict = ann.to_dict()
            # Kiểm định cú pháp với DFA Bước 1.1
            is_valid, msg, meta = self.dfa.validate(ann.text)
            ann_dict["grammar_valid"] = is_valid
            ann_dict["plate_type"] = meta.get("plate_type", "UNKNOWN")
            ann_dict["province_name"] = meta.get("province_name", "")
            validated_objects.append(ann_dict)

        record = {
            "frame_id": frame_id,
            "image_path": os.path.relpath(image_path, self.output_dir) if os.path.isabs(image_path) else image_path,
            "width": image_size[0],
            "height": image_size[1],
            "scenario": scenario,
            "num_objects": len(validated_objects),
            "objects": validated_objects,
        }
        self.dataset_records.append(record)

    def export_annotations(self):
        """
        Xuất nhãn ra 3 định dạng:
        1. JSON: Toàn diện thông tin phân cấp (Bounding box, 4 góc, OCR, Layout, Kịch bản).
        2. CSV: Tương thích trực tiếp với pipeline huấn luyện fast-plate-ocr.
        3. YOLO-OBB: Định dạng Oriented Bounding Box cho bài toán phát hiện biển nghiêng.
        """
        # 1. Xuất file JSON
        json_path = os.path.join(self.output_dir, "annotations_full.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "version": "1.0",
                "description": "Vietnam Traffic Real-World Plate Dataset (Surveillance & Parking)",
                "total_frames": len(self.dataset_records),
                "frames": self.dataset_records,
            }, f, indent=2, ensure_ascii=False)

        # 2. Xuất file CSV phẳng (phục vụ OCR)
        csv_path = os.path.join(self.output_dir, "ocr_annotations.csv")
        csv_records = []
        for frame in self.dataset_records:
            f_path = frame["image_path"]
            scen = frame["scenario"]
            for obj in frame["objects"]:
                corners_str = ";".join([f"{c[0]},{c[1]}" for c in obj["corners"]])
                p_bbox_str = ",".join(map(str, obj["plate_bbox"]))
                v_bbox_str = ",".join(map(str, obj["vehicle_bbox"]))
                csv_records.append({
                    "frame_path": f_path,
                    "label": obj["text"],
                    "layout": obj["layout"],
                    "vehicle_type": obj["vehicle_type"],
                    "scenario": scen,
                    "plate_bbox": p_bbox_str,
                    "corners_4pt": corners_str,
                    "vehicle_bbox": v_bbox_str,
                    "grammar_valid": obj["grammar_valid"],
                })

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            if csv_records:
                writer = csv.DictWriter(f, fieldnames=list(csv_records[0].keys()))
                writer.writeheader()
                writer.writerows(csv_records)

        # 3. Xuất thư mục nhãn YOLO-OBB (Oriented Bounding Box)
        obb_dir = os.path.join(self.output_dir, "yolo_obb_labels")
        os.makedirs(obb_dir, exist_ok=True)
        for frame in self.dataset_records:
            base_name = os.path.splitext(os.path.basename(frame["image_path"]))[0]
            txt_path = os.path.join(obb_dir, f"{base_name}.txt")
            w, h = frame["width"], frame["height"]
            
            with open(txt_path, "w", encoding="utf-8") as f:
                for obj in frame["objects"]:
                    # Lớp 0: Biển 1 dòng, Lớp 1: Biển 2 dòng
                    cls_id = 0 if obj["layout"] == "1_LINE" else 1
                    # Chuẩn hóa tọa độ 4 góc về dải [0, 1]
                    norm_pts = []
                    for pt in obj["corners"]:
                        norm_pts.append(f"{pt[0] / w:.6f}")
                        norm_pts.append(f"{pt[1] / h:.6f}")
                    line = f"{cls_id} " + " ".join(norm_pts) + "\n"
                    f.write(line)

        print(f"\n[XUẤT NHÃN THÀNH CÔNG]")
        print(f" -> Tệp JSON chi tiết: {json_path}")
        print(f" -> Tệp CSV cho OCR:  {csv_path}")
        print(f" -> Thư mục YOLO-OBB:  {obb_dir}\n")
        return json_path, csv_path, obb_dir
