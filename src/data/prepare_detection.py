"""
FAST-PLATE-OCR-VIETNAMESE: Detection Dataset Ingestion & Preparation
Tải và chuẩn hóa dataset toàn cảnh thực tế bomaich/vnlicenseplate phục vụ module Detection.
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import glob
import shutil
import json
import yaml
import cv2
from typing import Dict, Any, List

import kagglehub
from tqdm import tqdm


def prepare_detection_dataset(
    output_dir: str = "data/detection_dataset",
    copy_files: bool = True,
) -> Dict[str, Any]:
    """
    Chuẩn hóa và đồng bộ hóa tập dữ liệu bomaich/vnlicenseplate.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 65)
    print("[1/3] ĐỒNG BỘ DATASET PHÁT HIỆN BIỂN SỐ THỰC TẾ (bomaich/vnlicenseplate)")
    print("=" * 65)

    # 1. Tải hoặc lấy đường dẫn từ cache kagglehub
    print("-> Đang kiểm tra/tải dữ liệu qua kagglehub...")
    kaggle_path = kagglehub.dataset_download("bomaich/vnlicenseplate")
    print(f"-> Thư mục nguồn Kaggle: {kaggle_path}")

    stats: Dict[str, Any] = {}

    for split in ["train", "valid", "test"]:
        src_img_dir = os.path.join(kaggle_path, split, "images")
        src_lbl_dir = os.path.join(kaggle_path, split, "labels")

        dest_split_dir = os.path.join(output_dir, split)
        dest_img_dir = os.path.join(dest_split_dir, "images")
        dest_lbl_dir = os.path.join(dest_split_dir, "labels")

        os.makedirs(dest_img_dir, exist_ok=True)
        os.makedirs(dest_lbl_dir, exist_ok=True)

        if not os.path.exists(src_img_dir):
            continue

        img_files = [f for f in os.listdir(src_img_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        split_records: List[Dict[str, Any]] = []

        print(f"\n[Xử lý phân vùng: {split}] {len(img_files)} ảnh...")

        for fname in tqdm(img_files, desc=f"Đồng bộ {split}"):
            src_img_file = os.path.join(src_img_dir, fname)
            dest_img_file = os.path.join(dest_img_dir, fname)

            base_name, _ = os.path.splitext(fname)
            src_lbl_file = os.path.join(src_lbl_dir, f"{base_name}.txt")
            dest_lbl_file = os.path.join(dest_lbl_dir, f"{base_name}.txt")

            if copy_files and not os.path.exists(dest_img_file):
                shutil.copyfile(src_img_file, dest_img_file)

            # Đọc kích thước ảnh
            im = cv2.imread(src_img_file)
            if im is None:
                continue
            h, w = im.shape[:2]

            # Đọc nhãn YOLO
            objects: List[Dict[str, Any]] = []
            if os.path.exists(src_lbl_file):
                if copy_files and not os.path.exists(dest_lbl_file):
                    shutil.copyfile(src_lbl_file, dest_lbl_file)

                with open(src_lbl_file, "r") as lf:
                    lines = lf.readlines()

                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:5])

                        # Chuyển đổi sang pixel coordinates
                        x1 = int(round((cx - bw / 2.0) * w))
                        y1 = int(round((cy - bh / 2.0) * h))
                        x2 = int(round((cx + bw / 2.0) * w))
                        y2 = int(round((cy + bh / 2.0) * h))

                        # Khởi tạo 4 góc: Top-Left, Top-Right, Bottom-Right, Bottom-Left
                        corners = [
                            [x1, y1],
                            [x2, y1],
                            [x2, y2],
                            [x1, y2],
                        ]

                        objects.append({
                            "class_id": cls_id,
                            "bbox_yolo": [cx, cy, bw, bh],
                            "bbox_xyxy": [x1, y1, x2, y2],
                            "corners": corners,
                        })

            split_records.append({
                "image_filename": fname,
                "image_path": os.path.join(split, "images", fname),
                "width": w,
                "height": h,
                "num_plates": len(objects),
                "objects": objects,
            })

        # Lưu metadata json cho từng split
        meta_json_path = os.path.join(dest_split_dir, "metadata.json")
        with open(meta_json_path, "w", encoding="utf-8") as mf:
            json.dump(split_records, mf, indent=2, ensure_ascii=False)

        stats[split] = {
            "num_images": len(split_records),
            "num_plates": sum(r["num_plates"] for r in split_records),
            "metadata_file": meta_json_path,
        }

    # Ghi tệp dataset_info.yaml
    yaml_info = {
        "dataset_name": "bomaich/vnlicenseplate",
        "description": "Tập dữ liệu ảnh toàn cảnh thực tế phục vụ huấn luyện mô hình phát hiện biển số xe Việt Nam",
        "source": "https://www.kaggle.com/datasets/bomaich/vnlicenseplate",
        "num_classes": 1,
        "names": ["license_plate"],
        "splits": stats,
    }

    yaml_path = os.path.join(output_dir, "dataset_info.yaml")
    with open(yaml_path, "w", encoding="utf-8") as yf:
        yaml.dump(yaml_info, yf, default_flow_style=False, allow_unicode=True)

    print("\n" + "=" * 65)
    print("[ĐỒNG BỘ THÀNH CÔNG] Dữ liệu lưu tại:", output_dir)
    print(f"-> Train: {stats.get('train', {}).get('num_images', 0)} ảnh ({stats.get('train', {}).get('num_plates', 0)} biển số)")
    print(f"-> Valid: {stats.get('valid', {}).get('num_images', 0)} ảnh ({stats.get('valid', {}).get('num_plates', 0)} biển số)")
    print(f"-> Test:  {stats.get('test', {}).get('num_images', 0)} ảnh ({stats.get('test', {}).get('num_plates', 0)} biển số)")
    print("=" * 65)

    return stats


if __name__ == "__main__":
    prepare_detection_dataset()
