"""
FAST-PLATE-OCR-VIETNAMESE: High-Performance Synthetic Data Engine
Động cơ sinh tập dữ liệu lớn tự động kèm nhãn gán tuyệt đối cho mô hình OCR.
"""

import os
import sys
import csv
import time
import argparse
from typing import Dict, Any, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

# Đảm bảo root directory luôn có trong sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import cv2
from tqdm import tqdm

from src.synthetic.generator import generate_plate_specification
from src.synthetic.renderer import LicensePlateRenderer
from src.synthetic.textures import apply_random_weathering
from src.synthetic.augmentor import augment_plate_image


def generate_single_sample(args: Tuple[int, str, bool]) -> Dict[str, Any]:
    """
    Sinh 1 mẫu ảnh biển số hoàn chỉnh bao gồm tất cả các tầng nhiễu vật lý.
    args: (index, output_dir, save_image)
    """
    idx, output_dir, save_image = args
    
    # Khởi tạo renderer cục bộ cho từng tiến trình
    renderer = LicensePlateRenderer()

    # 1. Sinh cấu trúc văn bản hợp lệ theo ngữ pháp pháp lý
    spec = generate_plate_specification()

    # 2. Kết xuất hình ảnh đồ họa cơ sở
    clean_plate = renderer.render(spec)

    # 3. Phủ texture bề mặt vật lý (phay kim loại, xước, bùn đất, vết ố)
    weathered_plate = apply_random_weathering(clean_plate)

    # 4. Áp dụng góc xoay xạ ảnh 3D, bóng đổ, chói lóa, nhòe và nhiễu cảm biến
    final_plate = augment_plate_image(weathered_plate, is_square=spec["is_two_line"])

    # 5. Lưu ảnh vào đĩa nếu yêu cầu
    filename = f"plate_{idx:07d}.jpg"
    rel_path = os.path.join("images", filename)
    abs_path = os.path.join(output_dir, rel_path)

    if save_image:
        cv2.imwrite(abs_path, final_plate, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    return {
        "file_path": rel_path,
        "label": spec["normalized"],
        "display_text": spec["display_text"].replace("\n", " "),
        "layout": spec["layout"],
        "plate_type": spec["plate_type"],
        "province": spec.get("province", ""),
        "series": spec.get("series", ""),
        "number": spec.get("number", ""),
    }


class SyntheticDataEngine:
    """Động cơ tạo tập dữ liệu quy mô lớn (300.000 - 500.000 ảnh)."""

    def __init__(self, output_dir: str = "dataset/synthetic"):
        self.output_dir = os.path.abspath(output_dir)
        self.images_dir = os.path.join(self.output_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)

    def generate_dataset(
        self,
        num_samples: int = 1000,
        num_workers: int = 4,
        batch_size: int = 1000,
        save_images: bool = True,
    ) -> str:
        """
        Khởi chạy tiến trình sinh dữ liệu song song đa nhân.
        Ghi chú: num_workers=None sẽ tự động sử dụng toàn bộ CPU cores.
        """
        print(f"\n=======================================================")
        print(f"BẮT ĐẦU SINH TẬP DỮ LIỆU GIẢ LẬP: {num_samples:,} MẪU")
        print(f"Thư mục lưu trữ: {self.output_dir}")
        print(f"Số luồng CPU: {num_workers if num_workers else 'Tất cả lõi'}")
        print(f"=======================================================\n")

        csv_path = os.path.join(self.output_dir, "annotations.csv")
        start_time = time.time()

        tasks = [(i, self.output_dir, save_images) for i in range(num_samples)]
        
        fieldnames = [
            "file_path",
            "label",
            "display_text",
            "layout",
            "plate_type",
            "province",
            "series",
            "number",
        ]

        # Mở file CSV để ghi luân phiên từng batch
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                # Xử lý theo chunk với progress bar
                futures = [executor.submit(generate_single_sample, task) for task in tasks]
                
                with tqdm(total=num_samples, desc="Tiến độ sinh dữ liệu", unit="ảnh") as pbar:
                    for future in as_completed(futures):
                        record = future.result()
                        writer.writerow(record)
                        pbar.update(1)

        elapsed = time.time() - start_time
        fps = num_samples / elapsed if elapsed > 0 else 0
        print(f"\n[HOÀN THÀNH] Đã sinh {num_samples:,} ảnh trong {elapsed:.2f}s ({fps:.1f} ảnh/giây).")
        print(f"Tệp nhãn đã lưu tại: {csv_path}\n")
        return csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vietnamese Plate Synthetic Generator")
    parser.add_argument("--num-samples", type=int, default=100, help="Số lượng ảnh cần sinh")
    parser.add_argument("--workers", type=int, default=4, help="Số luồng xử lý CPU")
    parser.add_argument("--output-dir", type=str, default="dataset/synthetic", help="Thư mục xuất kết quả")
    parser.add_argument("--no-save", action="store_true", help="Chỉ benchmark tốc độ, không ghi ảnh ra đĩa")

    args = parser.parse_args()
    engine = SyntheticDataEngine(output_dir=args.output_dir)
    engine.generate_dataset(
        num_samples=args.num_samples,
        num_workers=args.workers,
        save_images=not args.no_save,
    )
