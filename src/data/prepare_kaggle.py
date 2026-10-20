"""
FAST-PLATE-OCR-VIETNAMESE: Kaggle Dataset Ingestion & Sanitization Pipeline
Nạp và chuẩn hóa dữ liệu thực tế từ Kaggle: topkek69/vietnamese-license-plate-ocr.
Lọc qua máy trạng thái DFA để đảm bảo 100% nhãn đúng quy chuẩn pháp lý và chuẩn format fast-plate-ocr.
"""

import os
import sys
import shutil
import random
import pandas as pd
from typing import Dict, Any, List, Tuple
from tqdm import tqdm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import kagglehub
from src.grammar.dfa import VietnamPlateDFA


def sanitize_label(raw: str) -> str:
    """Sửa lỗi typo thông dụng của người gán nhãn (ví dụ gõ nhầm O thành 0)."""
    clean = str(raw).strip().upper().replace("-", "").replace(".", "").replace(" ", "").replace("_", "")
    
    # Sửa lỗi gõ chữ O ở vị trí số đầu tiên (mã tỉnh): VD "BOE" -> "80E" hoặc "3OE" -> "30E"
    if len(clean) >= 4:
        first_two = clean[:2]
        if first_two.startswith("B") and first_two[1] in "0123456789":
            pass # Có thể là biển quân sự BB, BC...
        # Chữ O ở vị trí thứ 2 của mã tỉnh (ví dụ '3O' -> '30', '5O' -> '50', '8O' -> '80')
        if clean[0].isdigit() and clean[1] == "O":
            clean = clean[0] + "0" + clean[2:]
        # Chữ O ở các vị trí số đuôi sau chữ cái
        # Tìm vị trí ký tự chữ cái cuối cùng
        last_letter_idx = -1
        for i, ch in enumerate(clean):
            if ch.isalpha():
                last_letter_idx = i
        if last_letter_idx != -1 and last_letter_idx < len(clean) - 1:
            # Toàn bộ phần sau chữ cái phải là số
            prefix = clean[:last_letter_idx + 1]
            tail = clean[last_letter_idx + 1:].replace("O", "0").replace("I", "1")
            clean = prefix + tail

    return clean


def prepare_kaggle_dataset(
    output_dir: str = "data/kaggle_dataset",
    val_split: float = 0.15,
    seed: int = 42,
    copy_files: bool = True,
):
    """
    Tải qua kagglehub, tiền xử lý, lọc qua DFA và xuất tập train/val chuẩn cho fast-plate-ocr.
    """
    print(f"\n{'=' * 65}")
    print("BẮT ĐẦU NẠP VÀ CHUẨN HÓA DATASET: topkek69/vietnamese-license-plate-ocr")
    print(f"{'=' * 65}")

    # 1. Tải bản mới nhất từ Kaggle thông qua kagglehub
    print("\n[Bước 1/4] Đang kiểm tra / tải dataset từ Kaggle qua kagglehub...")
    kaggle_path = kagglehub.dataset_download("topkek69/vietnamese-license-plate-ocr")
    print(f"-> Thư mục nguồn Kaggle: {kaggle_path}")

    # 2. Đọc file nhãn gốc
    crop_csv = os.path.join(kaggle_path, "labels", "crop_labels.csv")
    gen_csv = os.path.join(kaggle_path, "labels", "gen_labels.csv")

    records: List[Dict[str, Any]] = []

    if os.path.exists(crop_csv):
        df_crop = pd.read_csv(crop_csv)
        for _, row in df_crop.iterrows():
            src_img = os.path.join(kaggle_path, "cropped", str(row["Name"]))
            if os.path.exists(src_img):
                records.append({
                    "src_path": src_img,
                    "filename": str(row["Name"]),
                    "raw_label": str(row["Label"]),
                    "kaggle_type": row["Type"],
                    "source_category": "real_cropped",
                })

    if os.path.exists(gen_csv):
        df_gen = pd.read_csv(gen_csv)
        for _, row in df_gen.iterrows():
            src_img = os.path.join(kaggle_path, "generated", str(row["Name"]))
            if os.path.exists(src_img):
                records.append({
                    "src_path": src_img,
                    "filename": str(row["Name"]),
                    "raw_label": str(row["Label"]),
                    "kaggle_type": row["Type"],
                    "source_category": "kaggle_generated",
                })

    print(f"-> Tổng số mẫu phát hiện từ nguồn Kaggle: {len(records):,} mẫu.")

    # 3. Lọc qua Máy trạng thái hữu hạn (DFA Sanitizer)
    print("\n[Bước 2/4] Kiểm định cú pháp và chuẩn hóa qua DFA (Bước 1.1)...")
    dfa = VietnamPlateDFA()
    valid_records: List[Dict[str, Any]] = []
    rejected_count = 0

    dest_images_dir = os.path.join(output_dir, "images")
    os.makedirs(dest_images_dir, exist_ok=True)

    for rec in records:
        cleaned_label = sanitize_label(rec["raw_label"])
        is_valid, msg, meta = dfa.validate(cleaned_label)
        
        if is_valid:
            # Xác định layout hình học dựa trên thông số
            # Type 0 thường là biển dài 1 dòng hoặc ô tô, Type 1/2 là vuông hoặc đặc biệt
            k_type = rec["kaggle_type"]
            layout = "SQUARE" if k_type in (1, 2) else "LONG"

            rec["clean_label"] = cleaned_label
            rec["plate_type"] = meta.get("plate_type", "UNKNOWN")
            rec["province"] = meta.get("province_code", "")
            rec["layout"] = layout
            valid_records.append(rec)
        else:
            rejected_count += 1

    print(f"-> Mẫu hợp lệ đạt chuẩn DFA: {len(valid_records):,} / {len(records):,} ({len(valid_records)/len(records)*100:.2f}%)")
    print(f"-> Mẫu bị loại trừ (do gán nhãn sai/ảnh cắt cụt): {rejected_count:,} mẫu.")

    # 4. Sao chép ảnh vào workspace và phân chia train / val
    print(f"\n[Bước 3/4] Sao chép ảnh vào thư mục dự án: {dest_images_dir}...")
    random.seed(seed)
    random.shuffle(valid_records)

    processed_records: List[Dict[str, Any]] = []
    for rec in tqdm(valid_records, desc="Sao chép dữ liệu", unit="ảnh"):
        dest_filename = f"{rec['source_category']}_{rec['filename']}"
        dest_path = os.path.join(dest_images_dir, dest_filename)
        rel_img_path = os.path.join("images", dest_filename)

        if copy_files and not os.path.exists(dest_path):
            shutil.copyfile(rec["src_path"], dest_path)

        processed_records.append({
            "filepath": rel_img_path,
            "label": rec["clean_label"],
            "layout": rec["layout"],
            "plate_type": rec["plate_type"],
            "province": rec["province"],
            "source": rec["source_category"],
        })

    # 5. Phân chia Train / Validation
    print("\n[Bước 4/4] Phân chia tập Train / Val theo chuẩn fast-plate-ocr...")
    total_samples = len(processed_records)
    val_count = int(total_samples * val_split)
    train_count = total_samples - val_count

    val_records = processed_records[:val_count]
    train_records = processed_records[val_count:]

    # Xuất các file annotations CSV
    df_all = pd.DataFrame(processed_records)
    df_train = pd.DataFrame(train_records)
    df_val = pd.DataFrame(val_records)

    all_csv = os.path.join(output_dir, "annotations.csv")
    train_csv = os.path.join(output_dir, "train.csv")
    val_csv = os.path.join(output_dir, "val.csv")

    # Format chuẩn của fast-plate-ocr: filepath,label (kèm thông tin bổ trợ)
    df_all.to_csv(all_csv, index=False, encoding="utf-8")
    df_train.to_csv(train_csv, index=False, encoding="utf-8")
    df_val.to_csv(val_csv, index=False, encoding="utf-8")

    print(f"-> Toàn bộ tập dữ liệu ({len(df_all):,} mẫu): {all_csv}")
    print(f"-> Tập huấn luyện Train ({len(df_train):,} mẫu): {train_csv}")
    print(f"-> Tập kiểm định Validation ({len(df_val):,} mẫu): {val_csv}")
    print(f"{'=' * 65}")
    print("HOÀN THÀNH TIỀN XỬ LÝ VÀ CHUẨN HÓA DATASET KAGGLE THÀNH CÔNG!")
    print(f"{'=' * 65}\n")

    return {
        "all_csv": all_csv,
        "train_csv": train_csv,
        "val_csv": val_csv,
        "total_valid": len(df_all),
        "train_count": len(df_train),
        "val_count": len(df_val),
    }


if __name__ == "__main__":
    prepare_kaggle_dataset(copy_files=True)
