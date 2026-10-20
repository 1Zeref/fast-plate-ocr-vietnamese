"""
FAST-PLATE-OCR-VIETNAMESE: Synthetic & Grammar Consistency Test
Kiểm tra chéo giữa Động cơ sinh dữ liệu (Bước 1.2) và Bộ máy kiểm định ngữ pháp DFA (Bước 1.1).
"""

import os
import sys
import csv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.grammar.dfa import VietnamPlateDFA
from src.synthetic.engine import SyntheticDataEngine


def test_consistency():
    output_dir = "dataset/test_consistency"
    num_samples = 200
    engine = SyntheticDataEngine(output_dir=output_dir)
    csv_file = engine.generate_dataset(num_samples=num_samples, num_workers=2, save_images=False)

    dfa = VietnamPlateDFA()

    total_tested = 0
    passed = 0
    failed = []

    with open(csv_file, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_tested += 1
            label = row["label"]
            layout = row["layout"]
            ok, msg, meta = dfa.validate(label)
            if ok:
                passed += 1
            else:
                failed.append((label, row["display_text"], msg))

    print(f"\n============================================================")
    print(f"KẾT QUẢ KIỂM TRA TƯƠNG THÍCH SYNTHETIC <-> GRAMMAR DFA:")
    print(f"Tổng số mẫu thử: {total_tested}")
    print(f"Số mẫu hợp lệ ngữ pháp: {passed} / {total_tested} ({passed / total_tested * 100:.2f}%)")
    
    if failed:
        print(f"CÁC MẪU LỖI:")
        for lab, disp, err in failed[:10]:
            print(f" - Chuỗi: {lab} | Hiển thị: {disp} | Lỗi: {err}")
        assert False, f"Có {len(failed)} mẫu không vượt qua DFA!"

    print("TẤT CẢ CÁC MẪU SINH RA ĐỀU KHỚP HOÀN TOÀN VỚI NGỮ PHÁP QUY CHUẨN!")
    print(f"============================================================\n")


if __name__ == "__main__":
    test_consistency()
