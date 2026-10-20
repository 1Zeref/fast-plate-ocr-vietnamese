"""
FAST-PLATE-OCR-VIETNAMESE: Unit Tests for CCT Architecture & Adapter (Step 3.1)
Kiểm thử tính tương thích cấu hình plate_config_vietnam.yaml, tầng Positional Embedding và 10 heads logits.
"""

import os
import sys
import yaml
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ocr.cct_adapter import CCTPlateVietnamModel
from src.grammar.dfa import VietnamPlateDFA
from src.grammar.constants import EXCLUDED_LETTERS


def test_plate_config_yaml():
    """Kiểm tra tính đúng đắn và chuẩn hóa của file plate_config_vietnam.yaml."""
    cfg_path = "configs/plate_config_vietnam.yaml"
    assert os.path.exists(cfg_path), f"Không tìm thấy cấu hình: {cfg_path}"

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 1. Ràng buộc slots và alphabet
    assert cfg["max_plate_slots"] == 10, "max_plate_slots phải là 10 để bao phủ toàn bộ biển số VN"
    assert len(cfg["alphabet"]) == 33, f"Alphabet phải có 33 ký tự, thực tế: {len(cfg['alphabet'])}"
    assert cfg["pad_char"] == "_", "Ký tự đệm phải là '_'"
    assert cfg["pad_char"] in cfg["alphabet"], "pad_char phải nằm trong alphabet"

    # Kiểm tra không chứa ký tự bị loại trừ (I, O, J, W)
    for ch in EXCLUDED_LETTERS:
        assert ch not in cfg["alphabet"], f"Ký tự bị cấm '{ch}' không được phép có trong alphabet"

    # 2. Ràng buộc kích thước hình ảnh
    assert cfg["img_height"] == 64, f"img_height phải là 64, thực tế: {cfg['img_height']}"
    assert cfg["img_width"] == 256, f"img_width phải là 256, thực tế: {cfg['img_width']}"
    assert cfg["channels"] == 3, "channels phải là 3"

    print("[PASS] test_plate_config_yaml (Cấu hình YAML chuẩn xác)")


def test_cct_model_forward_shape():
    """Kiểm tra forward pass qua mô hình CCT với tensor đầu vào chuẩn (B, 3, 64, 256)."""
    import torch
    model = CCTPlateVietnamModel(config_path="configs/plate_config_vietnam.yaml")
    model.eval()

    # Tạo batch 2 ảnh dummy kích thước chuẩn tắc sau nắn phẳng
    dummy_input = torch.randn(2, 3, 64, 256)

    with torch.no_grad():
        logits = model(dummy_input)

    # Kỳ vọng đầu ra: [Batch_Size=2, Max_Plate_Slots=10, Num_Classes=33]
    assert logits.shape == (2, 10, 33), f"Logits shape không đúng: {logits.shape} vs (2, 10, 33)"

    print(f"[PASS] test_cct_model_forward_shape (Logits: {list(logits.shape)})")


def test_cct_decode_and_grammar_validation():
    """Kiểm tra bộ giải mã (Decoder) và xác thực ngữ pháp với DFA Bước 1.1."""
    import torch
    model = CCTPlateVietnamModel(config_path="configs/plate_config_vietnam.yaml")

    # Giả lập logits của biển số ô tô hợp lệ: "30A12345__" (8 ký tự + 2 đệm)
    target_str = "30A12345__"
    logits = torch.full((1, 10, 33), -10.0)

    for slot_idx, ch in enumerate(target_str):
        class_idx = model.alphabet.index(ch)
        logits[0, slot_idx, class_idx] = 10.0 # Xác suất cực cao tại class_idx

    decoded = model.decode(logits)
    assert len(decoded) == 1
    res = decoded[0]

    # Kiểm tra text sau giải mã đã cắt bỏ ký tự padding '_'
    assert res["text"] == "30A12345", f"Kỳ vọng '30A12345', thực tế: '{res['text']}'"
    assert res["confidence"] > 0.99, f"Confidence kỳ vọng > 0.99, thực tế: {res['confidence']}"

    # Kiểm tra chứng thực qua DFA
    dfa = VietnamPlateDFA()
    is_valid, msg, meta = dfa.validate(res["text"])
    assert is_valid, f"Chuỗi giải mã '{res['text']}' không hợp lệ ngữ pháp: {msg}"
    assert meta["province_name"] == "Hà Nội"

    print(f"[PASS] test_cct_decode_and_grammar_validation (Giải mã '{res['text']}', Conf: {res['confidence']})")


if __name__ == "__main__":
    print("=" * 60)
    print("CHẠY BỘ KIỂM THỬ THIẾT LẬP CẤU HÌNH & ADAPTER CCT (BƯỚC 3.1)")
    print("=" * 60)
    test_plate_config_yaml()
    test_cct_model_forward_shape()
    test_cct_decode_and_grammar_validation()
    print("=" * 60)
    print("TẤT CẢ CÁC BÀI TEST BƯỚC 3.1 ĐÃ THÀNH CÔNG VƯỢT QUA!")
    print("=" * 60)
