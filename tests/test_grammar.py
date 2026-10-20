"""
FAST-PLATE-OCR-VIETNAMESE: Unit Tests for Grammar & DFA
Kiểm thử tính đúng đắn của DFA và Bảng ký tự quy chuẩn.
"""

import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Thêm thư mục gốc vào PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.grammar.dfa import VietnamPlateDFA
from src.grammar.constants import ALPHABET, PAD_CHAR, EXCLUDED_LETTERS


def test_alphabet_constraints():
    """Kiểm tra ràng buộc của bảng ký tự."""
    assert PAD_CHAR in ALPHABET, "Ký tự pad phải nằm trong alphabet"
    for char in EXCLUDED_LETTERS:
        assert char not in ALPHABET, f"Ký tự bị cấm '{char}' không được phép có trong alphabet"
    # Tổng ký tự: 10 chữ số + 22 chữ cái + 1 pad = 33
    assert len(ALPHABET) == 33, f"Alphabet kỳ vọng 33 ký tự, thực tế: {len(ALPHABET)}"
    print("[PASS] test_alphabet_constraints")


def test_civilian_cars():
    """Kiểm tra biển ô tô dân sự (5 số, 4 số, sê-ri 1 & 2 chữ cái)."""
    dfa = VietnamPlateDFA()

    valid_plates = [
        "30A-123.45",
        "51K 999.99",
        "29AA-123.45",
        "43B-012.34",
        "29S-1234",
        "30H-5678",
        "80A-001.23",
    ]

    for p in valid_plates:
        ok, msg, meta = dfa.validate(p)
        assert ok, f"Lỗi với biển hợp lệ '{p}': {msg}"
        assert meta["province_code"] is not None

    print(f"[PASS] test_civilian_cars ({len(valid_plates)} mẫu)")


def test_motorcycles():
    """Kiểm tra biển xe máy (1 dòng và 2 dòng)."""
    dfa = VietnamPlateDFA()

    # Kiểm tra 1 dòng chuẩn hóa
    motorcycles_single = [
        "29B1-123.45",
        "59AA-678.90",
        "29MD1-123.45",
        "29H4-1234",
    ]
    for p in motorcycles_single:
        ok, msg, meta = dfa.validate(p)
        assert ok, f"Lỗi với biển xe máy '{p}': {msg}"

    # Kiểm tra biển 2 dòng (vuông)
    double_lines = [
        ("29-B1", "123.45"),
        ("59-AA", "678.90"),
        ("29-H4", "1234"),
    ]
    for l1, l2 in double_lines:
        ok, msg, meta = dfa.validate_double_line(l1, l2)
        assert ok, f"Lỗi với biển 2 dòng '{l1}' - '{l2}': {msg}"

    print(f"[PASS] test_motorcycles ({len(motorcycles_single) + len(double_lines)} mẫu)")


def test_military():
    """Kiểm tra biển số quân đội theo TT 169/2021/TT-BQP."""
    dfa = VietnamPlateDFA()

    valid_military = [
        "TM-12-34",
        "QA-56-78",
        "VT-123.45",
        "BB-99-99",
        "TC-01-23",
        "KA-12-34",
    ]
    for p in valid_military:
        ok, msg, meta = dfa.validate(p)
        assert ok, f"Lỗi với biển quân sự '{p}': {msg}"
        assert meta["plate_type"] == "MILITARY"
        assert meta["military_unit"] is not None

    print(f"[PASS] test_military ({len(valid_military)} mẫu)")


def test_diplomatic():
    """Kiểm tra biển số ngoại giao và tổ chức quốc tế."""
    dfa = VietnamPlateDFA()

    valid_diplomatic = [
        "80-NG-166-01",
        "29-QT-001-02",
        "51-NN-123-45",
        "80-CV-050-01",
    ]
    for p in valid_diplomatic:
        ok, msg, meta = dfa.validate(p)
        assert ok, f"Lỗi với biển ngoại giao '{p}': {msg}"
        assert meta["plate_type"] == "DIPLOMATIC"
        assert meta["diplomatic_org"] is not None

    print(f"[PASS] test_diplomatic ({len(valid_diplomatic)} mẫu)")


def test_special_vehicles():
    """Kiểm tra xe chuyên dùng (Rơ-moóc, máy kéo, liên doanh, dự án)."""
    dfa = VietnamPlateDFA()

    valid_special = [
        "51R-123.45",
        "29RM-1234",
        "34MK-1234",
        "29LD-123.45",
        "51DA-999.99",
    ]
    for p in valid_special:
        ok, msg, meta = dfa.validate(p)
        assert ok, f"Lỗi với xe chuyên dùng '{p}': {msg}"

    print(f"[PASS] test_special_vehicles ({len(valid_special)} mẫu)")


def test_invalid_plates():
    """Kiểm tra phát hiện biển giả mạo hoặc sai quy chuẩn."""
    dfa = VietnamPlateDFA()

    invalid_cases = [
        ("30I-123.45", "Chứa ký tự bị cấm 'I'"),
        ("29O-123.45", "Chứa ký tự bị cấm 'O'"),
        ("51J-123.45", "Chứa ký tự bị cấm 'J'"),
        ("29W-123.45", "Chứa ký tự bị cấm 'W'"),
        ("13A-123.45", "Mã tỉnh 13 không tồn tại"),
        ("41B-999.99", "Mã tỉnh 41 không tồn tại"),
        ("87C-123.45", "Mã tỉnh 87 không tồn tại"),
        ("96K-123.45", "Mã tỉnh 96 không tồn tại"),
        ("ZZ-12-34",   "Mã quân sự ZZ không tồn tại"),
        ("80-NG-12",   "Biển ngoại giao thiếu số"),
        ("30A-12",     "Số quá ngắn"),
    ]

    for p, reason in invalid_cases:
        ok, msg, meta = dfa.validate(p)
        assert not ok, f"Biển '{p}' ({reason}) không được phép chấp nhận! Msg: {msg}"

    print(f"[PASS] test_invalid_plates ({len(invalid_cases)} mẫu vi phạm)")


if __name__ == "__main__":
    print("=" * 60)
    print("CHẠY BỘ KIỂM THỬ KHÔNG GIAN NGỮ PHÁP BIỂN SỐ XE VIỆT NAM")
    print("=" * 60)
    test_alphabet_constraints()
    test_civilian_cars()
    test_motorcycles()
    test_military()
    test_diplomatic()
    test_special_vehicles()
    test_invalid_plates()
    print("=" * 60)
    print("TẤT CẢ CÁC BÀI TEST ĐÃ THÀNH CÔNG VƯỢT QUA!")
    print("=" * 60)
