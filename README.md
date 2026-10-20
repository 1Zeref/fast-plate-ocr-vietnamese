# Fast Plate OCR Vietnamese 🚗🇻🇳

Nhận dạng ký tự quang học biển số xe Việt Nam (LPR/OCR) tốc độ cao dựa trên kiến trúc **Fast Plate OCR** (Compact Convolutional Transformer - CCT), tích hợp bản đồ ngữ pháp quy chuẩn pháp lý Việt Nam (Thông tư 24/2023/TT-BCA & Thông tư 169/2021/TT-BQP).

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

---

## 🌟 Tính Năng Nổi Bật

1. **Bản đồ Ngữ pháp & Bảng ký tự Quy chuẩn (DFA Grammar Engine):**
   - Loại trừ tuyệt đối các ký tự không hợp lệ: `I, O, J, W`.
   - Bảng chữ cái 33 tokens: 10 số + 22 chữ cái Latinh hợp lệ (`Q` quân sự/quốc tế, `R` rơ-moóc) + token padding `_`.
   - Kiểm định và phân loại 5 nhóm biển số: Ô tô dân sự (5 số, 4 số), Xe máy (2 dòng), Xe kinh doanh (biển vàng), Cơ quan nhà nước (biển xanh), Quân đội (biển đỏ) và Ngoại giao (`NG, QT, NN, CV`).
   - Kiểm định danh mục mã tỉnh thành 11–99 hợp lệ.

2. **Tích hợp Dataset thực tế Kaggle (`topkek69/vietnamese-license-plate-ocr`):**
   - Tự động tải thông qua `kagglehub`.
   - Lọc sạch dữ liệu qua máy trạng thái DFA: 11,619 mẫu đạt chuẩn (95.32%).
   - Tách sẵn tập huấn luyện `train.csv` (9,877 mẫu) và kiểm định `val.csv` (1,742 mẫu).

3. **Động cơ Sinh Dữ liệu Giả lập (Synthetic Data Engine):**
   - Sinh từ 300,000 – 500,000 mẫu ảnh biển số bổ trợ với tốc độ >150 ảnh/giây.
   - Mô phỏng đầy đủ hiệu ứng vật lý thực địa: phay xước nhôm, vết cào rách, đốm bùn đất, xoay xạ ảnh 3D (Homography), chói lóa ánh nắng/đèn pha, bóng đổ, nhòe chuyển động và nhiễu cảm biến camera.

4. **Tài liệu nguồn minh bạch:**
   - Đầy đủ nguồn gốc tra cứu tại [`docs/README.md`](docs/README.md).

---

## 🚀 Cài Đặt Nhanh

```bash
# Cài đặt thư viện phụ thuộc
pip install opencv-python pillow pyyaml pandas tqdm kagglehub
```

---

## 🛠️ Hướng Dẫn Sử Dụng

### 1. Kiểm tra tính hợp lệ của biển số qua DFA
```python
from src.grammar.dfa import VietnamPlateDFA

dfa = VietnamPlateDFA()

# Kiểm tra biển ô tô 5 số
is_valid, msg, meta = dfa.validate("30A-123.45")
print(is_valid, meta["province_name"])  # True, Hà Nội

# Kiểm tra biển quân sự
is_valid, msg, meta = dfa.validate("TM-12-34")
print(is_valid, meta["military_unit"])  # True, Bộ Tổng Tham mưu - Cơ quan BQP
```

### 2. Tải và chuẩn hóa dataset Kaggle
```bash
py src/data/prepare_kaggle.py
```

### 3. Sinh thêm dữ liệu giả lập (Synthetic Engine)
```bash
# Sinh 10,000 mẫu thử nghiệm
py src/synthetic/engine.py --num-samples 10000 --workers 4 --output-dir dataset/synthetic
```

---

## 📖 Tham Khảo Nguồn Dữ Liệu
Chi tiết đầy đủ về nguồn gốc mã nguồn, dataset Kaggle, thông tư quy chuẩn pháp lý BCA/BQP và phông chữ xem tại:
👉 **[Tài liệu nguồn chi tiết: docs/README.md](docs/README.md)**
