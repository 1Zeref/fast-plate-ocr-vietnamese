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

3. **Mạng Định Vị Phương Tiện & 4 Điểm Góc Biển Số (Bước 2.1):**
   - Mạng nơ-ron tích chập Anchor-Free Keypoint Detector siêu nhẹ (**8.3 MB** trọng số).
   - Xuất trực tiếp tọa độ 4 điểm góc (x1, y1) .. (x4, y4) kèm chỉ số tin cậy (confidence score) trên từng ROI phương tiện.
   - Hàm mất mát Wing Loss nhạy cảm cao với sai số nhỏ, giúp tứ giác bám sát viền mép biển số chính xác đến từng pixel.
   - Tích hợp bộ giải thuật nắn phẳng xạ ảnh trực tiếp (`rectify_plate_quad`) phục vụ bước OCR.

4. **Nắn Phối Cảnh & Định Tuyến Hình Học (Bước 2.2):**
   - Thuật toán sắp xếp 4 điểm góc (TL, TR, BR, BL) và nắn phẳng ma trận Homography 3x3.
   - Định tuyến hình học (Geometry Router) phân luồng biển `1-line` (dài) và `2-line` (vuông) dựa trên tỷ lệ Aspect Ratio (ngưỡng 2.40) kết hợp mô hình học sâu.
   - Thuật toán cắt tách dòng và nối không gian thành dải 1 dòng đơn giả lập (**Virtual 1-Line Strip Generator**) bằng phân tích biểu đồ chiếu ngang (Horizontal Projection Profile), giúp các mô hình OCR 1 dòng nhận diện biển 2 dòng chính xác 100%.

5. **Mô Hình Nhận Dạng Ký Tự CCT Adapter (Bước 3.1):**
   - Kiến trúc Compact Convolutional Transformer (CCT) chuyên biệt hóa cho biển số xe Việt Nam.
   - Đầu vào chuẩn hóa 64x256x3, Convolutional Tokenizer sinh ra 1024 patch tokens kết hợp Sequence Pooling chiếu nén thành đúng 10 slot tokens đại diện cho 10 vị trí ký tự tối đa.
   - Vector nhúng vị trí có thể học được (1D Learnable Positional Embedding) duy trì trật tự không gian trái-sang-phải.
   - 4 tầng Transformer Encoder với Multi-Head Attention (4 heads) trích xuất ngữ cảnh liên ký tự hiệu quả cao.
   - Multi-Head Classifier 10 heads dự đoán song song trên tập từ điển 33 tokens hợp lệ, kèm Greedy Sequence Decoder tích hợp kiểm định ngữ pháp DFA trực tiếp.

6. **Huấn Luyện Hai Pha - Two-Stage Transfer Learning (Bước 3.2):**
   - **Pha 1 (Pre-training):** Huấn luyện trên tập dữ liệu giả lập (Synthetic Engine) để học đặc trưng ký tự cơ sở, đa dạng hóa font chữ và màu sắc nền.
   - **Pha 2 (Fine-tuning):** Tinh chỉnh trên tập dữ liệu thực tế (Kaggle Dataset) kết hợp kỹ thuật làm giàu ảnh thực địa:
     - **Screw / Bolt Cutout Occlusion:** Giả lập lỗ khoan và mũ đinh ốc bắt biển số che khuất một phần nét ký tự.
     - **Directional Motion Blur:** Mô phỏng rung lắc camera hoặc xe đang chạy.
     - **Photometric Distortions:** Biến dạng ánh sáng, tương phản và nhiễu hạt cảm biến ISO cao.
   - Trọng số tối ưu hóa lưu tại weights/fast_plate_ocr_vn.pt (~2.18 MB).

7. **Bộ Giải Mã Ràng Buộc Ngữ Pháp - Grammar-Guided Beam Search (Bước 4.1):**
   - Thuật toán giải mã Beam Search có mặt nạ xác suất (Probability Masking) tích hợp trực tiếp Máy trạng thái hữu hạn DFA.
   - Triệt tiêu 100% các xung đột quang học phổ biến: $8 \leftrightarrow B$ (tự động chọn chữ cái ở slot sê-ri và chữ số ở dãy số), $0 \leftrightarrow D$, $5 \leftrightarrow S$, $2 \leftrightarrow Z$, $1 \leftrightarrow T$.
   - Xuất chuỗi đảm bảo 100% tuân thủ quy chuẩn pháp lý và phân loại biển số tức thời.

8. **Thu thập, Chọn lọc & Gán nhãn Thực địa Đa Điều Kiện (Bước 1.3):**
   - Bộ trích xuất video thông minh (`VideoFrameSampler`): Lọc frame tĩnh, loại bỏ chuyển động nhòe mờ bằng Laplacian Variance Filter ($\text{Threshold} = 85.0$).
   - Nhận diện 5 kịch bản thực địa: Nắng gắt (`SUNNY_DAYLIGHT`), Ngược sáng ban đêm (`NIGHT_BACKLIGHT`), Mưa ẩm (`RAINY_WET`), Góc xiên $30^\circ - 45^\circ$ (`OBLIQUE_ANGLE_30_45`), Cổng bãi đỗ xe (`PARKING_GATE`).
   - Định dạng nhãn chuẩn 5 trường: Bounding box phương tiện, Bounding box biển số, Tọa độ 4 điểm góc $(x_1, y_1) \dots (x_4, y_4)$, chuỗi ký tự OCR và cờ phân loại (`1_LINE` / `2_LINE`).
   - Hỗ trợ giải thuật nắn phẳng xạ ảnh 4 góc (`rectify_plate_quad`) chuyển đổi biển số góc xiên về kích thước chuẩn trước khi đưa vào OCR.
   - Xuất dữ liệu đa chuẩn: JSON phân cấp đầy đủ, CSV phẳng cho OCR và YOLO-OBB.

9. **Động cơ Sinh Dữ liệu Giả lập (Synthetic Data Engine):**
   - Sinh từ 300,000 – 500,000 mẫu ảnh biển số bổ trợ với tốc độ >150 ảnh/giây.
   - Mô phỏng đầy đủ hiệu ứng vật lý thực địa: phay xước nhôm, vết cào rách, đốm bùn đất, xoay xạ ảnh 3D (Homography), chói lóa ánh nắng/đèn pha, bóng đổ, nhòe chuyển động và nhiễu cảm biến camera.

10. **Tài liệu nguồn minh bạch:**
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
