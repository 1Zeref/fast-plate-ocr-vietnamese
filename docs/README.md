# FAST-PLATE-OCR-VIETNAMESE: TÀI LIỆU NGUỒN VÀ QUY CHUẨN KỸ THUẬT

Dự án nghiên cứu, tối ưu hóa và huấn luyện mô hình nhận dạng ký tự quang học biển số xe Việt Nam (Optical Character Recognition - OCR) tốc độ cao dựa trên kiến trúc **Compact Convolutional Transformer (CCT)** và mạng nơ-ron tích chập đa đầu ra.

---

## 1. Nguồn Mã Nguồn Gốc & Kiến Trúc Mô Hình (Original Source Code)

- **Dự án tham khảo:** [fast-plate-ocr](https://github.com/ankandrew/fast-plate-ocr)
- **Tác giả:** Andrés ([ankandrew](https://github.com/ankandrew))
- **Mô tả kiến trúc:**
  - Mô hình nhận diện ký tự dựa trên kiến trúc nhẹ CCT (Compact Convolutional Transformer) hoặc CNN Backbone kết hợp nhiều đầu phân loại (multi-head classification) tương ứng với từng slot ký tự trên biển số.
  - Hỗ trợ tối ưu hóa ONNX Runtime, TensorRT và OpenVINO cho tốc độ suy luận dưới $2\text{ms}$ trên CPU/GPU.
  - Sử dụng tệp cấu hình `vocab.yaml` và `plate_config.yaml` định nghĩa `alphabet`, `pad_char`, `max_plate_slots`, `img_height`, `img_width`.

---

## 2. Nguồn Dữ Liệu Thực Tế (Dataset Sources)

### 2.1. Dataset Kaggle: Vietnamese License Plate OCR
- **Nguồn:** [Kaggle - vietnamese-license-plate-ocr](https://www.kaggle.com/datasets/topkek69/vietnamese-license-plate-ocr)
- **Tác giả:** [topkek69](https://www.kaggle.com/topkek69)
- **Phương thức tải tự động:** Thông qua thư viện `kagglehub`:
  ```python
  import kagglehub
  path = kagglehub.dataset_download("topkek69/vietnamese-license-plate-ocr")
  ```
- **Quy mô tập dữ liệu gốc:**
  - Thư mục `cropped/`: 6,643 ảnh biển số thực tế cắt từ camera giao thông.
  - Thư mục `generated/`: 5,547 ảnh biển số tổng hợp đa dạng điều kiện.
  - Tổng số mẫu gốc: **12,190 ảnh**.
  - File nhãn gốc: `crop_labels.csv` và `gen_labels.csv` gồm các trường `Name`, `Label`, `Type`.
- **Quy trình làm sạch & Chuẩn hóa qua DFA:**
  - Áp dụng bộ lọc ngữ pháp pháp lý (Máy trạng thái hữu hạn DFA - Bước 1.1) tại `src/data/prepare_kaggle.py`.
  - Tự động khắc phục các lỗi typo gán nhãn phổ biến (nhầm chữ `O` thành số `0`, chữ `I` thành số `1`).
  - Loại bỏ **571 mẫu** lỗi gán nhãn sai hoặc ảnh bị crop cụt mất mã tỉnh/sê-ri.
  - Tổng số mẫu đạt chuẩn pháp lý: **11,619 mẫu (95.32%)**.
  - Phân chia tập dữ liệu chuẩn:
    - **Tập Train (85%):** 9,877 mẫu (`data/kaggle_dataset/train.csv`).
    - **Tập Val (15%):** 1,742 mẫu (`data/kaggle_dataset/val.csv`).

### 2.2. Động cơ Sinh Dữ liệu Giả lập (Synthetic Data Engine)
- Thiết kế tại `src/synthetic/` hỗ trợ sinh tự động từ $300,000 - 500,000$ mẫu ảnh bổ trợ với đầy đủ nhiễu vật lý thực địa (phay kim loại, xước cơ học, đốm bùn, xoay 3D Homography, chói lóa, bóng đổ, nhòe chuyển động).

### 2.3. Dữ Liệu Video Thực Địa, Trích Xuất & Gán Nhãn Đa Điểm (Bước 1.3)
- **Nguồn & Kịch bản thực tế:**
  - Hệ thống camera giám sát giao thông đường phố (ngã tư, làn cao tốc) và camera kiểm soát xe thông minh tại các bãi đỗ xe (GreenParking, Smart Parking Gates).
  - Bao phủ 5 kịch bản môi trường khắc nghiệt:
    1. `SUNNY_DAYLIGHT`: Nắng gắt chói chang, độ tương phản cực đại, bóng đen đổ xiên.
    2. `NIGHT_BACKLIGHT`: Ban đêm ngược sáng mạnh do đèn pha xe chiếu trực diện vào ống kính camera.
    3. `RAINY_WET`: Trời mưa, bề mặt đường ướt phản quang khuếch tán, giảm độ tương phản viền biển.
    4. `OBLIQUE_ANGLE_30_45`: Góc đặt camera xiên từ $30^\circ - 45^\circ$ (camera gắn trên cột đèn cao vỉa hè hoặc góc tường bãi xe).
    5. `PARKING_GATE`: Camera barrier bãi đỗ xe góc cao $20^\circ - 30^\circ$, cự ly gần ($1.5 - 3\text{m}$).
- **Cơ chế Lọc Chọn Khung Hình Thông Minh (`src/data/video_sampler.py`):**
  - **Lọc khung hình trùng lặp (Frame Differencing):** Tránh lưu các frame dư thừa khi phương tiện dừng đèn đỏ hoặc chờ barrier mở.
  - **Lọc nhòe mờ chuyển động (Laplacian Blur Filter):** Tự động tính phương sai toán tử vi phân bậc hai Laplacian $\text{Var}(\Delta I)$, loại bỏ toàn bộ khung hình có điểm dưới ngưỡng $\text{Threshold} = 85.0$.
  - **Phân loại kịch bản tự động:** Phân tích biểu đồ phân bố độ sáng và tỷ lệ bão hòa pixel để tự động gắn thẻ môi trường.
- **Đặc Tả Nhãn Chuẩn 5 Trường (`src/data/annotator.py`):**
  - `vehicle_bbox`: Tọa độ hộp bao phương tiện `[vx1, vy1, vx2, vy2]`.
  - `plate_bbox`: Tọa độ hộp bao biển số `[px1, py1, px2, py2]`.
  - `corners`: Tọa độ 4 điểm góc $(x_1, y_1), (x_2, y_2), (x_3, y_3), (x_4, y_4)$ theo thứ tự Top-Left, Top-Right, Bottom-Right, Bottom-Left.
  - `text`: Chuỗi ký tự biển số ground truth (được kiểm định bởi DFA Bước 1.1).
  - `layout`: Cờ phân loại hình học (`1_LINE` / `2_LINE`).
- **Thuật toán Nắn Phẳng Xạ Ảnh 4 Điểm (Perspective Rectification):**
  - Hàm `rectify_plate_quad` sử dụng ma trận biến đổi Homography $3 \times 3$ để chuyển đổi biển số góc xiên $30^\circ - 45^\circ$ về kích thước chuẩn chữ nhật ($520 \times 120$ cho biển dài, $340 \times 200$ cho biển vuông) trước khi đưa vào mô hình OCR.
- **Định Dạng Xuất Khẩu:**
  - `annotations_full.json`: Cấu trúc JSON phân cấp đầy đủ thông tin phương tiện, biển số, tọa độ 4 góc và kịch bản.
  - `ocr_annotations.csv`: Tệp CSV phẳng tương thích trực tiếp pipeline huấn luyện mô hình OCR.
  - `yolo_obb_labels/`: Thư mục nhãn định dạng YOLO Oriented Bounding Box (`class x1 y1 x2 y2 x3 y3 x4 y4`) cho bài toán phát hiện biển nghiêng.

---

---

## 3. Module Định Vị Phương Tiện & 4 Điểm Góc Biển Số (Bước 2.1)

- **Mục tiêu:** Định vị chính xác phương tiện và trích xuất tứ giác lồi của biển số trong ảnh độ phân giải cao bằng mạng nơ-ron Anchor-Free Keypoint Detector.
- **Kiến trúc mạng PlateCornerDetector (src/detector/model.py):**
  - **Backbone:** Khối tích chập nhẹ (Lightweight Backbone) sử dụng Depthwise Separable Convolutions và SiLU activation giúp tối ưu hóa số lượng tham số và tăng tốc độ suy luận.
  - **Neck (FPN):** Feature Pyramid Network hợp nhất đặc trưng nông (giữ độ phân giải cao cho 4 góc) và đặc trưng sâu ngữ nghĩa tại tỷ lệ /8$.
  - **Decoupled Prediction Heads:**
    1. conf_head: Bản đồ xác suất tâm biển số (Confidence map trong dải 0..1).
    2. box_head: Hồi quy độ lệch và kích thước hộp bao [dx, dy, w, h].
    3. corner_head: Hồi quy vector độ dời của 4 điểm góc [dx1, dy1, ..., dx4, dy4] so với tâm biển số.
    4. layout_head: Phân loại cấu trúc biển số (0: 1 dòng, 1: 2 dòng).
- **Hàm mất mát Đa Tác Vụ (src/detector/loss.py):**
  - **Wing Loss:** Hàm mất mát tiêu chuẩn chuyên dụng cho Landmark / Keypoint Regression (*Feng et al., CVPR*), có độ dốc cao khi sai số nhỏ giúp 4 điểm góc bám chặt viền mép biển số chính xác đến từng pixel.
- **Trọng số & Hiệu năng (weights/best_detector.pt):**
  - Dung lượng file trọng số: **8.3 MB** (cực kỳ nhẹ, tối ưu triển khai thiết bị biên / Edge AI, CPU, Jetson Nano).
  - Tốc độ huấn luyện: 11.43 giây trên CPU tiêu chuẩn (đạt Loss = 2.44).
  - Bộ suy luận (src/detector/infer.py): Tích hợp Non-Maximum Suppression (NMS) và tự động nắn phẳng xạ ảnh (
ectify_plate_quad) trả về ảnh biển số thẳng thớm phục vụ nhận diện OCR.

---

---

## 4. Module Nắn Phối Cảnh & Định Tuyến Hình Học (Bước 2.2)

- **Mục tiêu:** Khử hoàn toàn độ méo góc nghiêng của camera và phân luồng biển số, nối không gian biển 2 dòng thành dải 1 dòng chuẩn tắc.
- **Thuật toán Sắp xếp 4 Góc & Nắn Phối Cảnh (src/rectification/homography.py):**
  - Hàm order_quad_corners sắp xếp 4 điểm góc bất kỳ theo thứ tự Top-Left, Top-Right, Bottom-Right, Bottom-Left dựa trên cực trị tổng $ và hiệu $.
  - Tính ma trận Homography  \in \mathbb{R}^{3 \times 3}$ bằng cv2.getPerspectiveTransform và nắn phẳng bằng phép nội suy song tuyến (cv2.warpPerspective).
  - Hệ trục chuẩn tắc:  \times 128$ (biển dài) và  \times 240$ (biển vuông).
- **Bộ Định Tuyến Hình Học (src/rectification/router.py):**
  - Phân loại hình học: 1-line (biển dài) hoặc 2-line (biển vuông).
  - Kết hợp Ensemble giữa tỷ lệ cạnh hình học (Aspect Ratio /H$ với ngưỡng .40$) và xác suất từ mạng học sâu PlateCornerDetector (Bước 2.1).
- **Thuật toán Tách Dòng & Virtual 1-Line Strip (src/rectification/splitter.py):**
  - **Horizontal Projection Profile (Biểu đồ chiếu ngang):** Tính tổng số pixel ký tự theo từng hàng ngang (y) = \sum_x I_{\text{bin}}(x, y)$, tìm điểm thung lũng (local minimum) giữa hai dòng để xác định đường cắt {\text{split}}$.
  - **Đồng bộ chiều cao:** Cắt tách dòng 1 và dòng 2, chuẩn hóa về cùng chiều cao \text{ px}$ trong khi giữ nguyên tỷ lệ chiều rộng ký tự.
  - **Nối không gian (Horizontal Concatenation):** Ghép hai dòng theo chiều ngang kèm khoảng đệm phân cách màu nền (separator padding \text{ px}$):
    \text{Virtual 1-Line Strip} = [\text{Line 1} \quad | \quad \text{Separator} \quad | \quad \text{Line 2}]
  - Chuẩn hóa toàn bộ dải ảnh về kích thước  \times 64\text{ px}$ tương thích trực tiếp với kiến trúc nhận dạng OCR của fast-plate-ocr.
- **Pipeline Hoàn Chỉnh (src/rectification/pipeline.py):**
  - Tích hợp toàn diện: Tọa độ 4 góc $\to$ Nắn phẳng $\to$ Router $\to$ Virtual Strip $\to$ Tensor OCR.

---

## 5. Nguồn Văn Bản Quy Chuẩn Pháp Lý Việt Nam (Legal & Regulatory References)

Toàn bộ không gian ngữ pháp, bảng ký tự và quy tắc sê-ri được xây dựng nghiêm ngặt theo các văn bản quy phạm pháp luật hiện hành:

1. **Thông tư số 24/2023/TT-BCA** (Bộ Công an, ban hành ngày 01/07/2023, có hiệu lực từ 15/08/2023):
   - Quy định về cấp, thu hồi đăng ký, biển số xe cơ giới.
   - Quy định biển số định danh theo mã định danh cá nhân / CCCD.
   - Điều 37 & Phụ lục số 02: Quy định dải ký hiệu mã tỉnh thành từ 11 đến 99 (loại trừ các số chưa cấp: 13, 41, 42, 44, 45, 46, 87, 91, 96).
   - Quy định 20 chữ cái dân sự tiêu chuẩn: `A, B, C, D, E, F, G, H, K, L, M, N, P, S, T, U, V, X, Y, Z` (loại trừ `I, O, J, W`).
   - Mở rộng sê-ri 2 chữ cái liên tiếp cho ô tô và mô tô.

2. **Thông tư số 58/2020/TT-BCA** (Bộ Công an, ban hành ngày 16/06/2020):
   - Quy định về quy trình cấp, thu hồi đăng ký, biển số phương tiện giao thông cơ giới đường bộ.
   - Quy cách kích thước biển số:
     - Biển số dài: Chiều cao 110 mm, chiều dài 520 mm.
     - Biển số ngắn (vuông): Chiều cao 165 mm, chiều dài 330 mm.
   - Quy định bố cục dòng 1 và dòng 2 đối với biển số xe máy và ô tô biển vuông.

3. **Thông tư số 169/2021/TT-BQP** (Bộ Quốc phòng, ban hành ngày 23/12/2021):
   - Quy định về đăng ký, quản lý, sử dụng xe cơ giới, xe máy chuyên dùng trong Bộ Quốc phòng.
   - Phụ lục III: Danh mục ký hiệu chữ cái viết tắt của các cơ quan, đơn vị quân sự (Biển đỏ quân đội):
     - Cơ quan BQP & Tổng cục: `TM, TC, TH, TT, TK, TN`.
     - Quân khu, Quân chủng, Bộ Tư lệnh: `KA, KB, KC, KD, KV, KP, KK, KT, QA, QH, QB, QC, QM, BL`.
     - Binh chủng: `BB, BC, BK, BP, BH, BT`.
     - Quân đoàn & Học viện: `AA, AB, AC, AD, HA, HB, HC, HE, HD, HH, HT`.
     - Doanh nghiệp quân đội & Binh đoàn: `VT (Viettel), CB (MB Bank), CA, CD, CH, CM, CN, CP, CT, CV, AV, AT, AN, AX, AM`.

4. **Quy định biển số Ngoại giao & Tổ chức Quốc tế:**
   - Sê-ri `NG`: Đại sứ quán, viên chức ngoại giao.
   - Sê-ri `QT`: Tổ chức quốc tế (UN, WHO, v.v.).
   - Sê-ri `NN`: Tổ chức, cá nhân nước ngoài.
   - Sê-ri `CV`: Nhân viên hành chính kỹ thuật cơ quan ngoại giao.

---

## 6. Nguồn Phông Chữ & Typography Chuẩn

1. **Phông chữ nét vuông công nghiệp mới (Modern Square Font):**
   - Tiêu chuẩn: DIN 1451 Mittelschrift (áp dụng trong font hệ thống Windows `bahnschrift.ttf`).
   - Bản sao chép cục bộ: `assets/fonts/plate_square_modern.ttf`.
   - Đặc điểm: Đường nét đều đặn, góc bo bán kính hẹp, khoảng cách ký tự đồng nhất, tương thích quy chuẩn dập khuôn Thông tư 58/2020 và 24/2023.

2. **Phông chữ nét tròn truyền thống (Classic Round Font):**
   - Tiêu chuẩn: Arial Bold (`arialbd.ttf`) & Arial Narrow Bold (`ARIALNB.TTF`).
   - Bản sao chép cục bộ: `assets/fonts/plate_round_bold.ttf`, `assets/fonts/plate_round_narrow.ttf`.
   - Đặc điểm: Đường cong bo tròn rộng, phù hợp mô phỏng các thế hệ biển số 4 số cũ và biển 5 số giai đoạn trước 2020.

---

## 7. Cấu Trúc Dự Án (Repository Structure)

```text
fast-plate-ocr-vietnamese/
├── configs/
│   ├── vocab.yaml                    # Cấu hình từ vựng 33 tokens (0-9, A-Z trừ I,O,J,W và pad '_')
│   └── grammar_schema.yaml           # Bản đồ ngữ pháp, mã tỉnh, mã quân sự, quy tắc DFA
├── src/
│   ├── rectification/                # Module nắn phối cảnh & định tuyến (Bước 2.2)
│   │   ├── homography.py             # Sắp xếp 4 góc & nắn phẳng ma trận Homography
│   │   ├── router.py                 # Định tuyến hình học (1-line vs 2-line)
│   │   ├── splitter.py               # Phân tích chiếu ngang & sinh Virtual 1-Line Strip
│   │   └── pipeline.py               # Pipeline nắn phối cảnh & nạp tensor OCR
├── weights/
│   └── best_detector.pt              # Trọng số mô hình Detector 4 góc biển số (8.3 MB)
├── src/
│   ├── detector/                     # Mạng định vị xe & trích xuất 4 điểm góc (Bước 2.1)
│   │   ├── model.py                  # Kiến trúc Anchor-Free Keypoint Detector
│   │   ├── dataset.py                # DataLoader nạp ảnh và chuẩn hóa tọa độ 4 góc
│   │   ├── loss.py                   # Multi-task Loss (Wing Loss, Focal Loss, Smooth L1)
│   │   ├── infer.py                  # Engine suy luận, NMS & nắn phẳng biển số
│   │   └── train.py                  # Script huấn luyện & xuất ONNX
│   ├── grammar/                      # Động cơ máy trạng thái hữu hạn DFA
│   │   ├── constants.py              # Danh mục mã tỉnh 11-99, mã quân sự, sê-ri
│   │   ├── dfa.py                    # DFA validator xác thực và phân loại biển số
│   │   └── __init__.py
│   ├── data/                         # Xử lý và chuẩn hóa dữ liệu
│   │   ├── prepare_kaggle.py         # Pipeline nạp dataset Kaggle topkek69, lọc DFA, tách train/val
│   │   └── __init__.py
│   └── synthetic/                    # Động cơ sinh dữ liệu giả lập (Synthetic Engine)
│       ├── generator.py              # Sinh chuỗi ngẫu nhiên theo luật
│       ├── renderer.py               # Kết xuất phôi biển và đinh ốc
│       ├── textures.py               # Sinh texture kim loại, trầy xước, bùn đất
│       ├── augmentor.py              # Xoay 3D Homography, chói lóa, nhòe, nhiễu
│       ├── engine.py                 # Động cơ sinh dữ liệu đa tiến trình
│       └── __init__.py
├── assets/
│   └── fonts/                        # Bộ phông chữ chuẩn biển số Việt Nam
├── data/
│   └── kaggle_dataset/               # Dữ liệu Kaggle đã làm sạch (train.csv, val.csv, annotations.csv)
├── tests/
│   ├── test_grammar.py               # Unit test toàn diện không gian ngữ pháp
│   └── test_synthetic_consistency.py # Test kiểm tra chéo tính đồng bộ giữa Synthetic và DFA
└── docs/
    └── README.md                     # Tài liệu tham khảo và nguồn dữ liệu chi tiết
```
