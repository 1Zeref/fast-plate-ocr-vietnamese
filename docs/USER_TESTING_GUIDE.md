# Cẩm Nang Thử Nghiệm & Đánh Giá Thực Nghiệm Hệ Thống (WebUI Testing Guide)

Tài liệu hướng dẫn chi tiết dành cho người dùng cuối, đội ngũ kiểm thử (QA) và chuyên gia R&D thao tác trực quan với hệ sinh thái **Fast-Plate-OCR-Vietnamese** thông qua giao diện WebUI tương tác đa chức năng.

---

## 1. Khởi Động Giao Diện WebUI

### Cách 1: Khởi Chạy Nhanh Bằng Docker Compose (Khuyến Nghị)

```bash
# Đối với máy chủ / máy trạm có card NVIDIA GPU:
docker compose up -d

# Đối với máy tính thông thường (CPU):
docker compose -f docker-compose.cpu.yml up -d
```

Sau khi khởi chạy, mở trình duyệt web và truy cập địa chỉ:  
👉 **`http://localhost:7860`**

---

### Cách 2: Khởi Chạy Cục Bộ (Local Python Environment)

```bash
# Kích hoạt môi trường và chạy trực tiếp:
python app_webui.py --host 0.0.0.0 --port 7860
```

Nếu muốn chia sẻ giao diện ra ngoài Internet cho đối tác hoặc đồng nghiệp kiểm tra từ xa, thêm cờ `--share`:
```bash
python app_webui.py --share
```

---

## 2. Hướng Dẫn Sử Dụng Chi Tiết 4 Tab Chức Năng

```
┌────────────────────────────────────────────────────────────────────────┐
│                        ALPR VIETNAM WEBUI                              │
├─────────────────┬──────────────────┬─────────────────┬─────────────────┤
│ 1. Ảnh tĩnh     │ 2. Video File    │ 3. Camera RTSP  │ 4. Bóc tách Kỹ  │
│    (Single/     │    (Batch &      │    (Live        │    thuật        │
│     Batch Image)│     Tracking)    │     Realtime)   │    (Inspection) │
└─────────────────┴──────────────────┴─────────────────┴─────────────────┘
```

---

### Tab 1: Kiểm Thử Ảnh Tĩnh (Image Mode)

- **Mục đích:** Đánh giá nhanh khả năng nhận dạng trên từng ảnh đơn lẻ hoặc ảnh chụp toàn cảnh camera 1080p.
- **Thao tác thực hiện:**
  1. Kéo thả ảnh biển số từ máy tính vào ô **Ảnh Đầu Vào**, hoặc nhấp vào một trong các ảnh mẫu ở mục **📁 Ảnh Thử Nghiệm Mẫu**:
     - `car_1line_sample.jpg`: Biển số ô tô dài 1 dòng tiêu chuẩn.
     - `motorcycle_2line_sample.jpg`: Biển số xe máy vuông 2 dòng.
     - `traffic_scene_sample.jpg`: Khung hình giao thông toàn cảnh độ phân giải cao.
  2. Tùy chỉnh tham số nhận diện (nếu muốn thử nghiệm các kịch bản khác nhau):
     - **Detection Confidence Threshold (0.10 - 0.95):** Ngưỡng tin cậy phát hiện vị trí biển số (mặc định: `0.25`). Giảm ngưỡng nếu biển số bị mờ hoặc quá xa.
     - **Beam Width (1 - 10):** Số lượng nhánh tìm kiếm trong giải mã Beam Search (mặc định: `5`).
     - **Kích hoạt Ràng buộc Ngữ pháp DFA:** Bật để áp dụng luật biển số Việt Nam (Thông tư 24/2023/TT-BCA); Tắt để xem kết quả quang học thô (Greedy).
  3. Bấm nút **🔍 Nhận Diện Biển Số**.
- **Cách đọc kết quả:**
  - **Ảnh Kết Quả Chú Thích:** Hiển thị khung bao viền xanh, 4 điểm góc và nhãn chuỗi biển số kèm độ tin cậy.
  - **Bảng Kết Quả Chi Tiết:**
    - *Biển số:* Chuỗi text đã được chuẩn hóa.
    - *Bố cục:* `1 Dòng (Dài)` hoặc `2 Dòng (Vuông)`.
    - *Địa phương:* Tên tỉnh/thành phố đăng ký (ví dụ: Hà Nội, TP.HCM, Thái Nguyên).
    - *Phân loại:* `CIVILIAN_CAR` (ô tô con), `MOTORCYCLE` (xe máy), `MILITARY` (quân sự), v.v.
    - *Hợp lệ DFA:* Trạng thái kiểm chuẩn pháp lý `✅ Hợp lệ` hoặc `❌ Vi phạm cú pháp`.
    - *Thời gian xử lý:* Độ trễ suy luận toàn trình tính bằng mili-giây (thường từ $10 - 15\text{ ms}$).

---

### Tab 2: Kiểm Thử Video & Bám Vết (Video Tracking Mode)

- **Mục đích:** Đánh giá độ ổn định của thuật toán bám vết ByteTrack và cơ chế triệt tiêu nhấp nháy, lóa sáng đa khung hình (Temporal Consensus).
- **Thao tác thực hiện:**
  1. Tải lên tệp video giám sát giao thông định dạng `.mp4` hoặc `.avi`.
  2. Điều chỉnh tham số:
     - **Frame Skip (1 - 4):** Xử lý cách khung hình (mặc định `1` là xử lý từng frame; chọn `2` hoặc `3` để tăng tốc trên CPU yếu).
     - **Cửa sổ Đồng thuận (Consensus K Frames):** Số lượng khung hình rõ nét nhất được chọn lọc vào Best-K buffer để tiến hành bỏ phiếu có trọng số (mặc định `5` frames).
  3. Bấm nút **▶ Bắt Đầu Xử Lý Video**.
- **Cách đọc kết quả:**
  - Ô **Streaming Preview** sẽ hiển thị luồng khung hình được xử lý trực tiếp kèm **Track ID** cố định cho từng xe.
  - Bảng **Nhật Ký Phương Tiện Giao Thông (Vehicle Traffic Log)** sẽ tự động cập nhật theo thời gian thực mỗi khi có phương tiện mới xuất hiện và chốt kết quả biển số đồng thuận cuối cùng.

---

### Tab 3: Giám Sát Camera Trực Tiếp (Live RTSP / Webcam)

- **Mục đích:** Thử nghiệm triển khai camera thực địa theo thời gian thực (Realtime Surveillance).
- **Chế độ 1: Webcam Máy Tính:**
  - Bật webcam trên giao diện trình duyệt, hệ thống sẽ tự động stream từng khung hình, nhận diện biển số tức thì và hiển thị FPS thời gian thực (~35-45 FPS trên GPU, ~25-30 FPS trên CPU).
- **Chế độ 2: Camera IP (Luồng RTSP):**
  - Nhập địa chỉ luồng RTSP camera giao thông theo định dạng chuẩn:  
    `rtsp://<username>:<password>@<ip_address>:<port>/<stream_path>`
  - Bấm **📡 Kết Nối & Kiểm Tra Frame RTSP** để kiểm tra tính thông suốt của đường truyền.

---

### Tab 4: Bóc Tách Chẩn Đoán Kỹ Thuật (Explainable AI Inspector)

- **Mục đích:** Quan sát chi tiết "bên trong hộp đen" của mô hình AI, phục vụ việc phân tích nguyên nhân gốc rễ (Root Cause Analysis) khi gặp biển số lạ hoặc bị đọc sai.
- **Thao tác thực hiện:**
  1. Chọn hoặc tải ảnh cần kiểm tra vào ô **Ảnh Cần Chẩn Đoán**.
  2. Bấm nút **🔬 Bóc Tách Toàn Bộ Công Đoạn**.
- **Chuỗi 4 Bước Chuyển Đổi Hình Học (Visual Pipeline):**
  1. **[1] Vùng Crop Xe / Biển Gốc:** Bounding box được Detector phát hiện.
  2. **[2] Tứ Giác 4 Điểm Góc:** Đánh dấu trực quan 4 đỉnh góc lồi với màu sắc chuẩn hóa:
     - Đỏ: Góc trên - trái (Top-Left)
     - Xanh lá: Góc trên - phải (Top-Right)
     - Xanh dương: Góc dưới - phải (Bottom-Right)
     - Tím: Góc dưới - trái (Bottom-Left)
  3. **[3] Nắn Phẳng Homography:** Ảnh biển số đã được triệt tiêu hoàn toàn góc nghiêng 3D, duỗi thẳng về kích thước chuẩn tắc.
  4. **[4] Dải Ảnh Ghép Dòng Ảo (Virtual 1-Line Strip):** Nếu là biển vuông 2 dòng, thuật toán tự động tách dòng trên và dòng dưới rồi nối ngang thành dải ảnh duy nhất $64 \times 256$ pixel.
- **Báo Cáo Giải Thích Tự Động (Explainable AI Report):**
  - So sánh trực tiếp giữa chuỗi nhận dạng quang học ban đầu (**Greedy Raw**) và chuỗi sau khi áp mặt nạ ngữ pháp (**DFA Constrained**).
  - Liệt kê chi tiết từng vị trí ký tự nhầm lẫn quang học đã được nắn chỉnh tự động (ví dụ: phát hiện lỗi `$8 \to B$` tại vị trí sê-ri hoặc `$B \to 8$` tại vị trí dãy số).

---

## 3. Câu Hỏi Thường Gặp & Xử Lý Sự Cố (Troubleshooting FAQ)

| Vấn đề gặp phải | Nguyên nhân có thể | Hướng xử lý |
| :--- | :--- | :--- |
| **Không truy cập được `http://localhost:7860`** | Container chưa chạy hoặc cổng 7860 bị chiếm dụng | Kiểm tra lệnh `docker compose ps`. Nếu xung đột cổng, đổi `7861:7860` trong file `docker-compose.yml`. |
| **Ảnh tải lên không phát hiện được biển số** | Ngưỡng `Detection Confidence` đặt quá cao | Kéo thanh trượt `Confidence Threshold` xuống mức `0.15 - 0.20`. |
| **Biển số bị lóa đèn hoặc che khuất ốc vít** | Nhiễu quang học tức thời | Sử dụng **Tab 2 (Video Tracking)** để cơ chế `Temporal Consensus` tự động loại bỏ frame lóa và khôi phục ký tự chính xác. |
| **Báo lỗi kết nối RTSP Camera** | Sai mật khẩu, sai port hoặc tường lửa (Firewall) chặn port 554 | Dùng phần mềm VLC Media Player để kiểm tra luồng RTSP trước khi đưa vào WebUI. |
