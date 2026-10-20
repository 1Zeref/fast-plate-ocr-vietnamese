# Hướng Dẫn Thiết Lập & Vận Hành Môi Trường Thực Nghiệm Docker

Tài liệu hướng dẫn triển khai, đóng gói và kiểm chuẩn toàn diện hệ thống **Fast-Plate-OCR-Vietnamese** trên nền tảng Docker và Docker Compose (hỗ trợ cả GPU NVIDIA qua CUDA 12.1 và CPU đa luồng).

---

## 1. Kiến Trúc Môi Trường Container

- **Base Image:** `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04`
- **Ngôn ngữ & Runtime:** Python 3.10, PyTorch 2.2.0 (CUDA 12.1), ONNX Runtime GPU 1.17+, OpenCV Headless 4.8+.
- **Hệ điều hành cô lập:** Ubuntu 22.04 LTS (đã cài đặt đầy đủ `libgl1`, `ffmpeg`, `libglib2.0-0`, `fontconfig`).
- **Cấu trúc Thư mục Mount:**
  - `./samples:/app/samples:ro`: Thư mục chứa ảnh và video đầu vào ngoài máy host (chế độ chỉ đọc).
  - `./output:/app/output:rw`: Thư mục lưu trữ kết quả phân tích (ảnh vẽ bounding box, video tracklet, file log JSON).
  - `./docs:/app/docs:rw`: Tài liệu kỹ thuật dự án.

---

## 2. Yêu Cầu Tiên Quyết (Prerequisites)

1. **Máy tính cài đặt Docker Desktop** (Windows / macOS) hoặc **Docker Engine** (Linux Ubuntu/Debian).
2. **Docker Compose v2.0+** (`docker compose version`).
3. *(Tùy chọn cho chế độ GPU)*: Card đồ họa NVIDIA và **NVIDIA Container Toolkit** ([Hướng dẫn cài đặt chính thức](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)).

---

## 3. Hướng Dẫn Vận Hành Từng Bước

### Bước 1: Chuẩn Bị Dữ Liệu Test Vào Thư Mục `samples/`

Hệ thống đã chuẩn bị sẵn 3 mẫu thử nghiệm tiêu biểu trong thư mục `samples/`:
- `samples/car_1line_sample.jpg`: Biển số ô tô 1 dòng dài tiêu chuẩn.
- `samples/motorcycle_2line_sample.jpg`: Biển số xe máy 2 dòng vuông.
- `samples/traffic_scene_sample.jpg`: Khung hình giao thông toàn cảnh 1080p độ phân giải cao.

> Bạn có thể sao chép thêm bất kỳ tệp ảnh (`.jpg`, `.png`) hoặc video (`.mp4`, `.avi`) nào vào thư mục `samples/` ngoài máy host.

---

### Bước 2: Xây Dựng Image Docker (Build Image)

#### Chế độ GPU (NVIDIA CUDA 12.1):
```bash
docker compose build
```

#### Chế độ CPU (Máy không có GPU NVIDIA):
```bash
docker compose -f docker-compose.cpu.yml build
```

*Quá trình build sẽ tải base image CUDA, cài đặt các gói phụ thuộc hệ thống và tối ưu hóa layer caching. Image xuất ra có tag: `fast-plate-ocr-vietnamese:latest`.*

---

### Bước 3: Khởi Động Container & Chạy Thực Nghiệm

#### Khởi chạy nền (Detached mode):
```bash
# Đối với máy có GPU NVIDIA:
docker compose up -d

# Đối với máy chỉ có CPU:
docker compose -f docker-compose.cpu.yml up -d
```

#### Theo dõi Log Khởi Động & Chẩn Đoán Phần Cứng:
```bash
docker compose logs -f
```

Container sẽ tự động:
1. Kiểm tra driver `nvidia-smi` và xác nhận kết nối CUDA của PyTorch (`torch.cuda.is_available()`).
2. Quét toàn bộ tệp trong `/app/samples/`.
3. Chạy pipeline nhận dạng biển số (Module Nắn phẳng $\to$ CCT OCR $\to$ DFA Beam Search).
4. Xuất ảnh đã vẽ bounding box và biển số sang `/app/output/`.
5. Tạo tệp tổng kết `output/evaluation_summary.json`.

---

### Bước 4: Kiểm Chuẩn Khởi Động & Chạy Regression Test

Để xác nhận 100% module hoạt động bình thường bên trong container, chạy lệnh sau từ máy host:

```bash
docker compose exec alpr_vietnam_eval python -m unittest discover tests -p "test_*.py" -v
```

Hoặc chạy kiểm thử bằng `pytest`:
```bash
docker compose exec alpr_vietnam_eval pytest tests/ -v
```

> **Tiêu chuẩn đạt:** Toàn bộ **26/26 unit tests** đều hiển thị `OK` (bao gồm DFA grammar, Keypoint Detector, Homography Warp, CCT OCR, ByteTrack, Temporal Consensus, và ONNX Runtime benchmark).

---

### Bước 5: Kiểm Tra Kết Quả Xuất Ra Trên Máy Host

Sau khi chạy xong, hãy kiểm tra thư mục `output/` trên máy host của bạn:
- `output/annotated_car_1line_sample.jpg`: Ảnh biển xe ô tô với khung nhận dạng.
- `output/annotated_motorcycle_2line_sample.jpg`: Ảnh biển xe máy đã được nắn phẳng và ghép dòng.
- `output/annotated_traffic_scene_sample.jpg`: Ảnh giao thông toàn cảnh đã khoanh vùng biển số.
- `output/evaluation_summary.json`: Báo cáo chi tiết độ trễ (ms), chuỗi ký tự nhận diện, độ tin cậy và cú pháp DFA.

---

### Bước 6: Truy Cập Trực Tiếp Vào Terminal Của Container (Interactive Shell)

Nếu bạn muốn debug hoặc chạy lệnh thủ công bên trong môi trường container:

```bash
docker compose exec -it alpr_vietnam_eval bash
```

Bên trong container, bạn có thể chạy:
```bash
# Chạy demo với các tham số tùy chỉnh:
python demo.py --samples /app/samples --output /app/output --use-onnx --conf-threshold 0.25

# Chạy đo lường độ trễ chi tiết:
python -m src.deploy.benchmark

# Chạy kiểm chuẩn toàn diện hệ thống:
python -m src.deploy.validation
```

---

## 4. Dừng & Dọn Dẹp Container

Khi hoàn tất quá trình thử nghiệm:
```bash
# Dừng container
docker compose down

# Dừng và xóa toàn bộ volume (nếu cần dọn sạch)
docker compose down -v
```

---

## 5. Bảng Tra Cứu Lệnh Nhanh (Cheat Sheet)

| Tác vụ | Lệnh Thực Thi |
| :--- | :--- |
| **Build Image (GPU)** | `docker compose build` |
| **Build Image (CPU)** | `docker compose -f docker-compose.cpu.yml build` |
| **Khởi động nền** | `docker compose up -d` |
| **Xem log realtime** | `docker compose logs -f` |
| **Chạy Test Suite** | `docker compose exec alpr_vietnam_eval pytest tests/` |
| **Vào Shell Container** | `docker compose exec -it alpr_vietnam_eval bash` |
| **Chạy lại Demo** | `docker compose exec alpr_vietnam_eval python demo.py` |
| **Dừng hệ thống** | `docker compose down` |
