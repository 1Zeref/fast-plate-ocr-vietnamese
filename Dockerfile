# ==============================================================================
# FAST-PLATE-OCR-VIETNAMESE: Production Dockerfile
# NVIDIA CUDA 12.1 + cuDNN 8 + Python 3.10 + OpenCV + ONNX Runtime GPU / PyTorch
# ==============================================================================

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

LABEL maintainer="1Zeref <github.com/1Zeref>"
LABEL description="Fast Plate OCR for Vietnamese License Plates - Commercial Grade ALPR Pipeline"

# Thiết lập biến môi trường hệ thống
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

# Cài đặt các gói phụ thuộc cấp hệ thống cho xử lý ảnh, video và font chữ
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    libxrender-dev \
    libxext6 \
    libsm6 \
    fontconfig \
    curl \
    git \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Thiết lập thư mục làm việc chuẩn
WORKDIR /app

# Sao chép và cài đặt các gói phụ thuộc Python
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir onnxruntime-gpu && \
    pip cache purge

# Sao chép cấu trúc mã nguồn, cấu hình, font và trọng số vào container
COPY configs/ /app/configs/
COPY assets/ /app/assets/
COPY weights/ /app/weights/
COPY src/ /app/src/
COPY tests/ /app/tests/
COPY scripts/ /app/scripts/
COPY demo.py /app/demo.py
COPY app_webui.py /app/app_webui.py

# Khởi tạo các thư mục kết nối mount điểm kiểm chuẩn
RUN mkdir -p /app/samples /app/output /app/docs && \
    chmod +x /app/scripts/*.sh || true

# Biến môi trường ứng dụng
ENV PYTHONPATH=/app \
    USE_ONNX=true

# Cổng dịch vụ WebUI Gradio và API
EXPOSE 7860 8000

# Kịch bản khởi động mặc định
ENTRYPOINT ["/bin/bash", "/app/scripts/entrypoint.sh"]
CMD ["evaluate"]
