#!/bin/bash
# ==============================================================================
# FAST-PLATE-OCR-VIETNAMESE: Container Entrypoint & Hardware Diagnostic Script
# ==============================================================================

set -e

echo "========================================================================"
echo "    FAST-PLATE-OCR-VIETNAMESE: COMMERCIAL ALPR PIPELINE CONTAINER"
echo "========================================================================"

# 1. Kiểm tra kết nối phần cứng và Driver GPU
echo ">>> [1/3] Kiểm tra kết nối phần cứng & Trình điều khiển GPU:"
if command -v nvidia-smi &> /dev/null; then
    echo "    [OK] Lệnh nvidia-smi khả dụng:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
else
    echo "    [INFO] nvidia-smi không khả dụng. Container đang chạy ở chế độ CPU."
fi

# 2. Xác nhận khả năng truy cập phần cứng của PyTorch & ONNX
python3 -c "
import torch
print('    PyTorch Version:', torch.__version__)
print('    CUDA Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('    CUDA Device Count:', torch.cuda.device_count())
    print('    CUDA Device Name:', torch.cuda.get_device_name(0))
else:
    print('    [Notice] PyTorch sẽ chạy trên CPU đa luồng.')
try:
    import onnxruntime as ort
    print('    ONNX Runtime Providers:', ort.get_available_providers())
except Exception as e:
    print('    ONNX Runtime:', e)
"

# 3. Phân luồng lệnh theo tham số truyền vào
COMMAND="${1:-evaluate}"

if [ "$COMMAND" = "test" ] || [ "$RUN_TESTS" = "true" ]; then
    echo ""
    echo ">>> [2/3] Chạy toàn diện bộ kiểm thử hệ thống (Regression Tests)..."
    python3 -m unittest discover tests -p "test_*.py" -v
    echo ">>> [3/3] Toàn bộ testcase đã hoàn tất!"
    exit 0

elif [ "$COMMAND" = "bash" ] || [ "$COMMAND" = "sh" ]; then
    echo ">>> Khởi động interactive shell trong container..."
    exec /bin/bash

elif [ "$COMMAND" = "webui" ]; then
    echo ""
    echo ">>> [2/3] Khởi chạy Gradio WebUI tại http://0.0.0.0:7860 ..."
    exec python3 app_webui.py --host 0.0.0.0 --port 7860

elif [ "$COMMAND" = "evaluate" ]; then
    echo ""
    echo ">>> [2/3] Khởi chạy công cụ thực nghiệm ALPR trên thư mục /app/samples..."
    python3 demo.py --samples /app/samples --output /app/output

    echo ""
    echo ">>> [3/3] Hoàn tất thực nghiệm! Kết quả đã được xuất ra thư mục mount /app/output."
    echo "    Container sẽ duy trì trạng thái hoạt động để đón nhận thêm dữ liệu."
    echo "    Bạn có thể truy cập shell bằng lệnh: docker exec -it alpr_vietnam_container bash"
    
    # Giữ container chạy nền nếu ở chế độ detached
    tail -f /dev/null
else
    # Thực thi lệnh bất kỳ mà người dùng truyền vào
    exec "$@"
fi
