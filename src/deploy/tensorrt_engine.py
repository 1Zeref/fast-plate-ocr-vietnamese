"""
FAST-PLATE-OCR-VIETNAMESE: TensorRT Engine Exporter & Runtime Wrapper (Bước 6.1)
Tối ưu hóa sâu cho phần cứng GPU NVIDIA / Edge AI (Jetson Orin, Xavier, Nano, T4/L4/RTX).
Hỗ trợ sinh lệnh biên dịch trtexec (FP16/INT8) và lớp thực thi TensorRT runtime.
"""

import os
import sys
import logging
from typing import Dict, Any, Optional, Tuple, List

logger = logging.getLogger("TensorRTEngine")

try:
    import tensorrt as trt
    HAS_TRT = True
except ImportError:
    HAS_TRT = False

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False


def generate_trtexec_commands(
    detector_onnx: str = "weights/plate_detector.onnx",
    ocr_onnx: str = "weights/fast_plate_ocr_vn.onnx",
    output_dir: str = "weights/tensorrt",
) -> Dict[str, str]:
    """
    Sinh các lệnh trtexec chuẩn hóa tối ưu hóa cao nhất cho TensorRT (FP16 & INT8).
    """
    os.makedirs(output_dir, exist_ok=True)
    det_engine_fp16 = os.path.join(output_dir, "plate_detector_fp16.engine")
    ocr_engine_fp16 = os.path.join(output_dir, "fast_plate_ocr_vn_fp16.engine")

    cmd_detector_fp16 = (
        f"trtexec --onnx={detector_onnx} "
        f"--saveEngine={det_engine_fp16} "
        f"--fp16 "
        f"--minShapes=images:1x3x384x384 "
        f"--optShapes=images:1x3x384x384 "
        f"--maxShapes=images:8x3x384x384 "
        f"--memPoolSize=workspace:1024MiB"
    )

    cmd_ocr_fp16 = (
        f"trtexec --onnx={ocr_onnx} "
        f"--saveEngine={ocr_engine_fp16} "
        f"--fp16 "
        f"--minShapes=plate_strips:1x3x64x256 "
        f"--optShapes=plate_strips:4x3x64x256 "
        f"--maxShapes=plate_strips:16x3x64x256 "
        f"--memPoolSize=workspace:512MiB"
    )

    return {
        "detector_fp16_cmd": cmd_detector_fp16,
        "ocr_fp16_cmd": cmd_ocr_fp16,
        "det_engine_path": det_engine_fp16,
        "ocr_engine_path": ocr_engine_fp16,
    }


def write_tensorrt_build_scripts(output_dir: str = "scripts"):
    """Tạo tệp bash script và batch script phục vụ build engine một chạm trên server GPU / Jetson."""
    os.makedirs(output_dir, exist_ok=True)
    cmds = generate_trtexec_commands()

    sh_path = os.path.join(output_dir, "build_tensorrt_engines.sh")
    with open(sh_path, "w", encoding="utf-8") as f:
        f.write("#!/bin/bash\n")
        f.write("# Tối ưu hóa TensorRT FP16 cho Fast-Plate-OCR-Vietnamese\n")
        f.write("set -e\n\n")
        f.write("echo '>>> [1/2] Biên dịch TensorRT Engine cho Detector (FP16)...'\n")
        f.write(f"{cmds['detector_fp16_cmd']}\n\n")
        f.write("echo '>>> [2/2] Biên dịch TensorRT Engine cho CCT OCR (FP16)...'\n")
        f.write(f"{cmds['ocr_fp16_cmd']}\n\n")
        f.write("echo '>>> Hoàn tất tối ưu hóa TensorRT!'\n")

    bat_path = os.path.join(output_dir, "build_tensorrt_engines.bat")
    with open(bat_path, "w", encoding="utf-8") as f:
        f.write("@echo off\n")
        f.write("REM Tối ưu hóa TensorRT FP16 cho Fast-Plate-OCR-Vietnamese trên Windows GPU\n\n")
        f.write("echo [1/2] Bien dich TensorRT Engine cho Detector (FP16)...\n")
        f.write(f"{cmds['detector_fp16_cmd']}\n\n")
        f.write("echo [2/2] Bien dich TensorRT Engine cho CCT OCR (FP16)...\n")
        f.write(f"{cmds['ocr_fp16_cmd']}\n\n")
        f.write("echo Hoan tat toi uu hoa TensorRT!\n")

    return sh_path, bat_path


class TensorRTRuntimeEngine:
    """
    Lớp bao bọc suy luận TensorRT trực tiếp từ tệp *.engine (nếu tensorrt và pycuda được cài đặt).
    """
    def __init__(self, engine_path: str):
        self.engine_path = engine_path
        self.is_available = HAS_TRT and HAS_CUDA and os.path.exists(engine_path)
        if not self.is_available:
            logger.info("TensorRT runtime không sẵn sàng trên máy hiện tại. Sẽ dùng ONNXRuntime fallback.")
            self.context = None
            return

        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

    def is_loaded(self) -> bool:
        return self.context is not None
