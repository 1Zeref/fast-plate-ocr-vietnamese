"""
FAST-PLATE-OCR-VIETNAMESE: Deployment & Validation Package (Giai đoạn 6)
"""

from .export_onnx import export_detector_to_onnx, export_ocr_to_onnx, export_all
from .onnx_engine import ONNXPlateDetector, ONNXCCTPlateOCR, create_session, get_optimal_providers
from .tensorrt_engine import generate_trtexec_commands, write_tensorrt_build_scripts
from .benchmark import BenchmarkSuite
from .validation import ConfusionMatrixTracker, EndToEndSystemValidator

__all__ = [
    "export_detector_to_onnx",
    "export_ocr_to_onnx",
    "export_all",
    "ONNXPlateDetector",
    "ONNXCCTPlateOCR",
    "create_session",
    "get_optimal_providers",
    "generate_trtexec_commands",
    "write_tensorrt_build_scripts",
    "BenchmarkSuite",
    "ConfusionMatrixTracker",
    "EndToEndSystemValidator",
]
