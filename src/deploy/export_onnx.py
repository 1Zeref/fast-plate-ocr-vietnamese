"""
FAST-PLATE-OCR-VIETNAMESE: ONNX Export & Dynamic Quantization Module (Bước 6.1)
Chuyển đổi và lượng tử hóa mô hình Detector và CCT OCR sang định dạng ONNX/INT8
phục vụ tăng tốc suy luận trên các thiết bị phần cứng biên (Edge SoC, GPU nhúng).
"""

import os
import sys
import logging
from typing import Dict, Any, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import torch.nn as nn

try:
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

from src.detector.model import PlateCornerDetector
from src.ocr.cct_adapter import CCTPlateVietnamModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ExportONNX")


class ExportableDetectorWrapper(nn.Module):
    """Wrapper chuyển đổi output dictionary thành tuple để tương thích chuẩn ONNX đa nền tảng."""
    def __init__(self, detector: PlateCornerDetector):
        super().__init__()
        self.detector = detector

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.detector(x)
        return out["confidence"], out["bbox"], out["corners"], out["layout"]


def export_detector_to_onnx(
    weights_path: str = "weights/best_detector.pt",
    onnx_path: str = "weights/plate_detector.onnx",
    quant_path: Optional[str] = "weights/plate_detector_int8.onnx",
    input_size: Tuple[int, int] = (384, 384),
    opset: int = 18,
) -> Dict[str, str]:
    """
    Xuất mô hình PlateCornerDetector sang ONNX và lượng tử hóa động INT8.
    """
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    logger.info("Khởi tạo và tải trọng số Detector từ: %s", weights_path)
    model = PlateCornerDetector()
    if os.path.exists(weights_path):
        state = torch.load(weights_path, map_location="cpu")
        sd = state["model_state_dict"] if "model_state_dict" in state else state
        model.load_state_dict(sd)
    model.eval()

    wrapper = ExportableDetectorWrapper(model)
    wrapper.eval()

    dummy_input = torch.randn(1, 3, input_size[0], input_size[1], dtype=torch.float32)

    logger.info("Đang xuất ONNX Detector sang: %s (opset=%d)...", onnx_path, opset)
    torch.onnx.export(
        wrapper,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["confidence", "bbox", "corners", "layout"],
        dynamic_axes={
            "images": {0: "batch_size"},
            "confidence": {0: "batch_size"},
            "bbox": {0: "batch_size"},
            "corners": {0: "batch_size"},
            "layout": {0: "batch_size"},
        },
    )

    results = {"onnx": onnx_path}

    if quant_path and HAS_ONNX:
        logger.info("Đang lượng tử hóa động INT8 cho Detector sang: %s...", quant_path)
        quantize_dynamic(
            onnx_path,
            quant_path,
            weight_type=QuantType.QUInt8,
        )
        results["quant_int8"] = quant_path

    logger.info("Hoàn tất xuất Detector: %s", results)
    return results


def export_ocr_to_onnx(
    weights_path: str = "weights/fast_plate_ocr_vn.pt",
    config_path: str = "configs/plate_config_vietnam.yaml",
    onnx_path: str = "weights/fast_plate_ocr_vn.onnx",
    quant_path: Optional[str] = "weights/fast_plate_ocr_vn_int8.onnx",
    input_size: Tuple[int, int] = (64, 256),
    opset: int = 18,
) -> Dict[str, str]:
    """
    Xuất mô hình CCT OCR sang ONNX và lượng tử hóa động INT8.
    """
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    logger.info("Khởi tạo và tải trọng số CCT OCR từ: %s", weights_path)
    model = CCTPlateVietnamModel(config_path=config_path)
    if os.path.exists(weights_path):
        state = torch.load(weights_path, map_location="cpu")
        sd = state["model_state_dict"] if "model_state_dict" in state else state
        model.load_state_dict(sd)
    model.eval()

    dummy_input = torch.randn(1, 3, input_size[0], input_size[1], dtype=torch.float32)

    logger.info("Đang xuất ONNX CCT OCR sang: %s (opset=%d)...", onnx_path, opset)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["plate_strips"],
        output_names=["logits"],
        dynamic_axes={
            "plate_strips": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )

    results = {"onnx": onnx_path}

    if quant_path and HAS_ONNX:
        logger.info("Đang lượng tử hóa động INT8 cho CCT OCR sang: %s...", quant_path)
        quantize_dynamic(
            onnx_path,
            quant_path,
            weight_type=QuantType.QUInt8,
        )
        results["quant_int8"] = quant_path

    logger.info("Hoàn tất xuất CCT OCR: %s", results)
    return results


def export_all():
    """Hàm tiện ích xuất toàn bộ hệ thống mô hình sang ONNX & INT8."""
    print("=" * 60)
    print("XUẤT TOÀN BỘ MÔ HÌNH SANG ONNX & LƯỢNG TỬ HÓA INT8 (BƯỚC 6.1)")
    print("=" * 60)
    det_res = export_detector_to_onnx()
    ocr_res = export_ocr_to_onnx()
    return {"detector": det_res, "ocr": ocr_res}


if __name__ == "__main__":
    export_all()
