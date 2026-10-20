"""
FAST-PLATE-OCR-VIETNAMESE: High-Performance ONNX Runtime Engine (Bước 6.1)
Lớp bao bọc (wrapper) mô hình Detector và CCT OCR chạy trên ONNX Runtime,
tối ưu hóa đa luồng, hỗ trợ dynamic batching và tự động kích hoạt ExecutionProvider thích hợp
(CUDA / TensorRT / DirectML / CPU) cho các hệ thống phần cứng biên.
"""

import os
import sys
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.decoder.beam_search import GrammarGuidedBeamSearchDecoder

logger = logging.getLogger("ONNXEngine")


def get_optimal_providers() -> List[str]:
    """Tự động lựa chọn ExecutionProvider tối ưu nhất của hệ thống."""
    if not HAS_ORT:
        return []
    available = ort.get_available_providers()
    priority = [
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "DmlExecutionProvider",
        "OpenVINOExecutionProvider",
        "CPUExecutionProvider",
    ]
    selected = [p for p in priority if p in available]
    return selected if selected else ["CPUExecutionProvider"]


def create_session(model_path: str, num_threads: Optional[int] = None) -> "ort.InferenceSession":
    """Tạo phiên suy luận ONNXRuntime với cấu hình tối ưu hiệu năng tối đa."""
    if not HAS_ORT:
        raise ImportError("onnxruntime chưa được cài đặt trong môi trường!")

    if num_threads is None:
        num_threads = min(8, os.cpu_count() or 4)

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = num_threads
    opts.inter_op_num_threads = 2
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    providers = get_optimal_providers()
    logger.info("Khởi tạo ONNX session cho [%s] với Providers: %s", os.path.basename(model_path), providers)
    try:
        return ort.InferenceSession(model_path, sess_options=opts, providers=providers)
    except Exception as e:
        logger.warning("Không thể khởi tạo ONNX session với providers %s (%s). Fallback sang CPUExecutionProvider.", providers, e)
        return ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])


class ONNXPlateDetector:
    """
    Module Detector tăng tốc ONNX Runtime:
    Nhận diện phương tiện & 4 điểm góc biển số trên ảnh Full-Scene với độ trễ siêu thấp.
    """
    def __init__(
        self,
        model_path: str = "weights/plate_detector.onnx",
        num_threads: Optional[int] = None,
        conf_thresh: float = 0.5,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không tìm thấy tệp ONNX: {model_path}")
        self.session = create_session(model_path, num_threads=num_threads)
        self.input_name = self.session.get_inputs()[0].name
        self.conf_thresh = conf_thresh

    def forward(self, x: Union[np.ndarray, "torch.Tensor"]) -> Dict[str, Any]:
        """
        Thực hiện suy luận:
        x: tensor [B, 3, H, W] chuẩn hóa (0..1)
        Trả về dictionary gồm 'confidence', 'bbox', 'corners', 'layout'.
        """
        if HAS_TORCH and isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
            was_torch = True
        else:
            x_np = np.asarray(x, dtype=np.float32)
            was_torch = False

        outputs = self.session.run(None, {self.input_name: x_np})
        # Outputs: [confidence, bbox, corners, layout]
        conf_np, bbox_np, corners_np, layout_np = outputs

        if was_torch and HAS_TORCH:
            return {
                "confidence": torch.from_numpy(conf_np),
                "bbox": torch.from_numpy(bbox_np),
                "corners": torch.from_numpy(corners_np),
                "layout": torch.from_numpy(layout_np),
            }
        return {
            "confidence": conf_np,
            "bbox": bbox_np,
            "corners": corners_np,
            "layout": layout_np,
        }

    def __call__(self, x):
        return self.forward(x)


class ONNXCCTPlateOCR:
    """
    Module Compact Convolutional Transformer OCR tăng tốc ONNX Runtime:
    Giải mã chuỗi ký tự trên dải ảnh biển số 64x256 với Beam Search ngữ pháp ràng buộc.
    """
    def __init__(
        self,
        model_path: str = "weights/fast_plate_ocr_vn.onnx",
        config_path: str = "configs/plate_config_vietnam.yaml",
        num_threads: Optional[int] = None,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không tìm thấy tệp ONNX: {model_path}")
        self.session = create_session(model_path, num_threads=num_threads)
        self.input_name = self.session.get_inputs()[0].name

        # Cấu hình alphabet và decoder
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.alphabet = cfg.get("alphabet", "0123456789ABCDEFGHKLMNPQRSTUVXYZ_")
        self.pad_char = cfg.get("pad_char", "_")
        self.max_plate_slots = cfg.get("max_plate_slots", 10)

        self.beam_decoder = GrammarGuidedBeamSearchDecoder(
            alphabet=self.alphabet,
            pad_char=self.pad_char,
            max_slots=self.max_plate_slots,
        )

    def forward(self, x: Union[np.ndarray, "torch.Tensor"]) -> Union[np.ndarray, "torch.Tensor"]:
        """
        x: tensor dải ảnh [B, 3, 64, 256]
        Trả về logits [B, 10, 33]
        """
        if HAS_TORCH and isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
            was_torch = True
        else:
            x_np = np.asarray(x, dtype=np.float32)
            was_torch = False

        outputs = self.session.run(None, {self.input_name: x_np})
        logits_np = outputs[0]

        if was_torch and HAS_TORCH:
            return torch.from_numpy(logits_np)
        return logits_np

    def __call__(self, x):
        return self.forward(x)

    def decode(
        self,
        logits: Union[np.ndarray, "torch.Tensor"],
        use_grammar_beam_search: bool = True,
        beam_width: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Giải mã logits thành text với Grammar-Guided Beam Search.
        """
        if not (HAS_TORCH and isinstance(logits, torch.Tensor)):
            logits = torch.from_numpy(logits)

        if use_grammar_beam_search:
            batch_res = self.beam_decoder.decode_batch(logits, beam_width=beam_width, apply_grammar=True)
            for r in batch_res:
                if "plate_text" not in r and "text" in r:
                    r["plate_text"] = r["text"]
            return batch_res

        # Greedy decoding fallback
        probs = F.softmax(logits, dim=-1)
        confidences, predictions = torch.max(probs, dim=-1)
        results = []
        batch_size = logits.shape[0]

        for b in range(batch_size):
            pred_indices = predictions[b].cpu().numpy()
            slot_confs = confidences[b].cpu().numpy()
            chars = []
            confs = []
            for idx, conf in zip(pred_indices, slot_confs):
                ch = self.alphabet[idx]
                if ch != self.pad_char:
                    chars.append(ch)
                    confs.append(float(conf))
            plate_text = "".join(chars)
            avg_conf = float(np.mean(confs)) if confs else 0.0
            results.append({
                "plate_text": plate_text,
                "confidence": avg_conf,
                "is_valid": True,
                "raw_sequence": [self.alphabet[i] for i in pred_indices],
            })
        return results
