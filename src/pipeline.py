"""
FAST-PLATE-OCR-VIETNAMESE: Unified End-to-End LPR Pipeline
Hệ thống nhận dạng biển số xe Việt Nam toàn diện kết hợp song song 2 bộ dữ liệu:
1. Module Detection (bomaich/vnlicenseplate): Phát hiện xe & 4 góc biển số trong ảnh toàn cảnh 1080p.
2. Module Rectification: Nắn phẳng Homography & Sinh Virtual 1-Line Strip cho biển 2 dòng.
3. Module CCT OCR (topkek69/vietnamese-license-plate-ocr): Nhận dạng ký tự độ chính xác cao.
4. Module Grammar Beam Search: Giải mã ràng buộc ngữ pháp, triệt tiêu lỗi nhầm quang học 8 <-> B, 0 <-> D.
"""

import os
import sys
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.detector.infer import PlateDetectorInference
from src.rectification.pipeline import PlateRectificationPipeline
from src.ocr.cct_adapter import CCTPlateVietnamModel
from src.decoder.beam_search import GrammarGuidedBeamSearchDecoder


class VietnamesePlateOCRPipeline:
    """
    Quy trình tích hợp khép kín End-to-End từ ảnh chụp thực tế tới chuỗi ký tự chuẩn:
    Image (1080p) -> Detector -> 4 Corners -> Homography Warp -> Virtual Strip -> CCT OCR -> Grammar Beam Search.
    """

    def __init__(
        self,
        detector_weights: str = "weights/best_detector.pt",
        ocr_weights: str = "weights/fast_plate_ocr_vn.pt",
        config_path: str = "configs/plate_config_vietnam.yaml",
        device: Optional[str] = None,
        conf_threshold: float = 0.25,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.config_path = config_path

        # 1. Khởi tạo Module Phát hiện Biển số (Detection)
        self.detector = PlateDetectorInference(
            weights_path=detector_weights if os.path.exists(detector_weights) else None,
            conf_threshold=conf_threshold,
            device=self.device,
        )

        # 2. Khởi tạo Module Nắn phối cảnh & Phân luồng hình học
        self.rectifier = PlateRectificationPipeline(
            target_ocr_size=(256, 64),
            aspect_ratio_threshold=2.40,
        )

        # 3. Khởi tạo Mô hình CCT OCR
        self.ocr_model = CCTPlateVietnamModel(config_path=config_path).to(self.device)
        if os.path.exists(ocr_weights):
            ckpt = torch.load(ocr_weights, map_location=self.device)
            state_dict = ckpt.get("model_state_dict", ckpt)
            self.ocr_model.load_state_dict(state_dict)
        self.ocr_model.eval()

        # 4. Khởi tạo Bộ giải mã Grammar-Guided Beam Search
        self.decoder = GrammarGuidedBeamSearchDecoder(config_path=config_path)

    def _normalize_tensor(self, strip_bgr: np.ndarray) -> torch.Tensor:
        """Chuẩn hóa dải ảnh 64x256 thành Tensor đầu vào cho mạng CCT."""
        img_rgb = cv2.cvtColor(strip_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm_img = (img_rgb - mean) / std
        tensor = torch.from_numpy(norm_img).permute(2, 0, 1).unsqueeze(0).float()
        return tensor.to(self.device)

    def predict(
        self,
        image_input: Union[str, np.ndarray],
        beam_width: int = 5,
        use_grammar: bool = True,
        conf_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Nhận dạng toàn diện các biển số xe trong ảnh:
        image_input: Đường dẫn tệp ảnh hoặc mảng numpy BGR.
        """
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Không tìm thấy tệp ảnh: {image_input}")
            image = cv2.imread(image_input)
            if image is None:
                raise ValueError(f"Không thể đọc tệp ảnh: {image_input}")
        else:
            image = image_input.copy()

        h, w = image.shape[:2]

        # -------------------------------------------------------------
        # BƯỚC 1: ĐỊNH VỊ & PHÁT HIỆN BIỂN SỐ (DETECTION)
        # -------------------------------------------------------------
        if conf_threshold is not None:
            self.detector.conf_threshold = conf_threshold

        detections = self.detector.detect(image)

        # Cơ chế Fallback thông minh: Nếu ảnh truyền vào là ảnh crop sẵn biển số (như dataset topkek69)
        # mà detector không phát hiện được hộp bao trong khung lớn, coi toàn bộ ảnh là 1 biển số
        if len(detections) == 0:
            detections = [{
                "plate_bbox": [0, 0, w - 1, h - 1],
                "corners": [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
                "confidence": 1.0,
                "layout": "2_LINE" if (w / float(max(1, h))) < 2.2 else "1_LINE",
                "is_fallback": True,
            }]

        results: List[Dict[str, Any]] = []

        # -------------------------------------------------------------
        # BƯỚC 2 + 3 + 4: NẮN PHẲNG -> CCT OCR -> GRAMMAR BEAM SEARCH
        # -------------------------------------------------------------
        for det in detections:
            corners = det["corners"]
            det_conf = det["confidence"]

            # 2. Nắn phối cảnh & Sinh Virtual 1-Line Strip
            rect_res = self.rectifier.process(
                image=image,
                corners=corners,
                detector_layout_hint=det.get("layout"),
                detector_confidence=det_conf,
            )
            ocr_strip = rect_res["ocr_input_strip"]
            layout = rect_res["layout"]

            # 3. Chuẩn hóa Tensor & Suy luận mạng CCT
            input_tensor = self._normalize_tensor(ocr_strip)
            with torch.no_grad():
                logits = self.ocr_model(input_tensor)

            # 4. Giải mã ràng buộc ngữ pháp (Grammar-Guided Beam Search)
            decoded = self.decoder.decode_single(
                logits[0],
                beam_width=beam_width,
                apply_grammar=use_grammar,
            )

            dfa_meta = decoded.get("dfa_meta", {})

            results.append({
                "plate_text": decoded["text"],
                "ocr_confidence": decoded["confidence"],
                "detection_confidence": det_conf,
                "overall_confidence": round(float(det_conf * decoded["confidence"]), 4),
                "is_valid_grammar": decoded["is_valid"],
                "plate_type": dfa_meta.get("plate_type", "UNKNOWN"),
                "province_name": dfa_meta.get("province_name", ""),
                "province_code": dfa_meta.get("province_code", ""),
                "series": dfa_meta.get("series", ""),
                "number": dfa_meta.get("number", ""),
                "layout": layout,
                "bbox": det["plate_bbox"],
                "corners": corners,
                "corrections_made": decoded.get("corrections_made", []),
                "virtual_strip_shape": ocr_strip.shape,
            })

        return results

    def draw_predictions(
        self,
        image: np.ndarray,
        results: List[Dict[str, Any]],
        show_corners: bool = True,
    ) -> np.ndarray:
        """Vẽ hộp bao, 4 góc và chuỗi nhận diện lên ảnh trực quan."""
        vis_img = image.copy()
        for res in results:
            text = res["plate_text"]
            conf = res["overall_confidence"]
            bbox = res["bbox"]
            corners = res["corners"]

            # 1. Vẽ tứ giác 4 góc
            if show_corners and len(corners) == 4:
                pts = np.array(corners, np.int32).reshape((-1, 1, 2))
                cv2.polylines(vis_img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                for pt in corners:
                    cv2.circle(vis_img, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)
            else:
                cv2.rectangle(vis_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

            # 2. Vẽ nhãn văn bản
            label_str = f"{text} ({conf*100:.1f}%)"
            ty = max(20, bbox[1] - 8)
            cv2.putText(
                vis_img,
                label_str,
                (bbox[0], ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                lineType=cv2.LINE_AA,
            )

        return vis_img
