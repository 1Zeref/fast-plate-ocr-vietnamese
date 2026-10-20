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
from src.tracking.byte_tracker import ByteTracker
from src.tracking.quality import FrameQualityAssessor
from src.tracking.buffer import BestKFrameBuffer
from src.tracking.consensus import TemporalConsensusEngine


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
        use_onnx: bool = False,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.config_path = config_path
        self.use_onnx = use_onnx

        # 1. Khởi tạo Module Phát hiện Biển số (Detection)
        self.detector = PlateDetectorInference(
            weights_path=detector_weights if os.path.exists(detector_weights) else None,
            conf_threshold=conf_threshold,
            device=self.device,
        )
        if self.use_onnx and os.path.exists("weights/plate_detector.onnx"):
            from src.deploy.onnx_engine import ONNXPlateDetector
            self.detector.model = ONNXPlateDetector("weights/plate_detector.onnx")

        # 2. Khởi tạo Module Nắn phối cảnh & Phân luồng hình học
        self.rectifier = PlateRectificationPipeline(
            target_ocr_size=(256, 64),
            aspect_ratio_threshold=2.40,
        )

        # 3. Khởi tạo Mô hình CCT OCR
        if self.use_onnx and os.path.exists("weights/fast_plate_ocr_vn.onnx"):
            from src.deploy.onnx_engine import ONNXCCTPlateOCR
            self.ocr_model = ONNXCCTPlateOCR(
                model_path="weights/fast_plate_ocr_vn.onnx",
                config_path=config_path,
            )
        else:
            self.ocr_model = CCTPlateVietnamModel(config_path=config_path).to(self.device)
            if os.path.exists(ocr_weights):
                ckpt = torch.load(ocr_weights, map_location=self.device)
                state_dict = ckpt.get("model_state_dict", ckpt)
                self.ocr_model.load_state_dict(state_dict)
            self.ocr_model.eval()

        # 4. Khởi tạo Bộ giải mã Grammar-Guided Beam Search
        self.decoder = GrammarGuidedBeamSearchDecoder(config_path=config_path)

        # 5. Khởi tạo Bộ bám vết ByteTrack & Lọc chất lượng quang học (Giai đoạn 5 - Bước 5.1)
        self.tracker = ByteTracker(track_thresh=conf_threshold, low_score_thresh=0.10)
        self.quality_assessor = FrameQualityAssessor(sharpness_threshold=85.0)
        self.frame_buffer = BestKFrameBuffer(k=5, min_quality_threshold=0.35)
        self.consensus_engine = TemporalConsensusEngine(config_path=config_path)
        self.frame_count = 0

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

    def process_video_frame(
        self,
        frame: np.ndarray,
        frame_idx: Optional[int] = None,
        conf_threshold: Optional[float] = None,
        beam_width: int = 5,
        use_grammar: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Xử lý 1 frame trong luồng video liên tục theo thời gian thực (Bước 5.1):
        1. Định vị các biển số trong frame (Detector).
        2. Bám vết ByteTrack: Gán Tracklet ID bất biến cho từng xe qua thời gian.
        3. Đánh giá chất lượng quang học của frame (Laplace sharpness, geometric skew).
        4. Cập nhật vào Hàng đợi Best-K buffer cho từng Tracklet ID.
        5. Nếu Tracklet đã tích lũy đủ khung hình chất lượng cao, chuyển giao cho CCT OCR.
        """
        self.frame_count += 1
        current_fidx = frame_idx if frame_idx is not None else self.frame_count

        h, w = frame.shape[:2]

        # 1. Phát hiện các biển số trong frame
        if conf_threshold is not None:
            self.detector.conf_threshold = conf_threshold
        raw_detections = self.detector.detect(frame)

        # 2. Cập nhật thuật toán bám vết ByteTrack
        tracked_objects = self.tracker.update(raw_detections)

        active_track_ids = [t.track_id for t in tracked_objects]
        self.frame_buffer.cleanup_stale_tracks(active_track_ids, max_idle_frames=45, current_frame=current_fidx)

        frame_results: List[Dict[str, Any]] = []

        # 3. Duyệt qua từng Tracklet đang hoạt động
        for track in tracked_objects:
            track_id = track.track_id
            bbox = [int(v) for v in track.tlbr]
            corners = track.corners

            # Cắt vùng ảnh biển số để đánh giá chất lượng
            bx1, by1 = max(0, bbox[0]), max(0, bbox[1])
            bx2, by2 = min(w - 1, bbox[2]), min(h - 1, bbox[3])

            if bx2 <= bx1 or by2 <= by1:
                continue

            crop_plate = frame[by1:by2, bx1:bx2]

            # 4. Đánh giá chất lượng quang học của frame (Laplacian + Skew)
            q_info = self.quality_assessor.evaluate_quality(crop_plate, corners=corners, bbox=bbox)
            q_score = q_info["overall_quality"]

            # Nắn phối cảnh & tạo dải Virtual 1-Line Strip
            rect_res = self.rectifier.process(
                image=frame,
                corners=corners if corners else [[bx1, by1], [bx2, by1], [bx2, by2], [bx1, by2]],
                detector_layout_hint=track.layout,
                detector_confidence=track.score,
            )
            virtual_strip = rect_res["ocr_input_strip"]

            # 5. Cập nhật vào hàng đợi Best-K buffer
            was_accepted = self.frame_buffer.add_frame(
                track_id=track_id,
                frame_idx=current_fidx,
                quality_score=q_score,
                crop_image=crop_plate,
                corners=corners,
                bbox=bbox,
                virtual_strip=virtual_strip,
                quality_meta=q_info,
            )

            is_ready = self.frame_buffer.is_ready_for_recognition(track_id, min_frames=2, min_avg_quality=0.35)
            best_buffered = self.frame_buffer.get_best_frames(track_id)

            # 6. Nếu đã tích lũy đủ khung hình, thực hiện hợp nhất đa khung hình (Temporal Consensus)
            plate_text = ""
            ocr_conf = 0.0
            is_valid = False
            plate_type = "UNKNOWN"
            agreement_rate = 0.0
            outliers = []

            # Trích xuất logits và dự đoán cho các frame trong buffer nếu chưa có
            for b_frame in best_buffered:
                if "logits" not in b_frame and b_frame.get("virtual_strip") is not None:
                    strip_tensor = self._normalize_tensor(b_frame["virtual_strip"])
                    with torch.no_grad():
                        l = self.ocr_model(strip_tensor)
                    b_frame["logits"] = l[0].cpu().numpy()
                    dec = self.decoder.decode_single(l[0], beam_width=beam_width, apply_grammar=use_grammar)
                    b_frame["text"] = dec["text"]
                    b_frame["confidence"] = dec["confidence"]

            if is_ready and len(best_buffered) >= 2:
                consensus_res = self.consensus_engine.consolidate_tracklet(
                    track_id=track_id,
                    frames_data=best_buffered,
                    beam_width=beam_width,
                )
                plate_text = consensus_res["plate_text"]
                ocr_conf = consensus_res["consensus_confidence"]
                is_valid = consensus_res["is_valid"]
                plate_type = consensus_res.get("dfa_meta", {}).get("plate_type", "UNKNOWN")
                agreement_rate = consensus_res["agreement_rate"]
                outliers = consensus_res.get("outliers_suppressed", [])
            elif len(best_buffered) > 0 and "text" in best_buffered[0]:
                plate_text = best_buffered[0]["text"]
                ocr_conf = best_buffered[0].get("confidence", 0.0)

            track_summary = self.frame_buffer.get_track_summary(track_id)

            frame_results.append({
                "track_id": track_id,
                "bbox": bbox,
                "corners": corners,
                "detection_confidence": round(float(track.score), 4),
                "frame_quality": q_info,
                "buffered_k_frames": len(best_buffered),
                "is_ready_for_ocr": is_ready,
                "plate_text": plate_text,
                "ocr_confidence": ocr_conf,
                "is_valid_grammar": is_valid,
                "plate_type": plate_type,
                "agreement_rate": agreement_rate,
                "outliers_suppressed": outliers,
                "track_summary": track_summary,
            })

        return frame_results

    def consolidate_vehicle(self, track_id: int, beam_width: int = 5) -> Dict[str, Any]:
        """
        Đưa ra quyết định nhận diện chính xác duy nhất cho phương tiện (khi xe hoàn thành hành trình qua camera).
        Tự động giải phóng bộ nhớ tracklet sau khi xuất kết quả.
        """
        best_buffered = self.frame_buffer.get_best_frames(track_id)
        if not best_buffered:
            return {"track_id": track_id, "plate_text": "", "is_valid": False, "consensus_confidence": 0.0}

        # Đảm bảo tất cả frame đã có logits
        for b_frame in best_buffered:
            if "logits" not in b_frame and b_frame.get("virtual_strip") is not None:
                strip_tensor = self._normalize_tensor(b_frame["virtual_strip"])
                with torch.no_grad():
                    l = self.ocr_model(strip_tensor)
                b_frame["logits"] = l[0].cpu().numpy()
                dec = self.decoder.decode_single(l[0], beam_width=beam_width, apply_grammar=True)
                b_frame["text"] = dec["text"]
                b_frame["confidence"] = dec["confidence"]

        final_decision = self.consensus_engine.consolidate_tracklet(
            track_id=track_id,
            frames_data=best_buffered,
            beam_width=beam_width,
        )

        # Giải phóng track khỏi buffer
        self.frame_buffer.remove_track(track_id)
        return final_decision
