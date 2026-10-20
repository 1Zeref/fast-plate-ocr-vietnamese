"""
FAST-PLATE-OCR-VIETNAMESE: Inference Service Layer (Bước 1 WebUI)
Lớp bao bọc dịch vụ thống nhất phục vụ giao diện WebUI (Gradio / FastAPI),
tối ưu hóa nạp trước mô hình vào bộ nhớ, hỗ trợ bóc tách từng công đoạn trung gian
(Explainable AI / Pipeline Inspector) và streaming luồng video thời gian thực.
"""

import os
import sys
import time
import logging
from typing import Dict, Any, List, Tuple, Optional, Generator, Union

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
import pandas as pd

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.pipeline import VietnamesePlateOCRPipeline
from src.grammar.dfa import VietnamPlateDFA

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ALPRService")


class ALPRService:
    """
    Dịch vụ suy luận ALPR thống nhất:
    Nạp sẵn Pipeline vào RAM/VRAM để phục vụ suy luận tức thì với độ trễ thấp nhất.
    """

    _instance: Optional["ALPRService"] = None

    @classmethod
    def get_instance(
        cls,
        device: Optional[str] = None,
        use_onnx: bool = True,
        conf_threshold: float = 0.40,
    ) -> "ALPRService":
        """Lấy singleton instance để tránh nạp lại trọng số nhiều lần."""
        if cls._instance is None:
            cls._instance = cls(device=device, use_onnx=use_onnx, conf_threshold=conf_threshold)
        return cls._instance

    def __init__(
        self,
        device: Optional[str] = None,
        use_onnx: bool = True,
        conf_threshold: float = 0.40,
    ):
        self.device = device or ("cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu")
        self.use_onnx = use_onnx
        self.conf_threshold = conf_threshold
        logger.info("Khởi tạo ALPRService trên nền tảng: [%s] (ONNX=%s)...", self.device, self.use_onnx)

        # Khởi tạo Pipeline chính
        self.pipeline = VietnamesePlateOCRPipeline(
            device=self.device,
            conf_threshold=self.conf_threshold,
            use_onnx=self.use_onnx,
        )
        self.dfa = VietnamPlateDFA()
        logger.info("ALPRService đã sẵn sàng phục vụ suy luận!")

    def process_image(
        self,
        image_input: Union[str, np.ndarray],
        conf_threshold: float = 0.40,
        beam_width: int = 5,
        use_grammar: bool = True,
    ) -> Dict[str, Any]:
        """
        Nhận dạng toàn diện cho 1 ảnh tĩnh:
        Trả về ảnh chú thích (RGB), danh sách chi tiết các phát hiện và bảng DataFrame.
        """
        if isinstance(image_input, str):
            img_bgr = cv2.imread(image_input)
            if img_bgr is None:
                raise ValueError(f"Không thể đọc tệp ảnh: {image_input}")
        else:
            # Gradio truyền ảnh dạng numpy RGB hoặc BGR
            img_bgr = cv2.cvtColor(image_input, cv2.COLOR_RGB2BGR) if image_input.shape[-1] == 3 else image_input.copy()

        t0 = time.perf_counter()
        detections = self.pipeline.predict(
            img_bgr,
            beam_width=beam_width,
            use_grammar=use_grammar,
            conf_threshold=conf_threshold,
        )
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000.0

        # Vẽ chú thích trực quan lên ảnh
        annotated_bgr = self.pipeline.draw_predictions(img_bgr, detections)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        # Định dạng dữ liệu cho Dataframe
        df_rows = []
        for idx, det in enumerate(detections, 1):
            df_rows.append({
                "STT": idx,
                "Biển số": det["plate_text"],
                "Bố cục": "1 Dòng (Dài)" if det["layout"] == "1_LINE" else "2 Dòng (Vuông)",
                "Địa phương": det.get("province_name") or "Chưa xác định",
                "Phân loại": det.get("plate_type") or "CIVILIAN",
                "Hợp lệ DFA": "✅ Hợp lệ" if det["is_valid_grammar"] else "❌ Vi phạm cú pháp",
                "Độ tin cậy OCR": f"{det['ocr_confidence'] * 100:.1f}%",
                "Tin cậy Phát hiện": f"{det['detection_confidence'] * 100:.1f}%",
            })

        df_summary = pd.DataFrame(df_rows) if df_rows else pd.DataFrame(
            columns=["STT", "Biển số", "Bố cục", "Địa phương", "Phân loại", "Hợp lệ DFA", "Độ tin cậy OCR", "Tin cậy Phát hiện"]
        )

        return {
            "annotated_image": annotated_rgb,
            "detections": detections,
            "dataframe": df_summary,
            "latency_ms": round(elapsed_ms, 2),
            "num_plates": len(detections),
        }

    def inspect_pipeline(
        self,
        image_input: Union[str, np.ndarray],
        conf_threshold: float = 0.40,
        beam_width: int = 5,
        plate_idx: int = 0,
    ) -> Dict[str, Any]:
        """
        Bóc tách chi tiết các công đoạn trung gian cho Tab Inspector (Explainable AI):
        1. Vùng crop phương tiện (Vehicle / Plate Crop)
        2. Tứ giác 4 điểm góc phát hiện được
        3. Ảnh nắn phẳng Homography (64x256)
        4. Dải ảnh Virtual 1-Line Strip
        5. Cơ chế giải mã ràng buộc DFA & Sửa lỗi quang học
        """
        if isinstance(image_input, str):
            img_bgr = cv2.imread(image_input)
        else:
            img_bgr = cv2.cvtColor(image_input, cv2.COLOR_RGB2BGR) if image_input.shape[-1] == 3 else image_input.copy()

        h, w = img_bgr.shape[:2]
        self.pipeline.detector.conf_threshold = conf_threshold
        detections = self.pipeline.detector.detect(img_bgr)

        # Fallback nếu là ảnh crop biển số sẵn
        if not detections:
            detections = [{
                "plate_bbox": [0, 0, w - 1, h - 1],
                "corners": [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
                "confidence": 1.0,
                "layout": "2_LINE" if (w / float(max(1, h))) < 2.2 else "1_LINE",
            }]

        target_det = detections[min(plate_idx, len(detections) - 1)]
        bbox = target_det["plate_bbox"]
        corners = target_det["corners"]

        # Công đoạn 1: Crop Bounding Box
        bx1, by1, bx2, by2 = bbox
        bx1, by1 = max(0, bx1), max(0, by1)
        bx2, by2 = min(w - 1, bx2), min(h - 1, by2)
        step1_crop = img_bgr[by1:by2+1, bx1:bx2+1]
        if step1_crop.size == 0:
            step1_crop = img_bgr

        # Công đoạn 2: Vẽ 4 góc tứ giác lồi nổi bật
        step2_quad = img_bgr.copy()
        pts = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(step2_quad, [pts], isClosed=True, color=(0, 255, 255), thickness=3)
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255)] # TL=Đỏ, TR=Xanh lá, BR=Xanh dương, BL=Tím
        labels = ["TL (Góc 1)", "TR (Góc 2)", "BR (Góc 3)", "BL (Góc 4)"]
        for pt, col, lbl in zip(corners, colors, labels):
            cv2.circle(step2_quad, (int(pt[0]), int(pt[1])), 6, col, -1)
            cv2.putText(step2_quad, lbl, (int(pt[0]) + 5, int(pt[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        
        # Crop quanh quad
        pad = 20
        qx1, qy1 = max(0, bx1 - pad), max(0, by1 - pad)
        qx2, qy2 = min(w - 1, bx2 + pad), min(h - 1, by2 + pad)
        step2_quad_crop = step2_quad[qy1:qy2+1, qx1:qx2+1]

        # Công đoạn 3 & 4: Nắn phối cảnh & Ghép dòng ảo
        rect_res = self.pipeline.rectifier.process(
            image=img_bgr,
            corners=corners,
            detector_layout_hint=target_det.get("layout"),
            detector_confidence=target_det.get("confidence", 1.0),
        )
        step3_warped = rect_res.get("rectified_raw", rect_res.get("rectified_plate"))
        step4_virtual_strip = rect_res["ocr_input_strip"]
        layout = rect_res["layout"]

        # Công đoạn 5: Suy luận CCT OCR & Phân tích Ngữ pháp DFA
        input_tensor = self.pipeline._normalize_tensor(step4_virtual_strip)
        with torch.no_grad():
            logits = self.pipeline.ocr_model(input_tensor)

        decoded = self.pipeline.decoder.decode_single(logits[0], beam_width=beam_width, apply_grammar=True)
        greedy_raw = decoded.get("raw_greedy_text", "")
        constrained_text = decoded["text"]
        corrections = decoded.get("corrections_made", [])
        dfa_meta = decoded.get("dfa_meta", {})

        # Chuyển đổi các ảnh sang RGB để Gradio hiển thị
        to_rgb = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB) if x is not None and len(x.shape) == 3 else x

        explanation_text = (
            f"### Kết Quả Phân Tích Chuyên Sâu (Explainable AI):\n"
            f"- **Chuỗi Thô Ban Đầu (Greedy Unconstrained):** `{greedy_raw}`\n"
            f"- **Chuỗi Sau Khi Áp Mặt Nạ Ngữ Pháp (DFA Constrained):** `{constrained_text}`\n"
            f"- **Trạng Thái Kiểm Định Pháp Lý:** {'✅ HỢP LỆ (Thông tư 24/2023/TT-BCA)' if decoded['is_valid'] else '❌ VI PHẠM'}\n"
            f"- **Mã Tỉnh / Thành Phố:** {dfa_meta.get('province_name')} ({dfa_meta.get('province_code')})\n"
            f"- **Sê-ri Đăng Ký:** {dfa_meta.get('series')}\n"
            f"- **Dãy Số Thứ Tự:** {dfa_meta.get('number')}\n"
            f"- **Loại Phương Tiện Quy Chuẩn:** {dfa_meta.get('plate_type')}\n"
            f"- **Số Lượng Sửa Đổi Nhầm Lẫn Quang Học:** {len(corrections)} vị trí\n"
        )

        if corrections:
            explanation_text += "\n#### Chi Tiết Sửa Lỗi Quang Học Tự Động:\n"
            for c in corrections:
                explanation_text += (
                    f"  - **Vị trí Slot {c['slot']}:** Nhận dạng quang học ban đầu `{c['greedy_raw']}` "
                    f"-> Đã được DFA nắn chuẩn thành `{c['grammar_guided']}` (Loại: `{c['confusion_type']}`).\n"
                )

        return {
            "step1_crop": to_rgb(step1_crop),
            "step2_corners": to_rgb(step2_quad_crop),
            "step3_warped": to_rgb(step3_warped),
            "step4_virtual_strip": to_rgb(step4_virtual_strip),
            "layout": layout,
            "explanation": explanation_text,
            "decoded_info": decoded,
        }

    def process_video_generator(
        self,
        video_path: str,
        frame_skip: int = 1,
        consensus_k: int = 5,
        conf_threshold: float = 0.40,
    ) -> Generator[Tuple[np.ndarray, pd.DataFrame, str, float], None, None]:
        """
        Bộ Generator xử lý video thời gian thực cho Gradio:
        Yield liên tục từng frame đã annotate và bảng lịch sử phương tiện (Vehicle Traffic Log).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Không thể mở video: %s", video_path)
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_source = cap.get(cv2.CAP_PROP_FPS) or 25.0

        # Đặt lại bộ đệm
        self.pipeline.frame_buffer.buffers.clear()
        self.pipeline.frame_buffer.k = consensus_k
        self.pipeline.frame_count = 0

        frame_idx = 0
        logged_vehicles: Dict[int, Dict[str, Any]] = {}

        t_start = time.perf_counter()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % frame_skip != 0:
                continue

            t_frame_start = time.perf_counter()
            tracked_res = self.pipeline.process_video_frame(frame)
            active_tracks = tracked_res.get("active_tracks", [])

            # Vẽ bounding box và Track ID lên frame
            for trk in active_tracks:
                bbox = trk.get("plate_bbox", [0, 0, 0, 0])
                tid = trk.get("track_id", 0)
                conf = trk.get("confidence", 0.0)

                # Kiểm tra xem xe này đã có kết quả đồng thuận chưa
                c_res = self.pipeline.consolidate_vehicle(tid)
                plate_display = f"ID:{tid}"
                if c_res.get("success"):
                    plate_display = f"ID:{tid} | {c_res['final_plate']}"
                    if tid not in logged_vehicles:
                        timestamp_sec = frame_idx / fps_source
                        logged_vehicles[tid] = {
                            "Track ID": tid,
                            "Biển số Chốt": c_res["final_plate"],
                            "Độ tin cậy": f"{c_res['consensus_confidence'] * 100:.1f}%",
                            "Frames Tích lũy": c_res["num_frames_used"],
                            "Đồng thuận": f"{c_res['agreement_rate'] * 100:.1f}%",
                            "Hợp lệ DFA": "✅ Hợp lệ" if c_res["is_valid_grammar"] else "❌ Vi phạm",
                            "Thời điểm": f"{timestamp_sec:.1f}s",
                        }

                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                cv2.rectangle(frame, (bbox[0], max(0, bbox[1] - 25)), (bbox[0] + len(plate_display) * 11, bbox[1]), (0, 255, 0), -1)
                cv2.putText(
                    frame,
                    plate_display,
                    (bbox[0] + 3, max(0, bbox[1] - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 0),
                    2,
                )

            # Tính toán FPS thực tế
            t_frame_end = time.perf_counter()
            instant_fps = 1.0 / max(t_frame_end - t_frame_start, 1e-4)

            # Tạo DataFrame lịch sử xe
            df_log = pd.DataFrame(list(logged_vehicles.values())) if logged_vehicles else pd.DataFrame(
                columns=["Track ID", "Biển số Chốt", "Độ tin cậy", "Frames Tích lũy", "Đồng thuận", "Hợp lệ DFA", "Thời điểm"]
            )

            status_str = f"Đang xử lý Frame {frame_idx}/{total_frames} ({frame_idx/max(total_frames, 1)*100:.1f}%) | Xe ghi nhận: {len(logged_vehicles)}"
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            yield frame_rgb, df_log, status_str, round(instant_fps, 1)

        cap.release()
        logger.info("Hoàn tất xử lý video: %s frames.", frame_idx)

    def process_live_frame(
        self,
        frame_input: np.ndarray,
        conf_threshold: float = 0.40,
        use_grammar: bool = True,
    ) -> Tuple[np.ndarray, float, str]:
        """
        Xử lý từng frame cho luồng Webcam hoặc Camera RTSP trực tiếp:
        Đo đạc FPS thời gian thực và độ trễ từng mắt xích.
        """
        if frame_input is None:
            return np.zeros((384, 384, 3), dtype=np.uint8), 0.0, "Không có tín hiệu camera"

        img_bgr = cv2.cvtColor(frame_input, cv2.COLOR_RGB2BGR) if frame_input.shape[-1] == 3 else frame_input.copy()

        t0 = time.perf_counter()
        detections = self.pipeline.predict(
            img_bgr,
            beam_width=3,
            use_grammar=use_grammar,
            conf_threshold=conf_threshold,
        )
        t1 = time.perf_counter()

        elapsed_ms = (t1 - t0) * 1000.0
        fps = 1000.0 / max(elapsed_ms, 1e-3)

        annotated_bgr = self.pipeline.draw_predictions(img_bgr, detections)

        # Vẽ đồng hồ đo FPS lên góc trái khung hình
        cv2.putText(
            annotated_bgr,
            f"FPS: {fps:.1f} | Latency: {elapsed_ms:.1f}ms | Plates: {len(detections)}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        stats_text = f"Độ trễ toàn trình: **{elapsed_ms:.1f} ms** | FPS: **{fps:.1f}** | Số biển phát hiện: **{len(detections)}**"

        return annotated_rgb, round(fps, 1), stats_text
