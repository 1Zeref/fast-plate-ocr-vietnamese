"""
FAST-PLATE-OCR-VIETNAMESE: Demonstration & Evaluation CLI Tool
Chạy nhận dạng biển số xe trên các tệp ảnh và video trong thư mục samples/
và xuất ảnh chú thích trực quan kèm báo cáo tổng hợp JSON vào thư mục output/.
"""

import os
import sys
import time
import json
import argparse
import logging
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import cv2
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.pipeline import VietnamesePlateOCRPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ALPR_Eval")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def evaluate_directory(
    samples_dir: str = "samples",
    output_dir: str = "output",
    use_onnx: bool = True,
    device: str = "auto",
    conf_threshold: float = 0.40,
    min_ocr_conf: float = 0.15,
    min_overall_conf: float = 0.05,
) -> Dict[str, Any]:
    """Chạy toàn bộ pipeline trên tất cả các tệp trong samples_dir và xuất sang output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    if device == "auto":
        dev = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"
    else:
        dev = device

    logger.info("Khởi tạo Pipeline trên thiết bị: [%s] (ONNX Engine=%s, Conf Thresh=%.2f)...", dev, use_onnx, conf_threshold)
    pipeline = VietnamesePlateOCRPipeline(
        device=dev,
        conf_threshold=conf_threshold,
        use_onnx=use_onnx,
    )

    if not os.path.exists(samples_dir):
        logger.warning("Thư mục mẫu [%s] chưa tồn tại. Đang tạo mới...", samples_dir)
        os.makedirs(samples_dir, exist_ok=True)
        return {"processed_files": 0, "results": []}

    files = sorted(os.listdir(samples_dir))
    if not files:
        logger.warning("Không có tệp nào trong [%s]. Vui lòng sao chép ảnh/video vào thư mục này.", samples_dir)
        return {"processed_files": 0, "results": []}

    results_summary = []
    total_latency_ms = 0.0

    print("\n" + "=" * 70)
    print("BẮT ĐẦU PHÂN TÍCH DỮ LIỆU THỰC NGHIỆM (ALPR VIETNAM PIPELINE)")
    print("=" * 70)

    for fname in files:
        fpath = os.path.join(samples_dir, fname)
        ext = os.path.splitext(fname)[1].lower()

        if ext in IMAGE_EXTS:
            logger.info("Đang xử lý ảnh: %s", fname)
            img = cv2.imread(fpath)
            if img is None:
                continue

            t0 = time.perf_counter()
            dets = pipeline.predict(
                img,
                beam_width=5,
                use_grammar=True,
                conf_threshold=conf_threshold,
                min_ocr_conf=min_ocr_conf,
                min_overall_conf=min_overall_conf,
            )
            t1 = time.perf_counter()
            elapsed_ms = (t1 - t0) * 1000.0
            total_latency_ms += elapsed_ms

            # Vẽ bounding box và biển số lên ảnh
            annotated_img = pipeline.draw_predictions(img, dets)
            out_img_name = f"annotated_{os.path.splitext(fname)[0]}.jpg"
            out_img_path = os.path.join(output_dir, out_img_name)
            cv2.imwrite(out_img_path, annotated_img)

            plates_found = [d["plate_text"] for d in dets]
            confs = [d["ocr_confidence"] for d in dets]

            file_record = {
                "filename": fname,
                "type": "IMAGE",
                "latency_ms": round(elapsed_ms, 2),
                "num_plates_detected": len(dets),
                "plates": plates_found,
                "confidences": confs,
                "annotated_output": out_img_name,
                "detections": dets,
            }
            results_summary.append(file_record)

            print(f"-> [{fname}] Độ trễ: {elapsed_ms:.1f}ms | Phát hiện: {plates_found} | Ảnh xuất: {out_img_name}")

        elif ext in VIDEO_EXTS:
            logger.info("Đang xử lý video: %s (RTSP / Video Stream)", fname)
            cap = cv2.VideoCapture(fpath)
            if not cap.isOpened():
                continue

            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            out_vid_name = f"annotated_{os.path.splitext(fname)[0]}.mp4"
            out_vid_path = os.path.join(output_dir, out_vid_name)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_vid_path, fourcc, fps, (width, height))

            frame_idx = 0
            vid_t0 = time.perf_counter()
            active_plates = set()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                tracked_res = pipeline.process_video_frame(frame)
                # Vẽ thông tin tracking lên frame
                for trk in tracked_res.get("active_tracks", []):
                    bbox = trk.get("plate_bbox", [0, 0, 0, 0])
                    tid = trk.get("track_id", 0)
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"ID: {tid}",
                        (bbox[0], max(0, bbox[1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )
                writer.write(frame)
                frame_idx += 1

            cap.release()
            writer.release()
            vid_t1 = time.perf_counter()
            vid_elapsed = vid_t1 - vid_t0

            # Tổng hợp nhận diện cho các xe
            consolidated = []
            for tid in pipeline.frame_buffer.buffers.keys():
                c_res = pipeline.consolidate_vehicle(tid)
                if c_res.get("success"):
                    consolidated.append(c_res)
                    active_plates.add(c_res["final_plate"])

            vid_record = {
                "filename": fname,
                "type": "VIDEO",
                "total_frames": frame_idx,
                "processing_time_sec": round(vid_elapsed, 2),
                "effective_fps": round(frame_idx / max(vid_elapsed, 1e-4), 1),
                "vehicles_identified": list(active_plates),
                "consolidated_results": consolidated,
                "annotated_output": out_vid_name,
            }
            results_summary.append(vid_record)
            print(f"-> [{fname}] {frame_idx} frames | Tốc độ: {vid_record['effective_fps']} FPS | Xe xác nhận: {list(active_plates)}")

    # Ghi nhận kết quả ra tệp JSON
    json_path = os.path.join(output_dir, "evaluation_summary.json")
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": dev,
        "use_onnx": use_onnx,
        "conf_threshold": conf_threshold,
        "total_processed_files": len(results_summary),
        "results": results_summary,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 70)
    print(f"HOÀN TẤT! Toàn bộ kết quả và ảnh chú thích đã được lưu tại: {output_dir}/")
    print(f"Báo cáo tổng hợp JSON: {json_path}")
    print("=" * 70)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Fast Plate OCR Vietnamese Evaluation Tool")
    parser.add_argument("--samples", type=str, default="samples", help="Thư mục chứa dữ liệu test")
    parser.add_argument("--output", type=str, default="output", help="Thư mục ghi kết quả")
    parser.add_argument("--use-onnx", action="store_true", default=True, help="Sử dụng ONNX Runtime tăng tốc")
    parser.add_argument("--no-onnx", dest="use_onnx", action="store_false", help="Sử dụng PyTorch thay vì ONNX")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Thiết bị thực thi")
    parser.add_argument("--conf-threshold", type=float, default=0.40, help="Ngưỡng tin cậy phát hiện biển số")
    parser.add_argument("--min-ocr-conf", type=float, default=0.15, help="Ngưỡng tin cậy tối thiểu cho chuỗi OCR")
    parser.add_argument("--min-overall-conf", type=float, default=0.05, help="Ngưỡng tin cậy kết hợp tối thiểu")

    args = parser.parse_args()
    evaluate_directory(
        samples_dir=args.samples,
        output_dir=args.output,
        use_onnx=args.use_onnx,
        device=args.device,
        conf_threshold=args.conf_threshold,
        min_ocr_conf=args.min_ocr_conf,
        min_overall_conf=args.min_overall_conf,
    )


if __name__ == "__main__":
    main()
