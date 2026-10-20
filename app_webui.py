"""
FAST-PLATE-OCR-VIETNAMESE: WebUI Testing & Visualization Application
Giao diện WebUI tương tác đa chức năng xây dựng trên Gradio:
- Tab 1: Kiểm thử Ảnh tĩnh (Single / Batch Image Mode)
- Tab 2: Kiểm thử Video Tải lên (Video Tracking & Consensus Mode)
- Tab 3: Giám sát Camera Trực tiếp (Live Webcam / RTSP IP Camera)
- Tab 4: Bóc tách Chẩn đoán Kỹ thuật (Pipeline Inspector / Explainable AI)
"""

import os
import sys
import time
import argparse
import logging
from typing import Dict, Any, List, Tuple, Optional

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import cv2
import numpy as np
import pandas as pd
import gradio as gr

from src.service import ALPRService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("WebUI")

# Khởi tạo dịch vụ
service = ALPRService.get_instance(use_onnx=True)


# ==============================================================================
# HÀM XỬ LÝ SỰ KIỆN CHO CÁC TAB
# ==============================================================================

def on_predict_image(
    image,
    conf_thresh: float,
    beam_width: int,
    use_grammar: bool,
):
    """Xử lý Tab 1: Nhận diện ảnh tĩnh."""
    if image is None:
        return None, "⚠️ Vui lòng tải lên một hình ảnh trước khi nhận diện.", pd.DataFrame()

    res = service.process_image(
        image_input=image,
        conf_threshold=conf_thresh,
        beam_width=int(beam_width),
        use_grammar=use_grammar,
    )

    status_markdown = (
        f"### ⚡ Thống Kê Xử Lý:\n"
        f"- **Thời gian xử lý:** `{res['latency_ms']} ms` (~{1000.0/max(res['latency_ms'], 1e-3):.1f} FPS)\n"
        f"- **Số lượng biển số phát hiện:** `{res['num_plates']}`\n"
        f"- **Động cơ suy luận:** `ONNX Runtime Accelerated`\n"
        f"- **Ràng buộc Ngữ pháp DFA:** `{'Bật (Thông tư 24/2023)' if use_grammar else 'Tắt (Greedy)'}`\n"
    )

    return res["annotated_image"], status_markdown, res["dataframe"]


def on_inspect_pipeline(
    image,
    conf_thresh: float,
    beam_width: int,
):
    """Xử lý Tab 4: Bóc tách 5 công đoạn trung gian."""
    if image is None:
        return None, None, None, None, "⚠️ Vui lòng tải lên hình ảnh để bóc tách."

    insp = service.inspect_pipeline(
        image_input=image,
        conf_threshold=conf_thresh,
        beam_width=int(beam_width),
        plate_idx=0,
    )

    return (
        insp["step1_crop"],
        insp["step2_corners"],
        insp["step3_warped"],
        insp["step4_virtual_strip"],
        insp["explanation"],
    )


def on_predict_video_stream(
    video_path,
    frame_skip: int,
    consensus_k: int,
    conf_thresh: float,
):
    """Xử lý Tab 2: Generator streaming video theo thời gian thực."""
    if not video_path:
        yield None, pd.DataFrame(), "⚠️ Vui lòng tải lên một tệp video.", 0.0
        return

    generator = service.process_video_generator(
        video_path=video_path,
        frame_skip=int(frame_skip),
        consensus_k=int(consensus_k),
        conf_threshold=conf_thresh,
    )

    for frame_rgb, df_log, status_str, current_fps in generator:
        yield frame_rgb, df_log, status_str, current_fps


def on_process_webcam_live(
    frame,
    conf_thresh: float,
    use_grammar: bool,
):
    """Xử lý Tab 3: Webcam frame streaming."""
    if frame is None:
        return None, 0.0, "Không nhận được tín hiệu camera"

    annotated_rgb, fps, stats_text = service.process_live_frame(
        frame_input=frame,
        conf_threshold=conf_thresh,
        use_grammar=use_grammar,
    )
    return annotated_rgb, fps, stats_text


def on_connect_rtsp(rtsp_url: str, conf_thresh: float):
    """Xử lý Tab 3: Thử kết nối và đọc 1 frame từ camera RTSP."""
    if not rtsp_url or not rtsp_url.startswith("rtsp://"):
        return None, "❌ URL RTSP không hợp lệ. Định dạng chuẩn: `rtsp://<user>:<pass>@<ip>:<port>/<path>`"

    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        return None, f"❌ Không thể kết nối tới luồng RTSP: `{rtsp_url}`. Vui lòng kiểm tra địa chỉ mạng và quyền truy cập."

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return None, "❌ Đã kết nối nhưng không đọc được frame nào từ luồng RTSP."

    annotated_rgb, fps, stats = service.process_live_frame(frame, conf_threshold=conf_thresh)
    return annotated_rgb, f"✅ Kết nối RTSP thành công! {stats}"


# ==============================================================================
# XÂY DỰNG GIAO DIỆN GRADIO BLOCKS
# ==============================================================================

CUSTOM_CSS = """
.container { max-width: 1200px; margin: auto; }
.header-badge { background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); color: white; padding: 18px 25px; border-radius: 12px; margin-bottom: 20px; }
.tab-content { padding: 15px; }
.metric-box { background: #f8f9fa; border-left: 4px solid #2a5298; padding: 12px; border-radius: 6px; margin-bottom: 10px; }
"""

def build_webui() -> gr.Blocks:
    with gr.Blocks(title="Fast Plate OCR Vietnamese - ALPR WebUI") as demo:
        # Header Banner
        gr.HTML("""
        <div class="header-badge">
            <h1 style="margin: 0; font-size: 26px;">🇻🇳 Fast Plate OCR Vietnamese — Hệ Thống ALPR Thương Mại</h1>
            <p style="margin: 6px 0 0 0; opacity: 0.9; font-size: 14px;">
                Định vị 4 góc • Nắn phẳng phối cảnh • CCT Transformer OCR • Lọc Cú pháp DFA (Thông tư 24/2023/TT-BCA) • Bám vết ByteTrack & Đồng thuận Đa khung hình
            </p>
        </div>
        """)

        with gr.Tabs() as tabs:
            # ------------------------------------------------------------------
            # TAB 1: KIỂM THỬ ẢNH TĨNH (IMAGE MODE)
            # ------------------------------------------------------------------
            with gr.Tab("🖼️ 1. Kiểm Thử Ảnh Tĩnh", id="tab_image"):
                gr.Markdown("### Nhận diện biển số trên ảnh đơn lẻ hoặc ảnh giao thông toàn cảnh")
                with gr.Row():
                    # Cột điều khiển bên trái
                    with gr.Column(scale=1):
                        img_input = gr.Image(label="Ảnh Đầu Vào", type="numpy")
                        
                        gr.Markdown("#### ⚙️ Tham Số Nhận Diện")
                        slider_conf = gr.Slider(0.10, 0.95, value=0.40, step=0.05, label="Detection Confidence Threshold")
                        slider_beam = gr.Slider(1, 10, value=5, step=1, label="Beam Width (Độ rộng Beam Search)")
                        check_grammar = gr.Checkbox(value=True, label="Kích hoạt Ràng buộc Ngữ pháp DFA (Thông tư 24)")

                        btn_predict_img = gr.Button("🔍 Nhận Diện Biển Số", variant="primary", size="lg")

                        gr.Examples(
                            examples=[
                                "samples/car_1line_sample.jpg",
                                "samples/motorcycle_2line_sample.jpg",
                                "samples/traffic_scene_sample.jpg",
                            ],
                            inputs=img_input,
                            label="📁 Ảnh Thử Nghiệm Mẫu",
                        )

                    # Cột kết quả bên phải
                    with gr.Column(scale=1):
                        img_output = gr.Image(label="Ảnh Kết Quả Chú Thích (Annotated)", type="numpy")
                        txt_metrics = gr.Markdown("Vui lòng tải ảnh và bấm **Nhận Diện Biển Số**.")

                with gr.Row():
                    df_results = gr.Dataframe(
                        label="📋 Bảng Kết Quả Chi Tiết & Kiểm Chuẩn Pháp Lý",
                        interactive=False,
                    )

                btn_predict_img.click(
                    fn=on_predict_image,
                    inputs=[img_input, slider_conf, slider_beam, check_grammar],
                    outputs=[img_output, txt_metrics, df_results],
                )

            # ------------------------------------------------------------------
            # TAB 2: KIỂM THỬ VIDEO & BÁM VẾT (VIDEO TRACKING MODE)
            # ------------------------------------------------------------------
            with gr.Tab("🎬 2. Kiểm Thử Video & Bám Vết", id="tab_video"):
                gr.Markdown("### Bám vết xe theo thời gian thực (ByteTrack) & Hợp nhất quyết định đa khung hình (Temporal Consensus)")
                with gr.Row():
                    with gr.Column(scale=1):
                        vid_input = gr.Video(label="Tải Lên Tệp Video (.mp4, .avi)")
                        
                        gr.Markdown("#### ⚙️ Tham Số Video & Tracking")
                        slider_skip = gr.Slider(1, 4, value=1, step=1, label="Frame Skip (Bước nhảy frame để tăng tốc)")
                        slider_consensus = gr.Slider(3, 8, value=5, step=1, label="Cửa sổ Đồng thuận (Consensus K Frames)")
                        slider_vid_conf = gr.Slider(0.10, 0.90, value=0.40, step=0.05, label="Detection Threshold")

                        btn_process_vid = gr.Button("▶ Bắt Đầu Xử Lý Video", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        vid_live_frame = gr.Image(label="Khung Hình Đang Xử Lý (Streaming Preview)", type="numpy")
                        txt_vid_status = gr.Textbox(label="Tiến độ & Trạng thái", value="Chờ video...")
                        num_vid_fps = gr.Number(label="FPS Thời Gian Thực", precision=1)

                with gr.Row():
                    df_traffic_log = gr.Dataframe(
                        label="🚗 Nhật Ký Phương Tiện Giao Thông (Vehicle Traffic Consensus Log)",
                        interactive=False,
                    )

                btn_process_vid.click(
                    fn=on_predict_video_stream,
                    inputs=[vid_input, slider_skip, slider_consensus, slider_vid_conf],
                    outputs=[vid_live_frame, df_traffic_log, txt_vid_status, num_vid_fps],
                )

            # ------------------------------------------------------------------
            # TAB 3: GIÁM SÁT CAMERA TRỰC TIẾP (LIVE RTSP / WEBCAM)
            # ------------------------------------------------------------------
            with gr.Tab("📹 3. Giám Sát Camera Trực Tiếp", id="tab_live"):
                gr.Markdown("### Kết nối luồng Camera RTSP IP hoặc Webcam thời gian thực")
                
                with gr.Tab("Webcam Máy Tính"):
                    with gr.Row():
                        with gr.Column():
                            webcam_input = gr.Image(sources=["webcam"], streaming=True, label="Webcam Trực Tiếp")
                            slider_webcam_conf = gr.Slider(0.10, 0.90, value=0.40, step=0.05, label="Detection Threshold")
                            check_webcam_grammar = gr.Checkbox(value=True, label="DFA Grammar Constraint")
                        with gr.Column():
                            webcam_output = gr.Image(label="Kết Quả Luồng Trực Tiếp", type="numpy")
                            num_webcam_fps = gr.Number(label="FPS Đo Được", precision=1)
                            md_webcam_stats = gr.Markdown("Sẵn sàng stream...")

                    webcam_input.stream(
                        fn=on_process_webcam_live,
                        inputs=[webcam_input, slider_webcam_conf, check_webcam_grammar],
                        outputs=[webcam_output, num_webcam_fps, md_webcam_stats],
                    )

                with gr.Tab("Camera IP (Luồng RTSP)"):
                    with gr.Row():
                        with gr.Column():
                            txt_rtsp_url = gr.Textbox(
                                label="Địa Chỉ URL Luồng RTSP",
                                placeholder="rtsp://admin:password@192.168.1.100:554/stream1",
                                value="rtsp://localhost:8554/live",
                            )
                            slider_rtsp_conf = gr.Slider(0.10, 0.90, value=0.40, step=0.05, label="Detection Threshold")
                            btn_connect_rtsp = gr.Button("📡 Kết Nối & Kiểm Tra Frame RTSP", variant="primary")
                        with gr.Column():
                            rtsp_preview = gr.Image(label="Ảnh Preview từ Camera RTSP", type="numpy")
                            txt_rtsp_status = gr.Markdown("Nhập URL và bấm Kết Nối...")

                    btn_connect_rtsp.click(
                        fn=on_connect_rtsp,
                        inputs=[txt_rtsp_url, slider_rtsp_conf],
                        outputs=[rtsp_preview, txt_rtsp_status],
                    )

            # ------------------------------------------------------------------
            # TAB 4: BÓC TÁCH CHẨN ĐOÁN KỸ THUẬT (EXPLAINABLE AI INSPECTOR)
            # ------------------------------------------------------------------
            with gr.Tab("🔬 4. Bóc Tách Chẩn Đoán Kỹ Thuật", id="tab_inspector"):
                gr.Markdown("### Quan sát chi tiết từng mắt xích trung gian: Phát hiện góc $\\to$ Nắn phẳng $\\to$ Ghép dòng ảo $\\to$ Giải mã DFA")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        insp_img_input = gr.Image(label="Ảnh Cần Chẩn Đoán", type="numpy")
                        slider_insp_conf = gr.Slider(0.10, 0.90, value=0.40, step=0.05, label="Detection Threshold")
                        slider_insp_beam = gr.Slider(1, 10, value=5, step=1, label="Beam Width")
                        btn_inspect = gr.Button("🔬 Bóc Tách Toàn Bộ Công Đoạn", variant="primary", size="lg")

                        gr.Examples(
                            examples=[
                                "samples/car_1line_sample.jpg",
                                "samples/motorcycle_2line_sample.jpg",
                            ],
                            inputs=insp_img_input,
                            label="📁 Mẫu Chẩn Đoán Điển Hình",
                        )

                    with gr.Column(scale=2):
                        gr.Markdown("#### 🔍 Chuỗi Chuyển Đổi Hình Học (Visual Pipeline)")
                        with gr.Row():
                            insp_step1 = gr.Image(label="[1] Vùng Crop Xe / Biển Gốc", type="numpy")
                            insp_step2 = gr.Image(label="[2] Tứ Giác 4 Điểm Góc (TL/TR/BR/BL)", type="numpy")
                        with gr.Row():
                            insp_step3 = gr.Image(label="[3] Nắn Phẳng Homography (64x256)", type="numpy")
                            insp_step4 = gr.Image(label="[4] Dải Ảnh Ghép Dòng Ảo (OCR Input)", type="numpy")

                with gr.Row():
                    md_insp_explanation = gr.Markdown("Vui lòng tải ảnh và bấm **Bóc Tách Toàn Bộ Công Đoạn** để xem báo cáo.")

                btn_inspect.click(
                    fn=on_inspect_pipeline,
                    inputs=[insp_img_input, slider_insp_conf, slider_insp_beam],
                    outputs=[insp_step1, insp_step2, insp_step3, insp_step4, md_insp_explanation],
                )

        # Footer
        gr.Markdown("""
        ---
        <div style="text-align: center; color: #666; font-size: 13px;">
            Fast Plate OCR Vietnamese © 2026 | Tuân thủ Quy chuẩn Biển số Thông tư 24/2023/TT-BCA & Thông tư 58/2020/TT-BCA
        </div>
        """)

    return demo


def main():
    parser = argparse.ArgumentParser(description="Fast Plate OCR Vietnamese WebUI Application")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Địa chỉ Host (mặc định: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7860, help="Cổng Port (mặc định: 7860)")
    parser.add_argument("--share", action="store_true", default=False, help="Tạo link public Gradio Share")
    args = parser.parse_args()

    app = build_webui()
    logger.info("Khởi động Gradio WebUI tại http://%s:%d ...", args.host, args.port)
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":
    main()
