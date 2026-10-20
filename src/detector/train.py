"""
FAST-PLATE-OCR-VIETNAMESE: Training Pipeline for Anchor-Free Plate & 4-Corner Detector
Quy trình huấn luyện mạng định vị biển số và trích xuất 4 điểm góc, tự động xuất trọng số .pt và ONNX.
"""

import os
import sys
import time
import argparse
from typing import Dict, Any, Tuple, Optional, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

try:
    import torch
    from torch.utils.data import DataLoader
    from src.detector.dataset import VehiclePlateCornerDataset
    from src.detector.model import PlateCornerDetector
    from src.detector.loss import PlateDetectorMultiTaskLoss
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def collate_fn(batch):
    """Gom cụm batch dữ liệu tùy biến cho bài toán phát hiện đối tượng đa kích thước."""
    images = torch.stack([item[0] for item in batch], dim=0)
    targets = [item[1] for item in batch]
    return images, targets


def build_training_targets(targets, grid_size: Tuple[int, int], device: str) -> Dict[str, Any]:
    """Chuyển đổi nhãn đối tượng thành các bản đồ đặc trưng (Target Feature Maps) tại tỷ lệ 1/8."""
    batch_size = len(targets)
    grid_h, grid_w = grid_size

    conf_map = torch.zeros((batch_size, 1, grid_h, grid_w), dtype=torch.float32, device=device)
    bbox_map = torch.zeros((batch_size, 4, grid_h, grid_w), dtype=torch.float32, device=device)
    corner_map = torch.zeros((batch_size, 8, grid_h, grid_w), dtype=torch.float32, device=device)
    layout_map = torch.zeros((batch_size, grid_h, grid_w), dtype=torch.long, device=device)

    for b in range(batch_size):
        t = targets[b]
        boxes = t["boxes"]
        corners = t["corners"]
        labels = t["labels"]

        for i in range(len(boxes)):
            box = boxes[i]
            # Tâm biển số chuẩn hóa [0, 1]
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0
            bw = box[2] - box[0]
            bh = box[3] - box[1]

            # Vị trí trên grid 1/8
            gx = int(torch.clamp(cx * grid_w, 0, grid_w - 1))
            gy = int(torch.clamp(cy * grid_h, 0, grid_h - 1))

            conf_map[b, 0, gy, gx] = 1.0
            bbox_map[b, 0, gy, gx] = cx - (gx + 0.5) / grid_w
            bbox_map[b, 1, gy, gx] = cy - (gy + 0.5) / grid_h
            bbox_map[b, 2, gy, gx] = bw
            bbox_map[b, 3, gy, gx] = bh

            # 4 điểm góc so với tâm
            corn = corners[i] # [4, 2]
            for c_idx in range(4):
                corner_map[b, c_idx * 2, gy, gx] = corn[c_idx, 0] - cx
                corner_map[b, c_idx * 2 + 1, gy, gx] = corn[c_idx, 1] - cy

            layout_map[b, gy, gx] = labels[i]

    return {
        "conf_map": conf_map,
        "bbox_map": bbox_map,
        "corner_map": corner_map,
        "layout_map": layout_map,
    }


def train_detector(
    json_path: str = "data/real_traffic_dataset/annotations_full.json",
    weights_dir: str = "weights",
    epochs: int = 15,
    batch_size: int = 8,
    lr: float = 1e-3,
    device: Optional[str] = None,
):
    """
    Tiến trình huấn luyện mô hình định vị biển số và xuất trọng số.
    """
    if not HAS_TORCH:
        print("[LỖI] Cần cài đặt PyTorch để thực hiện huấn luyện mô hình.")
        return

    os.makedirs(weights_dir, exist_ok=True)
    best_weights_path = os.path.join(weights_dir, "best_detector.pt")
    onnx_path = os.path.join(weights_dir, "plate_detector.onnx")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'=' * 65}")
    print(f"BẮT ĐẦU HUẤN LUYỆN DETECTOR 4 ĐIỂM GÓC BIỂN SỐ")
    print(f"Thiết bị tính toán: {device.upper()}")
    print(f"Số epochs: {epochs} | Batch size: {batch_size} | Learning rate: {lr}")
    print(f"Tệp nhãn nguồn: {json_path}")
    print(f"{'=' * 65}\n")

    # 1. Nạp Dataset & DataLoader
    dataset = VehiclePlateCornerDataset(json_path=json_path, target_size=(384, 384))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # 2. Khởi tạo Mô hình & Loss Function
    model = PlateCornerDetector(in_channels=3, base_channels=32).to(device)
    criterion = PlateDetectorMultiTaskLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_loss = float("inf")
    start_time = time.time()

    grid_h = 384 // 8
    grid_w = 384 // 8

    # 3. Vòng lặp huấn luyện (Training Loop)
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for images, targets in dataloader:
            images = images.to(device)
            target_maps = build_training_targets(targets, (grid_h, grid_w), device)

            optimizer.zero_grad()
            predictions = model(images)
            loss_dict = criterion(predictions, target_maps)
            loss = loss_dict["total_loss"]

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(1, num_batches)
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Loss: {avg_loss:.4f} - LR: {scheduler.get_last_lr()[0]:.6f}")

        # Lưu checkpoint tốt nhất
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_loss": best_loss,
                "input_size": (384, 384),
            }, best_weights_path)

    elapsed = time.time() - start_time
    print(f"\n[HOÀN THÀNH HUẤN LUYỆN] Thời gian: {elapsed:.2f}s | Best Loss: {best_loss:.4f}")
    print(f" -> Trọng số tốt nhất đã lưu tại: {best_weights_path}")

    # 4. Xuất mô hình sang định dạng ONNX Runtime phục vụ suy luận thời gian thực
    try:
        model.eval()
        dummy_input = torch.randn(1, 3, 384, 384, device=device)
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=["image_input"],
            output_names=["confidence", "bbox", "corners", "layout"],
            dynamic_axes={"image_input": {0: "batch_size"}},
            opset_version=14,
        )
        print(f" -> Mô hình ONNX đã xuất thành công: {onnx_path}")
    except Exception as e:
        print(f" [Lưu ý xuất ONNX]: {e}")

    print(f"{'=' * 65}\n")
    return best_weights_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Vehicle & Plate Corner Detector")
    parser.add_argument("--json-path", type=str, default="data/real_traffic_dataset/annotations_full.json")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train_detector(
        json_path=args.json_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
