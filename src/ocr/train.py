"""
FAST-PLATE-OCR-VIETNAMESE: Two-Stage Transfer Learning Trainer (Bước 3.2)
Huấn luyện mô hình CCT hai pha:
- Pha 1 (Pre-training): Huấn luyện trên dữ liệu giả lập (Synthetic Engine) để học biểu diễn ký tự đa phông, đa màu.
- Pha 2 (Fine-tuning): Tinh chỉnh trên tập dữ liệu thực tế (Kaggle Dataset) với kỹ thuật Screw Cutout, Motion Blur, Noise.
- Tối ưu hóa Character Error Rate (CER < 1%) và Full Match Accuracy (FMA).
"""

import os
import sys
import time
import argparse
from typing import Dict, Any, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Đảm bảo đường dẫn gốc dự án
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.ocr.cct_adapter import CCTPlateVietnamModel
from src.ocr.dataset import RealPlateOCRDataset, SyntheticPlateOCRDataset
from src.ocr.metrics import compute_cer, compute_accuracy, evaluate_predictions


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    label_smoothing: float = 0.05,
) -> float:
    """
    Huấn luyện 1 epoch trên các batch dữ liệu.
    """
    model.train()
    total_loss = 0.0
    total_batches = 0

    for images, targets, _ in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(images)  # [B, 10, num_classes]

        B, slots, num_classes = logits.shape
        loss = F.cross_entropy(
            logits.view(-1, num_classes),
            targets.view(-1),
            label_smoothing=label_smoothing,
        )

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    return total_loss / max(1, total_batches)


def evaluate_ocr(
    model: nn.Module,
    loader: DataLoader,
    dataset: RealPlateOCRDataset,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Đánh giá mô hình trên tập kiểm định: tính Loss, CER và Exact Match Accuracy.
    """
    model.eval()
    total_loss = 0.0
    total_batches = 0

    all_preds: List[str] = []
    all_gts: List[str] = []

    with torch.no_grad():
        for images, targets, raw_labels in loader:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            B, slots, num_classes = logits.shape
            loss = F.cross_entropy(logits.view(-1, num_classes), targets.view(-1))
            total_loss += loss.item()
            total_batches += 1

            pred_indices = logits.argmax(dim=-1).cpu().tolist()
            for idx_list, gt in zip(pred_indices, raw_labels):
                pred_str = dataset.decode_indices(idx_list)
                all_preds.append(pred_str)
                all_gts.append(gt)

    metrics = evaluate_predictions(all_preds, all_gts)
    metrics["val_loss"] = round(total_loss / max(1, total_batches), 4)
    return metrics


def run_two_stage_training(
    config_path: str = "configs/plate_config_vietnam.yaml",
    train_csv: str = "data/kaggle_dataset/train.csv",
    val_csv: str = "data/kaggle_dataset/val.csv",
    weights_dir: str = "weights",
    stage: str = "both",
    epochs_stage1: int = 3,
    epochs_stage2: int = 5,
    batch_size: int = 32,
    lr_stage1: float = 1e-3,
    lr_stage2: float = 3e-4,
    synthetic_samples: int = 2000,
    max_real_train: Optional[int] = None,
    max_real_val: Optional[int] = None,
    device_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Điều phối quy trình Two-Stage Transfer Learning.
    """
    os.makedirs(weights_dir, exist_ok=True)

    if device_name is not None:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[THIẾT BỊ HUẤN LUYỆN]: {device}")

    # Khởi tạo mô hình CCT
    model = CCTPlateVietnamModel(config_path=config_path).to(device)
    stage1_weights = os.path.join(weights_dir, "cct_stage1_synthetic.pt")
    final_weights = os.path.join(weights_dir, "fast_plate_ocr_vn.pt")

    # =========================================================================
    # PHA 1: TIỀN HUẤN LUYỆN TRÊN DỮ LIỆU GIẢ LẬP (PRE-TRAINING)
    # =========================================================================
    if stage in ("1", "both"):
        print("\n" + "=" * 65)
        print(">>> BẮT ĐẦU PHA 1: TIỀN HUẤN LUYỆN TRÊN DỮ LIỆU GIẢ LẬP (STAGE 1)")
        print(f"-> Số mẫu giả lập: {synthetic_samples:,} mẫu | Số Epochs: {epochs_stage1} | LR: {lr_stage1}")
        print("=" * 65)

        syn_dataset = SyntheticPlateOCRDataset(
            num_samples=synthetic_samples,
            config_path=config_path,
        )
        syn_loader = DataLoader(
            syn_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
        )

        optimizer_s1 = AdamW(model.parameters(), lr=lr_stage1, weight_decay=1e-4)
        scheduler_s1 = CosineAnnealingLR(optimizer_s1, T_max=epochs_stage1, eta_min=1e-5)

        for epoch in range(1, epochs_stage1 + 1):
            t0 = time.time()
            loss = train_one_epoch(model, syn_loader, optimizer_s1, device)
            scheduler_s1.step()
            dt = time.time() - t0
            print(f"[*] Stage 1 - Epoch [{epoch}/{epochs_stage1}] - Train Loss: {loss:.4f} - LR: {scheduler_s1.get_last_lr()[0]:.6f} ({dt:.1f}s)")

        # Lưu checkpoint Stage 1
        torch.save(model.state_dict(), stage1_weights)
        print(f"[OK] Đã lưu trọng số Stage 1 tại: {stage1_weights}")

    # =========================================================================
    # PHA 2: TINH CHỈNH TRÊN DỮ LIỆU THỰC TẾ (FINE-TUNING)
    # =========================================================================
    if stage in ("2", "both"):
        print("\n" + "=" * 65)
        print(">>> BẮT ĐẦU PHA 2: TINH CHỈNH TRÊN DỮ LIỆU THỰC TẾ & LÀM GIÀU (STAGE 2)")
        print(f"-> Tập thực tế: {train_csv} | Kiểm định: {val_csv}")
        print(f"-> Số Epochs: {epochs_stage2} | LR: {lr_stage2} | Augment: Screw Cutout + Motion Blur")
        print("=" * 65)

        # Nạp trọng số từ Stage 1 nếu chưa có trong RAM
        if stage == "2" and os.path.exists(stage1_weights):
            print(f"-> Nạp trọng số nền tảng từ Stage 1: {stage1_weights}")
            model.load_state_dict(torch.load(stage1_weights, map_location=device))

        val_dataset = RealPlateOCRDataset(
            csv_path=val_csv,
            config_path=config_path,
            is_train=False,
            max_samples=max_real_val,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        train_dataset = RealPlateOCRDataset(
            csv_path=train_csv,
            config_path=config_path,
            is_train=True,
            max_samples=max_real_train,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
        )

        optimizer_s2 = AdamW(model.parameters(), lr=lr_stage2, weight_decay=1e-4)
        scheduler_s2 = CosineAnnealingLR(optimizer_s2, T_max=epochs_stage2, eta_min=1e-6)

        best_cer = float("inf")
        best_metrics: Dict[str, Any] = {}

        for epoch in range(1, epochs_stage2 + 1):
            t0 = time.time()
            train_loss = train_one_epoch(model, train_loader, optimizer_s2, device)
            scheduler_s2.step()
            dt_train = time.time() - t0

            # Đánh giá sau mỗi epoch
            val_res = evaluate_ocr(model, val_loader, val_dataset, device)
            cer = val_res["cer"]
            acc = val_res["accuracy"]
            v_loss = val_res["val_loss"]

            print(
                f"[*] Stage 2 - Epoch [{epoch}/{epochs_stage2}] | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {v_loss:.4f} | "
                f"CER: {cer * 100:.2f}% | Exact Match: {acc * 100:.2f}% ({dt_train:.1f}s)"
            )

            # Lưu mô hình có CER tốt nhất
            if cer < best_cer:
                best_cer = cer
                best_metrics = val_res
                torch.save(model.state_dict(), final_weights)
                print(f"    --> [LƯU MÔ HÌNH TỐI ƯU] Trọng số xuất sắc nhất: {final_weights} (CER: {cer*100:.2f}%)")

        print("\n" + "=" * 65)
        print(f"[HOÀN THÀNH HUẤN LUYỆN HAI PHA] Trọng số lưu tại: {final_weights}")
        print(f"-> KẾT QUẢ ĐÁNH GIÁ TỐI ƯU TRÊN TẬP THỰC TẾ:")
        print(f"   * Character Error Rate (CER): {best_metrics.get('cer_percent', 0.0)}%")
        print(f"   * Full Match Accuracy (FMA): {best_metrics.get('accuracy_percent', 0.0)}%")
        print(f"   * Số mẫu khớp 100%: {best_metrics.get('exact_matches', 0)} / {best_metrics.get('total_samples', 0)}")
        print("=" * 65)
        return best_metrics

    # =========================================================================
    # CHẾ ĐỘ ĐÁNH GIÁ (EVALUATION ONLY)
    # =========================================================================
    if stage == "eval":
        if os.path.exists(final_weights):
            print(f"-> Nạp trọng số mô hình đã huấn luyện: {final_weights}")
            model.load_state_dict(torch.load(final_weights, map_location=device))
        val_dataset = RealPlateOCRDataset(csv_path=val_csv, config_path=config_path, is_train=False, max_samples=max_real_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        metrics = evaluate_ocr(model, val_loader, val_dataset, device)
        print("\n[KẾT QUẢ ĐÁNH GIÁ]:", metrics)
        return metrics

    return {}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện hai pha Transfer Learning cho fast-plate-ocr-vn")
    parser.add_argument("--stage", type=str, default="both", choices=["1", "2", "both", "eval"])
    parser.add_argument("--epochs1", type=int, default=3, help="Số epochs Pha 1")
    parser.add_argument("--epochs2", type=int, default=5, help="Số epochs Pha 2")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--synthetic_samples", type=int, default=1500)
    parser.add_argument("--max_train", type=int, default=None)
    parser.add_argument("--max_val", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    run_two_stage_training(
        stage=args.stage,
        epochs_stage1=args.epochs1,
        epochs_stage2=args.epochs2,
        batch_size=args.batch_size,
        synthetic_samples=args.synthetic_samples,
        max_real_train=args.max_train,
        max_real_val=args.max_val,
        device_name=args.device,
    )
