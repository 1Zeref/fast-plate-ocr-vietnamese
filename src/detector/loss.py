"""
FAST-PLATE-OCR-VIETNAMESE: Multi-Task Loss Functions for Plate & Corner Detection
Tích hợp Wing Loss cho tọa độ 4 điểm góc, Focal BCE Loss cho tâm biển và Smooth L1 cho Bbox.
"""

import math
from typing import Dict, Any, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    nn = object
    F = object


class WingLoss(nn.Module if HAS_TORCH else object):
    """
    Wing Loss: Hàm mất mát chuyên dụng tối ưu hồi quy các điểm đặc trưng (Landmarks / Corner Keypoints).
    Độ dốc cao khi sai số nhỏ giúp các góc tứ giác lồi bám sát viền mép biển số chính xác đến từng pixel.
    """
    def __init__(self, omega: float = 10.0, epsilon: float = 2.0):
        super().__init__()
        self.omega = omega
        self.epsilon = epsilon
        self.c = omega - omega * math.log(1.0 + omega / epsilon) if HAS_TORCH else 0.0

    def forward(self, pred, target):
        delta = (pred - target).abs()
        small_mask = delta < self.omega
        loss = torch.zeros_like(delta)
        loss[small_mask] = self.omega * torch.log(1.0 + delta[small_mask] / self.epsilon)
        loss[~small_mask] = delta[~small_mask] - self.c
        return loss.mean()


class PlateDetectorMultiTaskLoss(nn.Module if HAS_TORCH else object):
    """
    Bộ hàm mất mát đa nhiệm (Multi-task Loss):
    Loss = w_conf * Loss_conf + w_bbox * Loss_bbox + w_corners * Loss_corners + w_layout * Loss_layout
    """
    def __init__(
        self,
        w_conf: float = 2.0,
        w_bbox: float = 1.0,
        w_corners: float = 3.0,
        w_layout: float = 0.5,
    ):
        super().__init__()
        self.w_conf = w_conf
        self.w_bbox = w_bbox
        self.w_corners = w_corners
        self.w_layout = w_layout
        if HAS_TORCH:
            self.wing_loss = WingLoss(omega=10.0, epsilon=2.0)
            self.smooth_l1 = nn.SmoothL1Loss()
            self.bce_loss = nn.BCELoss()
            self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, predictions: Dict[str, Any], targets: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tính toán tổng hàm mất mát dựa trên dự đoán từ mô hình và nhãn ground-truth.
        """
        if not HAS_TORCH:
            return {"total_loss": 0.0}

        pred_conf = predictions["confidence"]
        target_conf = targets.get("conf_map", torch.zeros_like(pred_conf))

        # 1. Objectness / Center Loss (CenterNet Modified Focal Loss)
        # Giúp triệt tiêu mất cân bằng âm bản cực lớn giữa tâm biển số và nền
        pos_mask = target_conf >= 0.7
        neg_mask = target_conf < 0.7
        alpha = 2.0
        pos_loss = -torch.pow(1.0 - pred_conf[pos_mask].clamp(min=1e-6, max=1.0 - 1e-6), alpha) * torch.log(pred_conf[pos_mask].clamp(min=1e-6))
        neg_loss = -torch.pow(pred_conf[neg_mask].clamp(min=1e-6, max=1.0 - 1e-6), alpha) * torch.log((1.0 - pred_conf[neg_mask]).clamp(min=1e-6)) * torch.pow(1.0 - target_conf[neg_mask], 4.0)
        num_pos = max(1.0, float(pos_mask.sum().item()))
        loss_conf = (pos_loss.sum() + neg_loss.sum()) / num_pos

        # 2. Bbox Loss (chỉ tính tại các điểm có biển số)
        pos_mask = target_conf > 0.5
        if pos_mask.sum() > 0:
            pred_bbox = predictions["bbox"]
            target_bbox = targets.get("bbox_map", torch.zeros_like(pred_bbox))
            loss_bbox = self.smooth_l1(pred_bbox[pos_mask.repeat(1, 4, 1, 1)], target_bbox[pos_mask.repeat(1, 4, 1, 1)])

            # 3. 4-Corner Keypoint Loss (Wing Loss)
            pred_corners = predictions["corners"]
            target_corners = targets.get("corner_map", torch.zeros_like(pred_corners))
            loss_corners = self.wing_loss(pred_corners[pos_mask.repeat(1, 8, 1, 1)], target_corners[pos_mask.repeat(1, 8, 1, 1)])

            # 4. Layout Loss
            pred_layout = predictions["layout"]
            target_layout = targets.get("layout_map", torch.zeros((pred_layout.shape[0], pred_layout.shape[2], pred_layout.shape[3]), dtype=torch.long, device=pred_layout.device))
            loss_layout = self.ce_loss(pred_layout.permute(0, 2, 3, 1)[pos_mask.squeeze(1)], target_layout[pos_mask.squeeze(1)])
        else:
            loss_bbox = torch.tensor(0.0, device=pred_conf.device)
            loss_corners = torch.tensor(0.0, device=pred_conf.device)
            loss_layout = torch.tensor(0.0, device=pred_conf.device)

        total_loss = (
            self.w_conf * loss_conf
            + self.w_bbox * loss_bbox
            + self.w_corners * loss_corners
            + self.w_layout * loss_layout
        )

        return {
            "total_loss": total_loss,
            "loss_conf": loss_conf,
            "loss_bbox": loss_bbox,
            "loss_corners": loss_corners,
            "loss_layout": loss_layout,
        }
