"""
FAST-PLATE-OCR-VIETNAMESE: Anchor-Free 4-Corner Plate & Vehicle Detector Model
Kiến trúc mạng nơ-ron tích chập đa tác vụ định vị tứ giác lồi 4 góc biển số kèm chỉ số tin cậy.
"""

import math
from typing import Dict, Any, Tuple, List

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    nn = object
    F = object


class ConvBNAct(nn.Module if HAS_TORCH else object):
    """Khối tích chập kết hợp Batch Normalization và hàm kích hoạt SiLU/ReLU."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DepthwiseSeparableConv(nn.Module if HAS_TORCH else object):
    """Khối tích chập tách biệt theo chiều sâu (Depthwise Separable Convolution) tối ưu tốc độ."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.act(self.bn1(self.depthwise(x)))
        x = self.act(self.bn2(self.pointwise(x)))
        return x


class LightweightBackbone(nn.Module if HAS_TORCH else object):
    """Backbone trích xuất đặc trưng đa mức (C2: 1/4, C3: 1/8, C4: 1/16, C5: 1/32)."""
    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()
        # Stem: 1/2
        self.stem = ConvBNAct(in_channels, base_channels, kernel_size=3, stride=2, padding=1)
        # Stage 1: 1/4 (C2)
        self.stage1 = nn.Sequential(
            DepthwiseSeparableConv(base_channels, base_channels * 2, stride=2),
            ConvBNAct(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
        )
        # Stage 2: 1/8 (C3)
        self.stage2 = nn.Sequential(
            DepthwiseSeparableConv(base_channels * 2, base_channels * 4, stride=2),
            ConvBNAct(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
        )
        # Stage 3: 1/16 (C4)
        self.stage3 = nn.Sequential(
            DepthwiseSeparableConv(base_channels * 4, base_channels * 8, stride=2),
            ConvBNAct(base_channels * 8, base_channels * 8, kernel_size=3, padding=1),
        )
        # Stage 4: 1/32 (C5)
        self.stage4 = nn.Sequential(
            DepthwiseSeparableConv(base_channels * 8, base_channels * 8, stride=2),
            ConvBNAct(base_channels * 8, base_channels * 8, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x0 = self.stem(x)      # 1/2
        c2 = self.stage1(x0)   # 1/4
        c3 = self.stage2(c2)   # 1/8
        c4 = self.stage3(c3)   # 1/16
        c5 = self.stage4(c4)   # 1/32
        return c3, c4, c5


class FPNNeck(nn.Module if HAS_TORCH else object):
    """Feature Pyramid Network (FPN) hợp nhất đặc trưng nông (vị trí 4 góc) và đặc trưng sâu."""
    def __init__(self, in_channels_list: List[int] = [128, 256, 256], out_channels: int = 128):
        super().__init__()
        self.lateral3 = nn.Conv2d(in_channels_list[0], out_channels, kernel_size=1)
        self.lateral4 = nn.Conv2d(in_channels_list[1], out_channels, kernel_size=1)
        self.lateral5 = nn.Conv2d(in_channels_list[2], out_channels, kernel_size=1)

        self.smooth = ConvBNAct(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, c3, c4, c5):
        p5 = self.lateral5(c5)
        p4 = self.lateral4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.lateral3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        out = self.smooth(p3) # Trích xuất đặc trưng tại tỷ lệ 1/8
        return out


class PlateCornerDetector(nn.Module if HAS_TORCH else object):
    """
    Mạng định vị biển số và trích xuất 4 điểm góc tứ giác lồi (Anchor-Free Keypoint Detector).
    Đầu vào:
    - Ảnh [B, 3, H, W] (ví dụ 384x384 hoặc 640x640)
    Đầu ra (Outputs):
    1. 'confidence': [B, 1, H/8, W/8] - Xác suất tâm biển số (0..1)
    2. 'bbox': [B, 4, H/8, W/8] - Kích thước hộp bao [dx, dy, w, h]
    3. 'corners': [B, 8, H/8, W/8] - Độ dời 4 điểm góc so với tâm [(dx1, dy1), ..., (dx4, dy4)]
    4. 'layout': [B, 2, H/8, W/8] - Xác suất phân loại biển số 1 dòng vs 2 dòng
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()
        self.backbone = LightweightBackbone(in_channels=in_channels, base_channels=base_channels)
        self.neck = FPNNeck(
            in_channels_list=[base_channels * 4, base_channels * 8, base_channels * 8],
            out_channels=base_channels * 4,
        )

        feature_dim = base_channels * 4

        # Decoupled Prediction Heads:
        # 1. Objectness / Confidence Head (1 kênh)
        self.conf_head = nn.Sequential(
            ConvBNAct(feature_dim, feature_dim // 2, kernel_size=3, padding=1),
            nn.Conv2d(feature_dim // 2, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # 2. Bounding Box Head (4 kênh: offset_x, offset_y, width, height)
        self.bbox_head = nn.Sequential(
            ConvBNAct(feature_dim, feature_dim // 2, kernel_size=3, padding=1),
            nn.Conv2d(feature_dim // 2, 4, kernel_size=1),
        )

        # 3. 4-Corner Keypoint Regression Head (8 kênh: (dx1,dy1), (dx2,dy2), (dx3,dy3), (dx4,dy4))
        self.corner_head = nn.Sequential(
            ConvBNAct(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.Conv2d(feature_dim, 8, kernel_size=1),
        )

        # 4. Layout Classification Head (2 kênh: 1_LINE, 2_LINE)
        self.layout_head = nn.Sequential(
            ConvBNAct(feature_dim, feature_dim // 2, kernel_size=3, padding=1),
            nn.Conv2d(feature_dim // 2, 2, kernel_size=1),
        )

    def forward(self, x) -> Dict[str, Any]:
        c3, c4, c5 = self.backbone(x)
        feat = self.neck(c3, c4, c5)

        conf = self.conf_head(feat)
        bbox = self.bbox_head(feat)
        corners = self.corner_head(feat)
        layout = self.layout_head(feat)

        return {
            "confidence": conf,
            "bbox": bbox,
            "corners": corners,
            "layout": layout,
        }
