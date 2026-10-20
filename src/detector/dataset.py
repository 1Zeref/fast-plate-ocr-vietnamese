"""
FAST-PLATE-OCR-VIETNAMESE: Vehicle & 4-Corner Plate Dataset Loader
Nạp dữ liệu từ Bước 1.3, chuẩn hóa tọa độ hộp bao và 4 điểm góc cho mô hình Detector.
"""

import os
import json
import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

try:
    import torch
    from torch.utils.data import Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    Dataset = object


def letterbox(
    image: np.ndarray,
    target_size: Tuple[int, int] = (384, 384),
    fill_value: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float, int, int]:
    """
    Biến đổi ảnh theo thuật toán Letterbox cố định tỷ lệ khung hình:
    Tránh biến dạng hình học của biển số và padding viền xám tiêu chuẩn (114, 114, 114).
    """
    ih, iw = image.shape[:2]
    tw, th = target_size
    scale = min(tw / max(1, iw), th / max(1, ih))
    nw = int(round(iw * scale))
    nh = int(round(ih * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_x = (tw - nw) // 2
    pad_y = (th - nh) // 2
    canvas = np.full((th, tw, 3), fill_value, dtype=np.uint8)
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return canvas, scale, pad_x, pad_y


class VehiclePlateCornerDataset(Dataset):
    """
    Dataset phục vụ huấn luyện mô hình Anchor-Free Keypoint Plate Detector.
    Đầu vào:
    - json_path: Đường dẫn tệp nhãn annotations_full.json từ Bước 1.3.
    - target_size: Kích thước ảnh đầu vào mô hình (Width, Height), mặc định (384, 384).
    Đầu ra cho mỗi mẫu:
    - image: Tensor ảnh chuẩn hóa [3, H, W] trong dải [0.0, 1.0].
    - target: Dict chứa:
        - 'boxes': [N, 4] bounding box biển số [x_min, y_min, x_max, y_max] chuẩn hóa [0, 1].
        - 'corners': [N, 4, 2] tọa độ 4 điểm góc [(x1,y1), (x2,y2), (x3,y3), (x4,y4)] chuẩn hóa [0, 1].
        - 'vehicle_boxes': [N, 4] bounding box xe [0, 1].
        - 'labels': [N] phân loại layout (0: 1_LINE, 1: 2_LINE).
    """

    def __init__(
        self,
        json_path: str = "data/real_traffic_dataset/annotations_full.json",
        dataset_root: Optional[str] = None,
        target_size: Tuple[int, int] = (384, 384),
        is_train: bool = True,
    ):
        self.json_path = os.path.abspath(json_path)
        self.dataset_root = dataset_root or os.path.dirname(self.json_path)
        self.target_w, self.target_h = target_size
        self.is_train = is_train

        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"Không tìm thấy file nhãn: {self.json_path}")

        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                self.frames = data.get("frames", [])
            elif isinstance(data, list):
                self.frames = data
            else:
                self.frames = []

        # Lọc các frame có ít nhất 1 biển số
        self.samples = [
            f for f in self.frames
            if f.get("num_objects", 0) > 0 or f.get("num_plates", 0) > 0 or len(f.get("objects", [])) > 0
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Any, Dict[str, Any]]:
        sample = self.samples[idx]
        img_rel_path = sample["image_path"]
        img_full_path = os.path.join(self.dataset_root, img_rel_path)

        # Đọc ảnh
        img = cv2.imread(img_full_path)
        if img is None:
            # Tạo frame rỗng nếu ảnh bị thiếu
            img = np.zeros((self.target_h, self.target_w, 3), dtype=np.uint8)

        orig_h, orig_w = img.shape[:2]

        # Tiền xử lý theo chuẩn Letterbox cố định tỷ lệ
        canvas, scale, pad_x, pad_y = letterbox(img, target_size=(self.target_w, self.target_h))
        rgb_img = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        chw_img = np.transpose(rgb_img, (2, 0, 1))

        plate_boxes = []
        corners_list = []
        vehicle_boxes = []
        labels = []

        for obj in sample["objects"]:
            # Bounding box biển số (hỗ trợ cả plate_bbox và bbox_xyxy)
            pb = obj.get("plate_bbox") or obj.get("bbox_xyxy") or [0, 0, orig_w, orig_h]
            bx1 = np.clip((pb[0] * scale + pad_x) / float(self.target_w), 0.0, 1.0)
            by1 = np.clip((pb[1] * scale + pad_y) / float(self.target_h), 0.0, 1.0)
            bx2 = np.clip((pb[2] * scale + pad_x) / float(self.target_w), 0.0, 1.0)
            by2 = np.clip((pb[3] * scale + pad_y) / float(self.target_h), 0.0, 1.0)
            plate_boxes.append([bx1, by1, bx2, by2])

            # 4 điểm góc (Corners)
            raw_corners = obj.get("corners")
            if not raw_corners:
                raw_corners = [
                    [pb[0], pb[1]],
                    [pb[2], pb[1]],
                    [pb[2], pb[3]],
                    [pb[0], pb[3]],
                ]
            norm_corners = []
            for pt in raw_corners:
                cx = np.clip((pt[0] * scale + pad_x) / float(self.target_w), 0.0, 1.0)
                cy = np.clip((pt[1] * scale + pad_y) / float(self.target_h), 0.0, 1.0)
                norm_corners.append([cx, cy])
            corners_list.append(norm_corners)

            # Bounding box phương tiện
            vb = obj.get("vehicle_bbox") or pb
            v_box_norm = [
                np.clip((vb[0] * scale + pad_x) / float(self.target_w), 0.0, 1.0),
                np.clip((vb[1] * scale + pad_y) / float(self.target_h), 0.0, 1.0),
                np.clip((vb[2] * scale + pad_x) / float(self.target_w), 0.0, 1.0),
                np.clip((vb[3] * scale + pad_y) / float(self.target_h), 0.0, 1.0),
            ]
            vehicle_boxes.append(v_box_norm)

            # Label layout (0: 1_LINE, 1: 2_LINE)
            labels.append(1 if obj.get("layout") == "2_LINE" else 0)

        target = {
            "boxes": np.array(plate_boxes, dtype=np.float32),
            "corners": np.array(corners_list, dtype=np.float32),
            "vehicle_boxes": np.array(vehicle_boxes, dtype=np.float32),
            "labels": np.array(labels, dtype=np.int64),
            "frame_id": sample.get("frame_id", ""),
            "scenario": sample.get("scenario", ""),
        }

        if HAS_TORCH:
            tensor_img = torch.from_numpy(chw_img)
            target_torch = {
                "boxes": torch.from_numpy(target["boxes"]),
                "corners": torch.from_numpy(target["corners"]),
                "vehicle_boxes": torch.from_numpy(target["vehicle_boxes"]),
                "labels": torch.from_numpy(target["labels"]),
                "frame_id": target["frame_id"],
                "scenario": target["scenario"],
            }
            return tensor_img, target_torch

        return chw_img, target
