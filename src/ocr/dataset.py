"""
FAST-PLATE-OCR-VIETNAMESE: OCR Dataset Loader & Virtual Strip Pipeline
Dataset PyTorch phục vụ huấn luyện 2 pha (Synthetic Pre-training & Real Fine-tuning).
Tự động chuyển đổi biển 2 dòng về Virtual 1-Line Strip (64x256x3) và mã hóa nhãn nhúng 10 slots.
"""

import os
import yaml
import cv2
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional

import torch
from torch.utils.data import Dataset

from src.rectification.splitter import TwoLinePlateSplitter
from src.ocr.augment import PlateAugmentor
from src.synthetic.generator import generate_plate_specification
from src.synthetic.renderer import LicensePlateRenderer
from src.synthetic.textures import apply_random_weathering


class RealPlateOCRDataset(Dataset):
    """
    Dataset cho tập dữ liệu thực tế (Stage 2: Fine-tuning):
    - Đọc metadata từ tệp CSV (train.csv / val.csv).
    - Tự động chuẩn hóa biển vuông thành dải Virtual 1-Line Strip (64x256).
    - Áp dụng các tầng tăng cường Screw Cutout, Motion Blur, Noise nếu is_train=True.
    - Mã hóa chuỗi ký tự thành vector tensor có độ dài max_plate_slots (10).
    """

    def __init__(
        self,
        csv_path: str,
        base_dir: str = "data/kaggle_dataset",
        config_path: str = "configs/plate_config_vietnam.yaml",
        is_train: bool = True,
        augmentor: Optional[PlateAugmentor] = None,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.csv_path = csv_path
        self.base_dir = base_dir
        self.is_train = is_train

        # Đọc cấu hình quy chuẩn
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.alphabet: str = cfg.get("alphabet", "0123456789ABCDEFGHKLMNPQRSTUVXYZ_")
        self.pad_char: str = cfg.get("pad_char", "_")
        self.max_slots: int = cfg.get("max_plate_slots", 10)
        self.img_h: int = cfg.get("img_height", 64)
        self.img_w: int = cfg.get("img_width", 256)

        self.char2idx = {c: i for i, c in enumerate(self.alphabet)}
        self.idx2char = {i: c for i, c in enumerate(self.alphabet)}
        self.pad_idx = self.char2idx[self.pad_char]

        # Đọc danh sách mẫu
        df = pd.read_csv(csv_path)
        if max_samples is not None and max_samples < len(df):
            df = df.iloc[:max_samples]
        self.records = df.to_dict("records")

        # Khởi tạo bộ cắt ghép Virtual Strip và Augmentor
        self.splitter = TwoLinePlateSplitter(strip_height=self.img_h, target_ocr_size=(self.img_w, self.img_h))
        self.augmentor = augmentor if augmentor is not None else (PlateAugmentor() if is_train else None)

    def __len__(self) -> int:
        return len(self.records)

    def encode_label(self, text: str) -> torch.Tensor:
        """
        Mã hóa chuỗi nhãn sang chỉ số alphabet và thêm padding '_' đến đúng max_slots (10).
        Ví dụ: "30A12345" -> "30A12345__" -> tensor([10 slots])
        """
        text_clean = "".join([c for c in text.upper() if c in self.char2idx])
        padded = text_clean.ljust(self.max_slots, self.pad_char)[: self.max_slots]
        indices = [self.char2idx[c] for c in padded]
        return torch.tensor(indices, dtype=torch.long)

    def decode_indices(self, indices: List[int]) -> str:
        """
        Chuyển đổi vector chỉ số thành chuỗi ký tự, tự động loại bỏ padding '_'.
        """
        chars = [self.idx2char.get(idx, "") for idx in indices]
        return "".join(chars).replace(self.pad_char, "")

    def preprocess_image(self, img_bgr: np.ndarray, layout: str = "LONG") -> np.ndarray:
        """
        Nắn chỉnh hình học và đồng bộ hóa thành dải ảnh phẳng 64x256.
        """
        h, w = img_bgr.shape[:2]
        ar = w / float(max(1, h))

        # Nếu là biển vuông (aspect ratio < 2.2 hoặc nhãn ghi SQUARE), kích hoạt Virtual 1-Line Strip
        if ar < 2.2 or layout == "SQUARE":
            processed_img, _ = self.splitter.create_virtual_1line_strip(img_bgr)
        else:
            processed_img = cv2.resize(img_bgr, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)

        return processed_img

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        rec = self.records[idx]
        img_rel = rec["filepath"]
        abs_path = os.path.join(self.base_dir, img_rel)

        img_bgr = cv2.imread(abs_path)
        if img_bgr is None:
            # Fallback nếu ảnh hỏng
            img_bgr = np.full((self.img_h, self.img_w, 3), 128, dtype=np.uint8)

        layout = str(rec.get("layout", "LONG"))
        plate_strip = self.preprocess_image(img_bgr, layout=layout)

        # Áp dụng Augmentations (ốc vít, nhòe, nhiễu) trong pha huấn luyện
        if self.is_train and self.augmentor is not None:
            plate_strip = self.augmentor(plate_strip, is_train=True)

        # Chuyển đổi BGR -> RGB và chuẩn hóa tensor [0, 1] rồi z-score
        img_rgb = cv2.cvtColor(plate_strip, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # Chuẩn hóa chuẩn ImageNet: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm_img = (img_rgb - mean) / std

        # Chuyển layout (H, W, C) -> (C, H, W)
        img_tensor = torch.from_numpy(norm_img).permute(2, 0, 1).float()

        raw_label = str(rec["label"])
        label_tensor = self.encode_label(raw_label)

        return img_tensor, label_tensor, raw_label


class SyntheticPlateOCRDataset(Dataset):
    """
    Dataset cho tập dữ liệu giả lập (Stage 1: Pre-training):
    - Tự động sinh ngẫu nhiên hàng chục nghìn biển số hợp chuẩn văn bản pháp lý.
    - Kết xuất đồ họa cơ sở + phủ texture vật lý (xước kim loại, bụi bùn).
    - Tạo dải Virtual 1-Line Strip với tỷ lệ đa dạng.
    """

    def __init__(
        self,
        num_samples: int = 10000,
        config_path: str = "configs/plate_config_vietnam.yaml",
        augmentor: Optional[PlateAugmentor] = None,
    ):
        super().__init__()
        self.num_samples = num_samples

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.alphabet: str = cfg.get("alphabet", "0123456789ABCDEFGHKLMNPQRSTUVXYZ_")
        self.pad_char: str = cfg.get("pad_char", "_")
        self.max_slots: int = cfg.get("max_plate_slots", 10)
        self.img_h: int = cfg.get("img_height", 64)
        self.img_w: int = cfg.get("img_width", 256)

        self.char2idx = {c: i for i, c in enumerate(self.alphabet)}
        self.idx2char = {i: c for i, c in enumerate(self.alphabet)}
        self.pad_idx = self.char2idx[self.pad_char]

        self.renderer = LicensePlateRenderer()
        self.splitter = TwoLinePlateSplitter(strip_height=self.img_h, target_ocr_size=(self.img_w, self.img_h))
        self.augmentor = augmentor if augmentor is not None else PlateAugmentor()

    def __len__(self) -> int:
        return self.num_samples

    def encode_label(self, text: str) -> torch.Tensor:
        text_clean = "".join([c for c in text.upper() if c in self.char2idx])
        padded = text_clean.ljust(self.max_slots, self.pad_char)[: self.max_slots]
        indices = [self.char2idx[c] for c in padded]
        return torch.tensor(indices, dtype=torch.long)

    def decode_indices(self, indices: List[int]) -> str:
        chars = [self.idx2char.get(idx, "") for idx in indices]
        return "".join(chars).replace(self.pad_char, "")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        # 1. Sinh cấu hình biển số chuẩn ngữ pháp
        spec = generate_plate_specification()
        label = spec["normalized"]

        # 2. Kết xuất hình ảnh đồ họa
        plate_bgr = self.renderer.render(spec)

        # 3. Phủ texture vật lý thời tiết
        weathered = apply_random_weathering(plate_bgr)

        # 4. Chuẩn hóa dải Virtual 1-Line Strip
        if spec["is_two_line"]:
            plate_strip, _ = self.splitter.create_virtual_1line_strip(weathered)
        else:
            plate_strip = cv2.resize(weathered, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)

        # 5. Áp dụng augmentor
        plate_strip = self.augmentor(plate_strip, is_train=True)

        # 6. Tensor hóa
        img_rgb = cv2.cvtColor(plate_strip, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm_img = (img_rgb - mean) / std

        img_tensor = torch.from_numpy(norm_img).permute(2, 0, 1).float()
        label_tensor = self.encode_label(label)

        return img_tensor, label_tensor, label
