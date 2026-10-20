"""
FAST-PLATE-OCR-VIETNAMESE: Unit Test Suite for Step 3.2
Kiểm thử bộ làm giàu dữ liệu (Screw Cutout, Motion Blur),
Bộ đánh giá khoảng cách Levenshtein & CER,
Dataset Loaders và quy trình huấn luyện hai pha Transfer Learning.
"""

import os
import sys
import unittest
import numpy as np
import torch

# Đảm bảo import được module từ thư mục gốc
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ocr.augment import (
    apply_screw_cutout,
    apply_motion_blur,
    apply_photometric_distortions,
    PlateAugmentor,
)
from src.ocr.metrics import (
    levenshtein_distance,
    compute_cer,
    compute_accuracy,
    evaluate_predictions,
)
from src.ocr.dataset import RealPlateOCRDataset, SyntheticPlateOCRDataset
from src.ocr.cct_adapter import CCTPlateVietnamModel
from src.ocr.train import train_one_epoch, evaluate_ocr


class TestOCRTrainingStep32(unittest.TestCase):

    def setUp(self):
        self.config_path = "configs/plate_config_vietnam.yaml"
        self.train_csv = "data/kaggle_dataset/train.csv"
        self.val_csv = "data/kaggle_dataset/val.csv"

    def test_augmentations(self):
        """Kiểm tra các kỹ thuật làm giàu ảnh nâng cao (Screw cutout, Motion blur, Photometric)."""
        dummy_img = np.full((64, 256, 3), 200, dtype=np.uint8)

        # 1. Kiểm tra ốc vít khoan biển số
        screw_img = apply_screw_cutout(dummy_img, num_screws_range=(2, 4), p=1.0)
        self.assertEqual(screw_img.shape, dummy_img.shape)
        # Điểm ảnh phải có sự thay đổi do vẽ ốc vít
        self.assertTrue(np.any(screw_img != dummy_img))

        # 2. Kiểm tra nhòe chuyển động
        blur_img = apply_motion_blur(dummy_img, kernel_size_range=(5, 5), p=1.0)
        self.assertEqual(blur_img.shape, dummy_img.shape)

        # 3. Kiểm tra biến dạng quang học
        photo_img = apply_photometric_distortions(dummy_img, p=1.0)
        self.assertEqual(photo_img.shape, dummy_img.shape)

        # 4. Kiểm tra Class PlateAugmentor
        augmentor = PlateAugmentor(screw_prob=1.0, blur_prob=1.0, photo_prob=1.0)
        augmented = augmentor(dummy_img, is_train=True)
        self.assertEqual(augmented.shape, (64, 256, 3))
        print("[PASS] test_augmentations (Screw Cutout, Motion Blur, Photometric)")

    def test_metrics_calculation(self):
        """Kiểm thử tính toán Levenshtein Distance, CER và Exact Match Accuracy."""
        # 1. Trùng khớp 100%
        d1 = levenshtein_distance("30A12345", "30A12345")
        self.assertEqual(d1, 0)

        # 2. Sai 1 ký tự
        d2 = levenshtein_distance("30A12345", "30A12346")
        self.assertEqual(d2, 1)

        # 3. Thay thế, chèn, xóa
        d3 = levenshtein_distance("29B1", "29B12")
        self.assertEqual(d3, 1)

        # 4. Kiểm tra CER
        preds = ["30A12345", "29B11111"]
        gts = ["30A12345", "29B11112"]  # 1 ký tự sai trên tổng 16 ký tự
        cer = compute_cer(preds, gts)
        self.assertAlmostEqual(cer, 1.0 / 16.0, places=4)

        acc = compute_accuracy(preds, gts)
        self.assertEqual(acc, 0.5)

        eval_res = evaluate_predictions(preds, gts)
        self.assertEqual(eval_res["total_samples"], 2)
        self.assertEqual(eval_res["exact_matches"], 1)
        self.assertAlmostEqual(eval_res["cer_percent"], 6.25, places=2)
        print("[PASS] test_metrics_calculation (Levenshtein, CER, Exact Match)")

    def test_datasets(self):
        """Kiểm tra Dataset Loader cho cả dữ liệu giả lập và thực tế."""
        # 1. Synthetic Dataset
        syn_ds = SyntheticPlateOCRDataset(num_samples=5, config_path=self.config_path)
        self.assertEqual(len(syn_ds), 5)
        img_s, lbl_s, raw_s = syn_ds[0]
        self.assertEqual(img_s.shape, torch.Size([3, 64, 256]))
        self.assertEqual(lbl_s.shape, torch.Size([10]))
        self.assertIsInstance(raw_s, str)

        # 2. Real Dataset
        if os.path.exists(self.train_csv):
            real_ds = RealPlateOCRDataset(self.train_csv, max_samples=10, is_train=True)
            self.assertEqual(len(real_ds), 10)
            img_r, lbl_r, raw_r = real_ds[0]
            self.assertEqual(img_r.shape, torch.Size([3, 64, 256]))
            self.assertEqual(lbl_r.shape, torch.Size([10]))
            decoded = real_ds.decode_indices(lbl_r.tolist())
            self.assertEqual(decoded, raw_r)

        print("[PASS] test_datasets (Synthetic & Real Kaggle loaders)")

    def test_transfer_learning_step(self):
        """Kiểm tra 1 bước tối ưu gradient descent của CCT Model trên batch dữ liệu."""
        model = CCTPlateVietnamModel(config_path=self.config_path)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        device = torch.device("cpu")

        # Tạo dummy loader
        x = torch.randn(4, 3, 64, 256)
        y = torch.randint(0, 33, (4, 10))
        dummy_dataset = torch.utils.data.TensorDataset(x, y, torch.tensor([0, 1, 2, 3]))
        loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=4)

        initial_loss = None
        for images, targets, _ in loader:
            logits = model(images)
            B, slots, c = logits.shape
            initial_loss = torch.nn.functional.cross_entropy(logits.view(-1, c), targets.view(-1)).item()
            break

        # Thực hiện 1 epoch tối ưu
        loss_val = train_one_epoch(model, loader, optimizer, device)
        self.assertIsInstance(loss_val, float)
        self.assertGreater(loss_val, 0.0)
        print(f"[PASS] test_transfer_learning_step (Initial loss: {initial_loss:.4f} -> Train step: {loss_val:.4f})")


if __name__ == "__main__":
    print("=" * 60)
    print("CHẠY BỘ KIỂM THỬ HUẤN LUYỆN HAI PHA & TĂNG CƯỜNG DỮ LIỆU (BƯỚC 3.2)")
    print("=" * 60)
    unittest.main()
