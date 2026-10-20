"""
FAST-PLATE-OCR-VIETNAMESE: OCR Module
Kiến trúc CCT Adapter và quy trình huấn luyện hai pha Transfer Learning.
"""

from src.ocr.cct_adapter import CCTPlateVietnamModel
from src.ocr.augment import PlateAugmentor, apply_screw_cutout, apply_motion_blur, apply_photometric_distortions
from src.ocr.metrics import compute_cer, compute_accuracy, evaluate_predictions, levenshtein_distance
from src.ocr.dataset import RealPlateOCRDataset, SyntheticPlateOCRDataset
from src.ocr.train import run_two_stage_training, train_one_epoch, evaluate_ocr

__all__ = [
    "CCTPlateVietnamModel",
    "PlateAugmentor",
    "apply_screw_cutout",
    "apply_motion_blur",
    "apply_photometric_distortions",
    "compute_cer",
    "compute_accuracy",
    "evaluate_predictions",
    "levenshtein_distance",
    "RealPlateOCRDataset",
    "SyntheticPlateOCRDataset",
    "run_two_stage_training",
    "train_one_epoch",
    "evaluate_ocr",
]
