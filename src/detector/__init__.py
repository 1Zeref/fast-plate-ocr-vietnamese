"""
FAST-PLATE-OCR-VIETNAMESE: Plate & Vehicle Corner Detector Package
"""

from .dataset import VehiclePlateCornerDataset
from .model import PlateCornerDetector
from .loss import WingLoss, PlateDetectorMultiTaskLoss
from .infer import PlateDetectorInference
from .train import train_detector

__all__ = [
    "VehiclePlateCornerDataset",
    "PlateCornerDetector",
    "WingLoss",
    "PlateDetectorMultiTaskLoss",
    "PlateDetectorInference",
    "train_detector",
]
