"""
FAST-PLATE-OCR-VIETNAMESE: Synthetic Data Engine Package
"""

from .generator import generate_plate_specification
from .renderer import LicensePlateRenderer
from .textures import PhysicalTextureGenerator, apply_random_weathering
from .augmentor import RealisticAugmentor, augment_plate_image
from .engine import SyntheticDataEngine, generate_single_sample

__all__ = [
    "generate_plate_specification",
    "LicensePlateRenderer",
    "PhysicalTextureGenerator",
    "apply_random_weathering",
    "RealisticAugmentor",
    "augment_plate_image",
    "SyntheticDataEngine",
    "generate_single_sample",
]
