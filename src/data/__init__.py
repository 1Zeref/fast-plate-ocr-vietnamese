"""
FAST-PLATE-OCR-VIETNAMESE: Data Ingestion, Video Sampling & Annotation Package
"""

from .prepare_kaggle import prepare_kaggle_dataset, sanitize_label
from .video_sampler import VideoFrameSampler
from .annotator import (
    LicensePlateAnnotation,
    RealDatasetManager,
    rectify_plate_quad,
)
from .real_pipeline import (
    create_realistic_scene_frame,
    generate_real_traffic_dataset,
)

__all__ = [
    "prepare_kaggle_dataset",
    "sanitize_label",
    "VideoFrameSampler",
    "LicensePlateAnnotation",
    "RealDatasetManager",
    "rectify_plate_quad",
    "create_realistic_scene_frame",
    "generate_real_traffic_dataset",
]
