from dataclasses import dataclass, field
from typing import Optional
import numpy as np

CAPTURE_VERSION = 'image-mirrored-v1'
FEATURE_VERSION = 'hands-normalization-v3'

@dataclass
class RawHandSample:
    """Mirrored RGB image landmarks; fixed anatomical Left/Right slots, never screen order.
    Each present slot is (21,3) normalized image coordinates, not world coordinates.
    x/z use width and y uses height when pixelized by the student's normalizer.
    """
    left: Optional[np.ndarray]
    right: Optional[np.ndarray]
    width: int
    height: int
    timestamp: float
    batch: str = ''
    capture_version: str = CAPTURE_VERSION
    image_mirrored: bool = True
    camera: dict = field(default_factory=dict)

@dataclass
class FeatureResult:
    valid: bool
    features: Optional[np.ndarray] = None
    reason: str = ''
    featureVersion: str = FEATURE_VERSION

@dataclass
class Prediction:
    label: str = 'NONE'
    valid: bool = False
    nearest_distance: Optional[float] = None
    vote_ratio: float = 0.0
