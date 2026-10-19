from dataclasses import dataclass
from typing import Optional
import numpy as np

FEATURE_VERSION = "mixed-hands-131-v1"

@dataclass
class HandSlots:
    left: Optional[np.ndarray]
    right: Optional[np.ndarray]
    width: int
    height: int
    timestamp: float

@dataclass
class FeatureResult:
    valid: bool
    features: Optional[np.ndarray] = None
    reason: str = ""
    featureVersion: str = FEATURE_VERSION

@dataclass
class Prediction:
    label: str = "NONE"
    valid: bool = False
    nearest_distance: Optional[float] = None
    vote_ratio: float = 0.0
