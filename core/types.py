from dataclasses import dataclass, field

CAPTURE_VERSION = "mediapipe-raw-v1"


@dataclass
class RawHand:
    """One detected hand, in MediaPipe output order; no side assignment or rejection.

    landmarks/world_landmarks each contain 21 [x, y, z] points.
    handedness retains all returned category fields and scores.
    """
    landmarks: list
    world_landmarks: list
    handedness: list


@dataclass
class RawHandSample:
    """All detected hands, before student interpretation. No hands means [].

    Image landmarks use normalized image coordinates; world landmarks use meters
    with an origin at each hand's geometric center, not a shared scene origin.
    """
    hands: list[RawHand]
    width: int
    height: int
    timestamp: float
    capture_version: str = CAPTURE_VERSION
    image_mirrored: bool = True
    camera: dict = field(default_factory=dict)
