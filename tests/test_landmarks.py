from types import SimpleNamespace as NS

import numpy as np

from core.landmarks import Landmarks


def test_raw_detections_keep_low_scores_conflicts_world_coordinates_and_order():
    points = [NS(x=0.1, y=0.2, z=-0.3) for _ in range(21)]
    world = [NS(x=0.01, y=0.02, z=-0.03) for _ in range(21)]
    categories = [[NS(index=0, category_name="Left", display_name="", score=score)]
                  for score in (0.2, 0.9)]
    result = NS(hand_landmarks=[points, points], hand_world_landmarks=[world, world],
                handedness=categories)
    calls = []
    detector = Landmarks.__new__(Landmarks)
    detector.last_ms = -1
    def detect(image, timestamp):
        calls.append(timestamp)
        return result
    detector.detector = NS(detect_for_video=detect)
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    sample = detector.detect(frame, 1.0, {"index": 0})
    assert len(sample.hands) == 2
    assert [h.handedness[0]["score"] for h in sample.hands] == [0.2, 0.9]
    assert sample.hands[0].landmarks[0] == [0.1, 0.2, -0.3]
    assert sample.hands[0].world_landmarks[0] == [0.01, 0.02, -0.03]
    assert sample.width == 64 and sample.height == 48
    result.hand_landmarks = []
    result.hand_world_landmarks = []
    result.handedness = []
    assert detector.detect(frame, 1.0, {}).hands == []
    assert calls == [1000, 1001]


def test_preview_keeps_capture_pixels_and_stops_on_key(monkeypatch):
    import cv2
    from core.preview import show_preview
    from core.types import RawHand, RawHandSample
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    hand = RawHand([[0.5, 0.5, 0.0]] * 21, [],
                   [{"category_name": "Left", "score": 0.2}])
    sample = RawHandSample([hand], 64, 48, 1.0)
    displayed = []
    monkeypatch.setattr(cv2, "imshow", lambda name, image: displayed.append(image))
    monkeypatch.setattr(cv2, "waitKey", lambda delay: ord("q"))
    assert not show_preview(frame, sample)
    assert not frame.any()
    assert displayed[0].any()
