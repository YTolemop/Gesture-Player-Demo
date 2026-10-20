import json
import sys
import types
import numpy as np
from main import capture, parser
from core.types import RawHand, RawHandSample


def test_recording_without_windows_saves_samples_and_releases_on_interrupt(monkeypatch, tmp_path):
    closed = []
    class Camera:
        actual = {"index": 0}
        def __init__(self, index): self.count = 0
        def latest(self):
            self.count += 1
            if self.count > 2: raise KeyboardInterrupt
            return np.zeros((48, 64, 3), dtype=np.uint8), float(self.count)
        def close(self): closed.append("camera")
    class Detector:
        def __init__(self, **kwargs): pass
        def detect(self, frame, timestamp, camera):
            hand = RawHand([[0.1, 0.2, 0.3]] * 21, [[0.01, 0.02, 0.03]] * 21,
                           [{"index": 0, "category_name": "Left", "display_name": "", "score": 0.2}])
            return RawHandSample([hand] if timestamp == 1 else [], 64, 48,
                                 timestamp, camera=camera)
        def close(self): closed.append("detector")
    class Writer:
        def __init__(self, *args): self.count = 0
        def isOpened(self): return True
        def write(self, frame): self.count += 1
        def release(self): closed.append(f"video:{self.count}")
    import cv2
    monkeypatch.setitem(sys.modules, "core.camera", types.SimpleNamespace(Camera=Camera))
    monkeypatch.setitem(sys.modules, "core.landmarks", types.SimpleNamespace(Landmarks=Detector))
    monkeypatch.setattr(cv2, "VideoWriter", Writer)
    args = parser().parse_args(["record", "--label", "wave", "--channel", "left", "--purpose", "test", "--output", str(tmp_path)])
    capture(args)
    rows = [json.loads(line) for line in next(tmp_path.glob("*.jsonl")).read_text().splitlines()]
    assert len(rows) == 2 and rows[0]["channel"] == "left"
    assert "left" not in rows[0] and "right" not in rows[0]
    hand = rows[0]["hands"][0]
    assert len(hand["landmarks"]) == len(hand["world_landmarks"]) == 21
    assert hand["handedness"][0]["score"] == 0.2
    assert rows[1]["hands"] == []
    assert rows[0]["capture_version"] == "mediapipe-raw-v1"
    assert rows[0]["purpose"] == "test"
    assert rows[1]["timestamp"] > rows[0]["timestamp"]
    assert closed == ["video:2", "camera", "detector"]


def test_run_passes_empty_frames_to_student_and_sends_student_commands(monkeypatch):
    from network.commands import Commands
    received, sent, resets, closed, previews = [], [], [], [], []
    class Camera:
        actual = {"index": 0}
        def __init__(self, index): self.count = 0
        def latest(self):
            self.count += 1
            if self.count > 2:
                raise KeyboardInterrupt
            return np.zeros((48, 64, 3), dtype=np.uint8), float(self.count)
        def close(self): closed.append("camera")
    class Detector:
        def detect(self, frame, timestamp, camera):
            hands = [RawHand([[0, 0, 0]] * 21, [[0, 0, 0]] * 21, [])]
            return RawHandSample(hands if timestamp == 1 else [], 64, 48, timestamp)
        def close(self): closed.append("detector")
    class Pipeline:
        algorithms = {"normalization": True, "knn": True, "filterEnabled": False}
        def __init__(self, path): pass
        def reset(self): resets.append(True)
        def predict(self, sample):
            received.append(sample.hands)
            return Commands(right="BLOCK") if sample.hands else Commands()
    class Client:
        revision = 0
        def connect(self, *args): return {"roomCode": "room"}
        def send(self, commands, revision): sent.append(commands)
        def send_preview(self, frame, sample, revision):
            previews.append((frame.copy(), sample, revision))
        def close(self): closed.append("client")
    # Even frames less than 100ms apart must reach the transport's edge detection.
    import main
    counter = iter(range(100))
    monkeypatch.setattr(main.time, "monotonic", lambda: float(next(counter)) * .03)
    monkeypatch.setitem(sys.modules, "core.camera", types.SimpleNamespace(Camera=Camera))
    monkeypatch.setitem(sys.modules, "core.landmarks", types.SimpleNamespace(Landmarks=Detector))
    monkeypatch.setitem(sys.modules, "student.pipeline", types.SimpleNamespace(Pipeline=Pipeline))
    monkeypatch.setitem(sys.modules, "network.arena_client", types.SimpleNamespace(ArenaClient=Client))
    args = parser().parse_args(["run", "--server", "https://ga-game.codpy.org",
                               "--pair-code", "012345"])
    capture(args)
    assert len(received) == 2 and received[1] == []
    assert sent == [Commands(right="BLOCK"), Commands()]
    assert len(previews) == 2  # Web preview also runs without --preview.
    assert previews[0][1].hands == received[0]
    assert previews[1][1].hands == []
    assert len(resets) == 1  # Context reset only; missing-hand decisions belong to students.
    assert closed == ["client", "camera", "detector"]
