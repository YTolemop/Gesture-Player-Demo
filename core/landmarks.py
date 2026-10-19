from pathlib import Path
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from core.types import RawHand, RawHandSample

MODEL = Path(__file__).resolve().parents[1] / 'assets/models/hand_landmarker.task'

class Landmarks:
    def __init__(self):
        if not MODEL.is_file():
            raise RuntimeError('缺少關鍵點模型，請執行 python main.py prepare')
        from core.resources import verify_model
        verify_model()
        # Native Windows file APIs in MediaPipe may fail on non-ASCII paths.
        # Loading verified bytes also keeps the model strictly local/offline.
        self.detector = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_buffer=MODEL.read_bytes()), running_mode=vision.RunningMode.VIDEO, num_hands=2))
        self.last_ms = -1

    def detect(self, frame, timestamp, camera):
        self.last_ms = max(self.last_ms + 1, int(timestamp * 1000))
        result = self.detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), self.last_ms)
        hands = []
        for index, points in enumerate(result.hand_landmarks):
            hands.append(RawHand(
                landmarks=[[p.x, p.y, p.z] for p in points],
                world_landmarks=[[p.x, p.y, p.z] for p in result.hand_world_landmarks[index]],
                handedness=[{
                    "index": category.index,
                    "category_name": category.category_name,
                    "display_name": category.display_name,
                    "score": category.score,
                } for category in result.handedness[index]],
            ))
        return RawHandSample(hands, frame.shape[1], frame.shape[0], timestamp,
                             camera=camera)

    def close(self):
        self.detector.close()
