from core.types import Prediction


class StudentFilter:
    def __init__(self):
        self.hold_seconds = 0.15
        self.reset()

    def reset(self):
        self.candidate = None
        self.start_time = None
        return None

    def update(self, prediction: Prediction, timestamp: float) -> str:
        if not prediction.valid or prediction.label == 'NONE':
            self.reset()
            return 'NONE'

        if prediction.label != self.candidate:
            self.candidate = prediction.label
            self.start_time = timestamp
            return 'NONE'

        elapsed = timestamp - self.start_time
        if elapsed >= self.hold_seconds:
            return self.candidate
        return 'NONE'
