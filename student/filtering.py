import math
from student.types import Prediction


class StudentFilter:
    def __init__(self):
        self.hold_seconds = 0.05
        self.fast_hold_seconds = 0.02
        self.strong_vote = 0.8
        self.reset()

    def reset(self):
        self.candidate = None
        self.start_time = None
        self.frames = 0
        self.strong_frames = 0
        self.stable_label = 'NONE'
        self.transition_start = None
        self.last_timestamp = None
        return None

    def update(self, prediction: Prediction, timestamp: float) -> str:
        if not math.isfinite(timestamp) or not prediction.valid or prediction.label == 'NONE':
            self.reset()
            return 'NONE'

        # A pause or a clock reset must not reuse evidence from an old frame.
        if self.last_timestamp is not None and (timestamp <= self.last_timestamp or timestamp - self.last_timestamp > 0.5):
            self.reset()
        self.last_timestamp = timestamp

        if prediction.label == self.stable_label:
            self.candidate = None
            self.transition_start = None
            return self.stable_label

        if prediction.label != self.candidate:
            self.candidate = prediction.label
            self.start_time = timestamp
            self.frames = 1
            self.strong_frames = int(prediction.vote_ratio >= self.strong_vote)
        else:
            self.frames += 1
            self.strong_frames = self.strong_frames + 1 if prediction.vote_ratio >= self.strong_vote else 0

        elapsed = timestamp - self.start_time
        hold = self.fast_hold_seconds if self.strong_frames >= 2 else self.hold_seconds
        if self.frames >= 2 and elapsed >= hold:
            self.stable_label = self.candidate
            self.transition_start = None
            return self.stable_label

        # Bridge a brief valid-label glitch, but never hold an old action
        # indefinitely when candidates keep changing.
        if self.stable_label != 'NONE':
            if self.transition_start is None:
                self.transition_start = timestamp
            if timestamp - self.transition_start < self.hold_seconds:
                return self.stable_label
        return 'NONE'
