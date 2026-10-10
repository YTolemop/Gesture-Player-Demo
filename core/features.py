import time
import threading
import numpy as np
from core.types import FeatureResult, RawHandSample, Prediction, CAPTURE_VERSION
from core.labels import LABELS
from student.normalization import StudentNormalizer
from student.filtering import StudentFilter

def raw_valid(sample: RawHandSample, config: dict) -> tuple[bool, str]:
    if sample.capture_version != CAPTURE_VERSION or not sample.image_mirrored:
        return False, '擷取契約不相容'
    if sample.width <= 0 or sample.height <= 0:
        return False, '影像尺寸無效'
    if sample.left is not None:
        return False, '只能使用右手，請將左手移出畫面'
    if sample.right is None:
        return False, '未偵測到右手'
    slots = [sample.left, sample.right]
    if not any(v is not None for v in slots):
        return False, '無手'
    for hand in slots:
        if hand is None:
            continue
        if hand.shape != (21, 3) or not np.isfinite(hand).all():
            return False, '原始關鍵點無效'
        delta = (hand[9] - hand[0]) * np.array([sample.width, sample.height, sample.width])
        if np.linalg.norm(delta) < 10:
            return False, '手掌尺度小於10px'
    return True, ''

def feature(sample, config, normalizer=None):
    valid, reason = raw_valid(sample, config)
    if not valid:
        return FeatureResult(False, reason=reason, featureVersion=config['feature_version'])
    result = (normalizer or StudentNormalizer()).transform(sample, config)
    if not isinstance(result, FeatureResult):
        raise ValueError('StudentNormalizer.transform 必須回傳 FeatureResult')
    if not result.valid:
        if result.features is not None:
            raise ValueError('無效特徵必须為 None')
        return result
    d = 63
    if result.featureVersion != config['feature_version'] or not isinstance(result.features, np.ndarray) or result.features.shape != (d,) or result.features.dtype != np.float32 or not np.isfinite(result.features).all():
        raise ValueError('正規化輸出版本／維度／float32／有限值不符接口')
    return result

def validate_prediction(pred):
    if not isinstance(pred, Prediction) or pred.label not in LABELS or not np.isfinite(pred.vote_ratio) or not 0 <= pred.vote_ratio <= 1:
        raise ValueError('StudentKNN.predict 的 Prediction 無效')
    if pred.nearest_distance is not None and (not np.isfinite(pred.nearest_distance) or pred.nearest_distance < 0):
        raise ValueError('nearest_distance 必須非負有限值或 None')
    return pred.label if pred.valid else 'NONE'

class OutputRouter:
    def __init__(self):
        self.lock = threading.Lock()
        self.label, self.generated = 'NONE', -float('inf')
        self.error = ''
        self.filter = StudentFilter()

    def clear(self, enabled=False):
        with self.lock:
            self.label, self.generated = 'NONE', time.monotonic()
        if enabled:
            self.filter.reset()

    def update(self, prediction=None, tracked=False, enabled=False):
        now = time.monotonic()
        try:
            label = validate_prediction(prediction) if tracked else 'NONE'
            if not tracked and enabled:
                self.filter.reset()
            if tracked and enabled:
                label = self.filter.update(prediction, now)
                if label not in LABELS:
                    raise ValueError('StudentFilter.update 必須回傳固定標籤')
            self.error = ''
        except Exception as exc:
            label, self.error = 'NONE', str(exc)
        with self.lock:
            self.label, self.generated = label, now
        return label

    def read(self):
        with self.lock:
            return self.label if time.monotonic() - self.generated <= .5 else 'NONE'
