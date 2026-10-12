import numpy as np
from core.types import RawHandSample, FeatureResult


class StudentNormalizer:
    def transform(self, sample: RawHandSample, config: dict) -> FeatureResult:
        features = np.zeros(131, dtype=np.float32)
        wrists = [None, None]
        sizes = [None, None]

        for slot, hand in enumerate((sample.left, sample.right)):
            if hand is None:
                continue
            points = np.array(hand, dtype=np.float64, copy=True)
            points *= np.array([sample.width, sample.height, sample.width])
            wrists[slot] = points[0].copy()
            centered = points - points[0]
            sizes[slot] = np.linalg.norm(centered[9])
            start = slot * 63
            features[start:start + 63] = (centered / sizes[slot]).reshape(63)
            features[126 + slot] = 2.0

        if wrists[0] is None and wrists[1] is None:
            return FeatureResult(
                valid=False, reason='無手',
                featureVersion=config['feature_version'],
            )
        if wrists[0] is not None and wrists[1] is not None:
            relative = (wrists[1] - wrists[0]) / ((sizes[0] + sizes[1]) / 2)
            relative[2] = 0.0  # 每隻手的z參考原點不同，只保留相對x、y。
            features[128:131] = relative

        return FeatureResult(
            valid=True, features=features,
            featureVersion=config['feature_version'],
        )