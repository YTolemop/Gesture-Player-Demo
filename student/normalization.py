import numpy as np
from core.types import RawHandSample, FeatureResult


class StudentNormalizer:
    def transform(self, sample: RawHandSample, config: dict) -> FeatureResult:
        if sample.right is None:
            return FeatureResult(False, reason='Demo 使用右手模型', featureVersion=config['feature_version'])
        # 換成一致的像素尺度；建立副本，不修改輸入。
        points = np.array(sample.right, dtype=np.float64, copy=True)
        points[:, 0] *= sample.width
        points[:, 1] *= sample.height
        points[:, 2] *= sample.width

        # 以手腕為原點，再除以手掌尺度。
        centered = points - points[0]
        palm_size = np.linalg.norm(centered[9])
        normalized = centered / palm_size

        features = normalized.reshape(63).astype(np.float32)
        return FeatureResult(
            valid=True,
            features=features,
            featureVersion=config['feature_version'],
        )
