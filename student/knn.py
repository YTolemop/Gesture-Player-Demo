import numpy as np
from core.labels import LABELS
from core.types import Prediction


class StudentKNN:
    def fit(self, features, labels, config):
        self.training_features = features.copy()
        self.training_labels = labels.copy()
        self.k = int(config.get('k', 5))
        self.vote_threshold = float(config.get('vote_threshold', 0.6))
        self.distance_threshold = float(config.get('distance_threshold', 1.0))
        return None

    def predict(self, features):
        differences = self.training_features - features
        distances = np.linalg.norm(differences, axis=1)

        order = np.argsort(distances, kind='stable')
        neighbor_indices = order[:self.k]
        neighbor_labels = self.training_labels[neighbor_indices]

        counts = np.bincount(neighbor_labels, minlength=len(LABELS))
        class_id = int(np.argmax(counts))
        ratio = float(counts[class_id] / self.k)
        distance = float(distances[neighbor_indices[0]])

        valid = (
            distance <= self.distance_threshold
            and ratio >= self.vote_threshold
        )
        label = LABELS[class_id] if valid else 'NONE'
        return Prediction(
            label=label,
            valid=valid,
            nearest_distance=distance,
            vote_ratio=ratio,
        )
