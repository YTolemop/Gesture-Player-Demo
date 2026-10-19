"""Completed Demo: raw MediaPipe -> independent left/right/both KNN banks."""
import json
from pathlib import Path
import numpy as np
from network.commands import Commands
from student.data import channel_slots, valid_slots, resolve_hands, records
from student.normalization import StudentNormalizer
from student.knn import StudentKNN
from student.filtering import StudentFilter
from student.labels import LABELS
from student.types import Prediction

ROOT = Path(__file__).resolve().parents[1]
CHANNELS = ("left", "right", "both")


def settings():
    path = ROOT / "config.json"
    config = json.loads((ROOT / "config.example.json").read_text(encoding="utf-8"))
    if path.exists():
        config.update(json.loads(path.read_text(encoding="utf-8")))
    return config


def features(slots, channel, config):
    slots = channel_slots(slots, channel)
    if not valid_slots(slots, channel):
        return None
    result = StudentNormalizer().transform(slots, config)
    if not result.valid or not np.isfinite(result.features).all():
        return None
    return result.features


def train(data_dir, model_path):
    config = settings()
    banks = {channel: ([], []) for channel in CHANNELS}
    rejected = {channel: 0 for channel in CHANNELS}
    batches = set()
    for slots, label, channel, batch in records(data_dir, "train", config):
        batches.add(batch)
        vector = features(slots, channel, config)
        if vector is None:
            rejected[channel] += 1
            continue
        banks[channel][0].append(vector)
        banks[channel][1].append(label)
    payload = {}
    counts = {}
    for channel, (x, y) in banks.items():
        if not x:
            raise ValueError(f"No usable {channel} training samples")
        payload[channel + "_features"] = np.asarray(x, dtype=np.float32)
        payload[channel + "_labels"] = np.asarray(y, dtype=np.int64)
        counts[channel] = len(y)
    # A single file binds settings and all banks into one loadable model.
    payload["metadata"] = np.asarray(json.dumps({"format": "demo-three-channel-v1", "config": config, "labels": LABELS, "training_batches": sorted(batches)}))
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    report = {"training_samples": counts, "rejected": rejected, "model": str(model_path)}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


class Pipeline:
    algorithms = {"normalization": True, "knn": True, "filterEnabled": False, "filter": True}

    def __init__(self, model_path=None):
        self.models = {}
        with np.load(model_path or ROOT / "models/model.npz", allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            if metadata["format"] != "demo-three-channel-v1" or metadata["labels"] != LABELS:
                raise ValueError("Unsupported Demo model format; run train")
            self.config = metadata["config"]
            for channel in CHANNELS:
                x, y = data[channel + "_features"], data[channel + "_labels"]
                if x.ndim != 2 or x.shape[1] != 131 or not len(x) or not np.isfinite(x).all() or y.shape != (len(x),) or not np.issubdtype(y.dtype, np.integer) or ((y < 0) | (y >= len(LABELS)-1)).any():
                    raise ValueError(f"Invalid {channel} model data")
                config = {**self.config, "k": min(int(self.config["k"]), len(x))}
                if config["k"] < 1:
                    raise ValueError("k must be positive")
                model = StudentKNN()
                model.fit(x, y, config)
                self.models[channel] = model
        # Filtering can be toggled without changing saved feature semantics.
        self.filter_enabled = bool(settings().get("filter_enabled", False))
        self.algorithms = {**self.algorithms, "filterEnabled": self.filter_enabled}
        self.filters = {channel: StudentFilter() for channel in CHANNELS}
        self.last_predictions = {}
        self.last_commands = Commands()

    def reset(self):
        for temporal_filter in self.filters.values():
            temporal_filter.reset()
        self.last_predictions = {}
        self.last_commands = Commands()

    def classify(self, slots, channel):
        vector = features(slots, channel, self.config)
        return self.models[channel].predict(vector) if vector is not None else Prediction()

    def decide(self, slots, filtered=True):
        predictions = {channel: self.classify(slots, channel) for channel in CHANNELS}
        self.last_predictions = predictions
        # Accepted joint gestures have exclusive priority, even during filter warmup.
        # This prevents single-hand actions from leaking out of a joint gesture.
        joint = predictions["both"]
        if joint.valid:
            self.filters["left"].reset()
            self.filters["right"].reset()
            label = self.filters["both"].update(joint, slots.timestamp) if filtered and self.filter_enabled else joint.label
            return Commands(both=label)
        self.filters["both"].reset()
        outputs = {}
        for channel in ("left", "right"):
            prediction = predictions[channel]
            if filtered and self.filter_enabled:
                outputs[channel] = self.filters[channel].update(prediction, slots.timestamp)
            else:
                outputs[channel] = prediction.label if prediction.valid else "NONE"
        return Commands(**outputs)

    def predict(self, sample):
        self.last_commands = self.decide(resolve_hands(sample, self.config))
        return self.last_commands


def evaluate(data_dir, model_path, output=None):
    pipeline = Pipeline(model_path)
    matrices = {channel: np.zeros((len(LABELS), len(LABELS)), dtype=int) for channel in CHANNELS}
    rejected = {channel: 0 for channel in CHANNELS}
    # Evaluate the final router, without temporal smoothing, on independent batches.
    for slots, label, channel, batch in records(data_dir, "test", pipeline.config):
        commands = pipeline.decide(slots, filtered=False)
        predicted = getattr(commands, channel)
        if channel != "both" and commands.both != "NONE":
            predicted = commands.both  # A joint takeover is a real routing error.
        matrices[channel][label, LABELS.index(predicted)] += 1
        if features(slots, channel, pipeline.config) is None:
            rejected[channel] += 1
    report = {"independent_test": True, "temporal_filter": False, "labels": LABELS,
              "channels": {}}
    for channel, matrix in matrices.items():
        total = int(matrix.sum())
        report["channels"][channel] = {"samples": total, "accuracy": float(np.trace(matrix) / total) if total else None,
            "invalid_samples": rejected[channel], "confusion_matrix": matrix.tolist()}
    if not any(v["samples"] for v in report["channels"].values()):
        raise ValueError("No independent test samples")
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report
