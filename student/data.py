"""Read existing NPZ batches without rewriting them; also read raw JSONL."""
import json
from pathlib import Path
import numpy as np
from core.types import RawHand, RawHandSample
from student.labels import LABELS
from student.types import HandSlots


def resolve_hands(sample, config):
    slots = {"left": None, "right": None}
    seen = set()
    conflicts = set()
    for hand in sample.hands:
        if not hand.handedness:
            continue
        category = max(hand.handedness, key=lambda c: c["score"])
        if not np.isfinite(category["score"]) or category["score"] < config["handedness_threshold"]:
            continue
        side = str(category["category_name"]).lower()
        if side not in slots:
            continue
        if config["swap_handedness"]:
            side = "right" if side == "left" else "left"
        if side in seen:
            conflicts.add(side)
        seen.add(side)
        slots[side] = np.asarray(hand.landmarks, dtype=np.float32)
    for side in conflicts:
        slots[side] = None
    return HandSlots(slots["left"], slots["right"], sample.width, sample.height, sample.timestamp)


def channel_slots(slots, channel):
    if channel == "left":
        return HandSlots(slots.left, None, slots.width, slots.height, slots.timestamp)
    if channel == "right":
        return HandSlots(None, slots.right, slots.width, slots.height, slots.timestamp)
    if channel == "both":
        return slots
    raise ValueError(f"Unknown channel: {channel}")


def valid_slots(slots, channel):
    required = (slots.left, slots.right) if channel == "both" else (
        slots.left if channel == "left" else slots.right,)
    if slots.width <= 0 or slots.height <= 0:
        return False
    for hand in required:
        if hand is None or hand.shape != (21, 3) or not np.isfinite(hand).all():
            return False
        scale = np.linalg.norm((hand[9] - hand[0]) * [slots.width, slots.height, slots.width])
        if scale < 10:
            return False
    return True


def command_label(label, config):
    command = config.get("gesture_commands", {}).get(label, label)
    if command not in LABELS or command == "NONE":
        raise ValueError(f"Gesture {label!r} needs a valid gesture_commands mapping")
    return LABELS.index(command)


def records(data_dir, purpose, config):
    """Yield slots, label ID, channel, batch name. Only requested purpose is read."""
    for path in sorted(Path(data_dir).glob("*.npz")):
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if meta.get("purpose") != purpose:
            continue
        if meta.get("capture_version") != "image-mirrored-v1":
            raise ValueError(f"Unsupported NPZ capture format: {path.name}")
        with np.load(path, allow_pickle=False) as data:
            raw, present, labels = data["raw"], data["present"], data["labels"]
        if raw.shape != (len(labels), 2, 21, 3) or present.shape != (len(labels), 2) or len(meta["samples"]) != len(labels):
            raise ValueError(f"Invalid NPZ shape: {path.name}")
        for i, metadata in enumerate(meta["samples"]):
            if metadata.get("capture_version") != "image-mirrored-v1" or not metadata.get("image_mirrored"):
                raise ValueError(f"Invalid capture metadata: {path.name}")
            label = int(labels[i])
            if not 0 <= label < len(LABELS) - 1 or label != labels[i]:
                raise ValueError(f"Invalid training label: {path.name}")
            channel = config["label_channels"][LABELS[label]]
            # These slots were already assigned (and swapped) at recording time.
            # Do not swap again or fabricate unavailable world coordinates/scores.
            slots = HandSlots(raw[i, 0] if present[i, 0] else None,
                              raw[i, 1] if present[i, 1] else None,
                              metadata["width"], metadata["height"], metadata["timestamp"])
            # A single-hand class recorded on the other side is mislabeled data.
            if channel == "left" and slots.right is not None or channel == "right" and slots.left is not None:
                slots = HandSlots(None, None, slots.width, slots.height, slots.timestamp)
            yield channel_slots(slots, channel), label, channel, path.stem
    # New recordings explicitly tag train/test; untagged initial records are train.
    for path in sorted(Path(data_dir).glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("purpose", "train") != purpose:
                continue
            if row.get("capture_version") != "mediapipe-raw-v1" or not row.get("image_mirrored"):
                raise ValueError(f"Unsupported JSONL capture format: {path.name}")
            sample = RawHandSample([RawHand(**h) for h in row["hands"]], row["width"], row["height"], row["timestamp"])
            channel = row["channel"]
            yield channel_slots(resolve_hands(sample, config), channel), command_label(row["label"], config), channel, path.stem
