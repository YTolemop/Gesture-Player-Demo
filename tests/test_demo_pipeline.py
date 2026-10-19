import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from core.types import RawHand, RawHandSample
from network.commands import Commands
from student.data import records, resolve_hands
from student.labels import LABELS
from student.pipeline import Pipeline, evaluate, train
from student.types import HandSlots, Prediction

ROOT = Path(__file__).resolve().parents[1]


def raw_sample(slots, swap=True):
    hands = []
    for side, points in (("Left", slots.left), ("Right", slots.right)):
        if points is None:
            continue
        native = ("Right" if side == "Left" else "Left") if swap else side
        hands.append(RawHand(points.tolist(), [], [{"index": 0,
            "category_name": native, "display_name": "", "score": 0.99}]))
    return RawHandSample(hands, slots.width, slots.height, slots.timestamp)


@pytest.fixture
def pipeline():
    return Pipeline(ROOT / "models/model.npz")


def test_preserved_independent_recordings(pipeline):
    report = evaluate(ROOT / "data", ROOT / "models/model.npz")
    assert [report["channels"][c]["samples"] for c in ("left", "right", "both")] == [198, 258, 195]
    assert all(report["channels"][c]["accuracy"] == 1.0 for c in ("left", "right", "both"))
    with np.load(ROOT / "models/model.npz", allow_pickle=False) as data:
        meta = json.loads(str(data["metadata"]))
        assert all(batch.startswith("train-") for batch in meta["training_batches"])
        assert len(meta["training_batches"]) == 10


def test_300_composed_single_hand_pairs_can_send_both_channels(pipeline):
    samples = defaultdict(list)
    for slots, label, channel, batch in records(ROOT / "data", "test", pipeline.config):
        if channel != "both" and len(samples[(channel, label)]) < 5:
            samples[(channel, label)].append(slots)
    checked = 0
    for (lc, ly), lefts in samples.items():
        if lc != "left":
            continue
        for (rc, ry), rights in samples.items():
            if rc != "right":
                continue
            for left in lefts:
                for right in rights:
                    pipeline.reset()
                    slots = HandSlots(left.left, right.right, left.width, left.height, 1.0)
                    assert pipeline.predict(raw_sample(slots, pipeline.config["swap_handedness"])) == Commands(left=LABELS[ly], right=LABELS[ry])
                    checked += 1
    assert checked == 300


def test_joint_output_is_exclusive_and_missing_hands_clear(pipeline):
    slots, label, _, _ = next(r for r in records(ROOT / "data", "test", pipeline.config) if r[2] == "both")
    assert pipeline.predict(raw_sample(slots, pipeline.config["swap_handedness"])) == Commands(both=LABELS[label])
    assert pipeline.predict(RawHandSample([], 640, 480, 2.0)) == Commands()
    one = HandSlots(slots.left, None, slots.width, slots.height, 3.0)
    command = pipeline.predict(raw_sample(one, pipeline.config["swap_handedness"]))
    assert command.right == command.both == "NONE"


def test_raw_scores_conflicts_and_swap_are_student_policy(pipeline):
    hand = RawHand([[0.1, 0.2, 0.0]] * 21, [], [{"score": 0.99, "category_name": "Left"}])
    sample = RawHandSample([hand], 640, 480, 1.0)
    slots = resolve_hands(sample, {**pipeline.config, "swap_handedness": True})
    assert slots.left is None and slots.right is not None
    assert resolve_hands(RawHandSample([hand, hand], 640, 480, 1.0), pipeline.config).right is None
    hand.handedness[0]["score"] = 0.1
    slots = resolve_hands(sample, pipeline.config)
    assert slots.left is slots.right is None


def test_filters_are_independent_and_joint_warmup_does_not_leak(pipeline, monkeypatch):
    pipeline.filter_enabled = True
    valid = {"left": Prediction("MOVE_LEFT", True, 0.1, 1.0),
             "right": Prediction("NORMAL_ATTACK", True, 0.1, 1.0), "both": Prediction()}
    monkeypatch.setattr(pipeline, "classify", lambda slots, channel: valid[channel])
    slots = HandSlots(None, None, 640, 480, 1.0)
    assert pipeline.decide(slots) == Commands()
    slots.timestamp = 1.033
    assert pipeline.decide(slots) == Commands("MOVE_LEFT", "NORMAL_ATTACK")
    valid["right"] = Prediction()
    slots.timestamp = 1.066
    assert pipeline.decide(slots) == Commands(left="MOVE_LEFT")
    valid["both"] = Prediction("BLOCK", True, .1, 1.0)
    slots.timestamp = 1.099
    assert pipeline.decide(slots) == Commands()
    slots.timestamp = 1.132
    assert pipeline.decide(slots) == Commands(both="BLOCK")
    pipeline.reset()
    assert all(f.stable_label == "NONE" for f in pipeline.filters.values())


def test_jsonl_train_test_split_and_invalid_samples(pipeline, tmp_path, monkeypatch):
    import student.pipeline as module
    monkeypatch.setattr(module, "settings", lambda: pipeline.config)
    rows = []
    for channel in ("left", "right", "both"):
        slots, label, _, _ = next(r for r in records(ROOT / "data", "test", pipeline.config) if r[2] == channel)
        for purpose in ("train", "test"):
            rows.append({**asdict(raw_sample(slots, pipeline.config["swap_handedness"])),
                         "label": LABELS[label], "channel": channel, "purpose": purpose})
    rows.append({**asdict(RawHandSample([],640,480,1.0)), "label": "JUMP", "channel": "left", "purpose": "train"})
    (tmp_path / "new.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    assert len(list(records(tmp_path,"train",pipeline.config))) == 4
    assert len(list(records(tmp_path,"test",pipeline.config))) == 3
    report = train(tmp_path, tmp_path / "new-model.npz")
    assert report["training_samples"] == {"left":1,"right":1,"both":1}
    assert report["rejected"]["left"] == 1
    assert set(Pipeline(tmp_path / "new-model.npz").models) == {"left","right","both"}
