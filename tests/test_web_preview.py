import base64

import cv2
import numpy as np
import pytest

from core.types import RawHand, RawHandSample
from network.preview import arena_preview
from test_transport import client


def sample(width=640, height=480, mirrored=True, hands=True):
    hand = RawHand([[.2, .3, 0]] * 21, [], [])
    return RawHandSample([hand] if hands else [], width, height, 1,
                         image_mirrored=mirrored)


@pytest.mark.parametrize("mirrored", [True, False])
@pytest.mark.parametrize("shape", [(480, 640), (360, 640), (640, 480)])
def test_preview_image_and_landmarks_share_unmirrored_letterbox(shape, mirrored):
    height, width = shape
    frame = np.zeros((height, width, 3), np.uint8)
    frame[:, :width // 2] = [0, 0, 255]
    frame[:, width // 2:] = [255, 0, 0]
    original = frame.copy()
    raw = sample(width, height, mirrored)
    payload = arena_preview(frame, raw)
    encoded = payload["image"].split(",", 1)[1]
    decoded = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (240, 320, 3)
    assert len(payload["image"]) <= 48000
    scaled_w = round(width * min(320 / width, 240 / height))
    scaled_h = round(height * min(320 / width, 240 / height))
    left, top = (320 - scaled_w) // 2, (240 - scaled_h) // 2
    assert payload["hands"][0][0] == pytest.approx({
        "x": (left + (.8 if mirrored else .2) * scaled_w) / 320,
        "y": (top + .3 * scaled_h) / 240,
    })
    # Colored halves reveal accidental double mirroring, even after JPEG loss.
    assert decoded[120, left + scaled_w // 4].argmax() == (0 if mirrored else 2)
    if top:
        assert decoded[0].max() < 10
    if left:
        assert decoded[:, 0].max() < 10
    np.testing.assert_array_equal(frame, original)
    assert raw.hands[0].landmarks[0] == [.2, .3, 0]


def test_size_bound_and_encoding_failure_preserve_empty_hands(monkeypatch):
    frame = np.random.default_rng(0).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    payload = arena_preview(frame, sample(hands=False))
    assert payload["hands"] == [] and len(payload["image"]) <= 48000
    monkeypatch.setattr(cv2, "imencode", lambda *args: (False, None))
    assert arena_preview(frame, sample(hands=False)) == {"hands": []}


def test_preview_rate_limit_context_and_disconnect(monkeypatch):
    c = client()
    clock = [10.0]
    monkeypatch.setattr("network.arena_client.time.monotonic", lambda: clock[0])
    frame = np.zeros((480, 640, 3), np.uint8)
    raw = sample(hands=False)
    revision = c.revision
    c.send_preview(frame, raw, revision)
    event, payload = c.sio.events[-1]
    assert event == "camera:frame"
    assert set(payload) == {"protocolVersion", "sessionId", "generation", "image", "hands"}
    assert payload["sessionId"] == "one" and payload["generation"] == 1
    assert payload["hands"] == []
    clock[0] += .1
    c.send_preview(frame, raw, revision)
    assert len(c.sio.events) == 1
    clock[0] += .1
    c._context({"sessionId": "two", "generation": 2})
    c.send_preview(frame, raw, revision)
    assert len(c.sio.events) == 1  # Never send a previous context's frame.
    c.send_preview(frame, raw, c.revision)
    assert c.sio.events[-1][1]["sessionId"] == "two"
    assert c.sio.events[-1][1]["generation"] == 2
    c._disconnect()
    clock[0] += 1
    c.send_preview(frame, raw)
    assert len(c.sio.events) == 2
    assert not c.sio.calls  # Preview never waits for gameplay ACKs.
