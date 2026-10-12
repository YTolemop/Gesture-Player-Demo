import base64
import threading
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from core.types import RawHandSample
from network.preview import encode_preview
from network.arena_client import ArenaClient


def sample(width=640, height=480, mirrored=True):
    hand = np.zeros((21, 3), dtype=np.float32)
    hand[:, 0] = .25
    hand[:, 1] = .5
    return RawHandSample(hand, None, width, height, 1.0, image_mirrored=mirrored)


def test_preview_unmirrors_image_and_points_without_mutating_capture():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, :320] = (0, 0, 255)
    frame[:, 320:] = (255, 0, 0)
    original = frame.copy()
    hand_sample = sample()
    payload = encode_preview(frame, hand_sample)
    jpeg = base64.b64decode(payload['image'].split(',', 1)[1])
    decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (240, 320, 3)
    assert decoded[120, 20, 0] > 240
    assert decoded[120, 300, 2] > 240
    assert payload['hands'][0][0] == pytest.approx({'x': .75, 'y': .5})
    assert np.array_equal(frame, original)
    assert hand_sample.left[0, 0] == .25


def test_letterbox_preserves_entire_image_and_skeleton_alignment():
    frame = np.full((360, 640, 3), 255, dtype=np.uint8)
    hand_sample = sample(640, 360, False)
    hand_sample.left[:, 1] = 0
    payload = encode_preview(frame, hand_sample)
    decoded = cv2.imdecode(np.frombuffer(base64.b64decode(payload['image'].split(',')[1]), np.uint8), cv2.IMREAD_COLOR)
    assert decoded[5, 160].max() < 5
    assert decoded[35, 160].min() > 240
    assert payload['hands'][0][0] == pytest.approx({'x': .25, 'y': .125})


def test_no_hands_and_invalid_coordinates():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    hand_sample = sample()
    hand_sample.left = None
    assert encode_preview(frame, hand_sample)['hands'] == []
    hand_sample = sample()
    hand_sample.left[0, 0] = np.nan
    assert encode_preview(frame, hand_sample)['hands'] == []
    hand_sample = sample()
    hand_sample.left[0, :2] = [-.1, 1.2]
    point = encode_preview(frame, hand_sample)['hands'][0][0]
    assert point == {'x': 1., 'y': 1.}


def client():
    c = ArenaClient.__new__(ArenaClient)
    c.lock = threading.RLock()
    c.stop = threading.Event()
    c.context = dict(sessionId='session', generation=2, source='recognizer', phase='PLAYING', matchId='match')
    c.pending_preview = None
    c.last_preview = 0
    c.seq = 0
    c.reset = lambda: None
    c.router = SimpleNamespace(clear=lambda: None, read=lambda: 'NONE')
    c.on_status = lambda _: None
    c.sio = SimpleNamespace(connected=True, namespaces=['/controller'])
    return c


def test_latest_only_queue_throttle_and_context_reset(monkeypatch):
    c = client()
    clock = [10.]
    monkeypatch.setattr('network.arena_client.time.monotonic', lambda: clock[0])
    monkeypatch.setattr('network.arena_client.encode_preview', lambda frame, _: {'hands': [], 'image': frame})
    c.preview('first', None)
    assert c.pending_preview['image'] == 'first'
    clock[0] += .05
    c.preview('too soon', None)
    assert c.pending_preview['image'] == 'first'
    clock[0] += .2
    c.preview('latest', None)
    assert c.pending_preview['image'] == 'latest'
    assert c.pending_preview['generation'] == 2
    c._context(dict(c.context, generation=3))
    assert c.pending_preview is None
    c.preview('new generation', None)
    assert c.pending_preview['generation'] == 3
    c._disconnected()
    assert c.pending_preview is None and c.context == {}


def test_sender_emits_preview_with_input_and_heartbeat_and_drops_stale():
    for generation, expected in [(2, True), (1, False)]:
        c = client()
        emitted = []
        c.sio.emit = lambda event, payload, namespace: emitted.append((event, payload))
        c.pending_preview = dict(protocolVersion=1, sessionId='session', generation=generation, hands=[])
        class OneIteration:
            count = 0
            def wait(self, _):
                self.count += 1
                return self.count > 1
        c.stop = OneIteration()
        c._send()
        events = [event for event, _ in emitted]
        assert ('camera:frame' in events) is expected
        assert 'input:state' in events and 'client:heartbeat' in events
        assert c.pending_preview is None


def test_preview_failure_never_interrupts_recognition(monkeypatch):
    c = client()
    def fail(*_):
        raise RuntimeError('encoder failure')
    monkeypatch.setattr('network.arena_client.encode_preview', fail)
    c.preview(None, None)
    assert c.pending_preview is None
    c.context['source'] = 'keyboard'
    c.last_preview = 0
    c.preview(None, None)
    assert c.last_preview == 0
