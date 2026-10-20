import time
import pytest
from network.arena_client import ArenaClient
from network.commands import Commands


class Transport:
    connected = True
    def __init__(self):
        self.calls = []
        self.events = []
        self.callbacks = []
    def call(self, event, value, **kwargs):
        self.calls.append((event, value))
        return {"ok": True, "seq": value.get("seq")}
    def emit(self, event, value, **kwargs):
        self.events.append((event, value))
        if "callback" in kwargs:
            self.callbacks.append(kwargs["callback"])
    def disconnect(self):
        self.connected = False


def client():
    c = ArenaClient()
    c.sio = Transport()
    c._context({"sessionId": "one", "matchId": "match", "generation": 1, "phase": "PLAYING"})
    return c


def packets(c):
    return [value for event, value in c.sio.events if event == "input:state"]


def test_atomic_payload_sequence_neutral_gate_and_context_reset():
    c = client()
    c.send(Commands("MOVE_LEFT", "BLOCK"))
    assert packets(c)[-1]["commands"] == Commands().payload()
    c.neutral_until = time.monotonic() - 1
    revision = c.revision
    c.send(Commands("MOVE_LEFT", "BLOCK"), revision)
    assert packets(c)[-1]["commands"]["right"] == "BLOCK"
    assert packets(c)[-1]["seq"] == 1
    c._context({"sessionId": "two", "matchId": "new", "generation": 2, "phase": "PLAYING"})
    c.neutral_until = 0
    c.send(Commands(both="JUMP"), revision)
    assert packets(c)[-1]["commands"] == Commands().payload()
    assert packets(c)[-1]["seq"] == 0
    assert packets(c)[-1]["generation"] == 2
    assert any(event == "client:heartbeat" for event, _ in c.sio.events)


def test_disconnect_and_close_clear_output():
    c = client()
    c.close()
    assert packets(c)[-1]["commands"] == Commands().payload()
    c._disconnect()
    with pytest.raises(RuntimeError, match="disconnected"):
        c.send(Commands())


def test_slow_ack_never_stalls_frames_or_loses_release_and_context(monkeypatch):
    c = client()
    c.neutral_until = 0
    clock = [10.0]
    monkeypatch.setattr("network.arena_client.time.monotonic", lambda: clock[0])
    c.send(Commands(left="JUMP"), c.revision)
    clock[0] += .03
    c.send(Commands(), c.revision)
    clock[0] += .03
    c.send(Commands(left="JUMP"), c.revision)
    assert [packet["commands"]["left"] for packet in packets(c)] == ["JUMP", "NONE", "JUMP"]
    # No ACK has arrived, but all three frames and a context update proceed.
    assert not c.sio.calls and c.last_result is None
    old_ack = c.sio.callbacks[-1]
    c._context({"sessionId": "two", "matchId": "new", "generation": 2})
    old_ack({"ok": False, "errorCode": "WRONG_SOURCE"})
    assert c.send_error is None
    c.send(Commands())
    assert packets(c)[-1]["sessionId"] == "two"
    c.sio.callbacks[-1]({"ok": True, "inputState": "NEUTRAL_REQUIRED"})
    assert c.last_result["inputState"] == "NEUTRAL_REQUIRED"


def test_unchanged_states_are_throttled_but_errors_are_reported(monkeypatch):
    c = client()
    c.neutral_until = 0
    clock = [10.0]
    monkeypatch.setattr("network.arena_client.time.monotonic", lambda: clock[0])
    c.send(Commands(right="MOVE_RIGHT"))
    clock[0] += .03
    c.send(Commands(right="MOVE_RIGHT"))
    assert len(packets(c)) == 1
    clock[0] += .08
    c.send(Commands(right="MOVE_RIGHT"))
    assert len(packets(c)) == 2
    c.sio.callbacks[-1]({"ok": False, "errorCode": "WRONG_SOURCE"})
    with pytest.raises(RuntimeError, match="WRONG_SOURCE"):
        c.send(Commands())
