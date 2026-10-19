import time
import pytest
from network.arena_client import ArenaClient
from network.commands import Commands


class Transport:
    connected = True
    def __init__(self):
        self.calls = []
        self.events = []
    def call(self, event, value, **kwargs):
        self.calls.append((event, value))
        return {"ok": True, "seq": value.get("seq")}
    def emit(self, event, value, **kwargs):
        self.events.append((event, value))
    def disconnect(self):
        self.connected = False


def client():
    c = ArenaClient()
    c.sio = Transport()
    c._context({"sessionId": "one", "matchId": "match", "generation": 1, "phase": "PLAYING"})
    return c


def test_atomic_payload_sequence_neutral_gate_and_context_reset():
    c = client()
    c.send(Commands("MOVE_LEFT", "BLOCK"))
    assert c.sio.calls[-1][1]["commands"] == Commands().payload()
    c.neutral_until = time.monotonic() - 1
    revision = c.revision
    c.send(Commands("MOVE_LEFT", "BLOCK"), revision)
    assert c.sio.calls[-1][1]["commands"]["right"] == "BLOCK"
    assert c.sio.calls[-1][1]["seq"] == 1
    c._context({"sessionId": "two", "matchId": "new", "generation": 2, "phase": "PLAYING"})
    c.neutral_until = 0
    c.send(Commands(both="JUMP"), revision)
    assert c.sio.calls[-1][1]["commands"] == Commands().payload()
    assert c.sio.calls[-1][1]["seq"] == 0
    assert c.sio.calls[-1][1]["generation"] == 2
    assert c.sio.events[0][0] == "client:heartbeat"


def test_disconnect_and_close_clear_output():
    c = client()
    c.close()
    assert c.sio.calls[-1][1]["commands"] == Commands().payload()
    c._disconnect()
    with pytest.raises(RuntimeError, match="disconnected"):
        c.send(Commands())
