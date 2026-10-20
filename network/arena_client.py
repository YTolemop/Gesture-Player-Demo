"""Socket.IO transport. Recognition belongs in student/pipeline.py."""
import threading
import time
import uuid
import socketio
from network.commands import Commands

NAMESPACE = "/controller"


class ArenaClient:
    def __init__(self):
        self.sio = socketio.Client(reconnection=False)
        self.lock = threading.RLock()
        self.context = {}
        self.revision = 0
        self.seq = 0
        self.neutral_until = 0
        self.last_heartbeat = 0
        self.last_preview = float("-inf")
        self.last_send = float("-inf")
        self.last_commands = None
        self.last_sent_revision = -1
        self.last_result = None
        self.send_error = None
        self.sio.on("controller:context", self._context, namespace=NAMESPACE)
        self.sio.on("disconnect", self._disconnect, namespace=NAMESPACE)

    def _context(self, value):
        with self.lock:
            changed = any(self.context.get(k) != value.get(k) for k in
                          ("sessionId", "matchId", "generation", "phase"))
            if self.context.get("sessionId") != value.get("sessionId"):
                self.seq = 0
            self.context.update(value)
            if changed:
                self.revision += 1
                self.neutral_until = time.monotonic() + .25
                self.last_result = None
                self.send_error = None

    def _disconnect(self, reason=None):
        with self.lock:
            self.context = {}
            self.revision += 1

    def call(self, event, payload):
        result = self.sio.call(event, {**payload, "requestId": str(uuid.uuid4())},
                               namespace=NAMESPACE, timeout=5)
        if not result.get("ok"):
            raise RuntimeError(result.get("errorCode", "ACK_ERROR"))
        return result

    def connect(self, server, room, pair_code, algorithms):
        try:
            self.sio.connect(server, namespaces=[NAMESPACE], socketio_path="socket.io")
            result = self.call("controller:pair", {
                "protocolVersion": 1, "roomCode": room, "pairCode": pair_code,
                "source": "recognizer", "algorithms": algorithms})
            self._context(result)
            return result
        except Exception:
            self.close()
            raise

    def send(self, commands: Commands, revision=None):
        if not isinstance(commands, Commands):
            raise TypeError("Pipeline.predict must return Commands")
        with self.lock:
            c = self.context
            if not self.sio.connected or not c.get("sessionId"):
                raise RuntimeError("Arena disconnected; generate a new pair code and run again")
            if self.send_error:
                raise RuntimeError(self.send_error)
            if (revision is not None and revision != self.revision) or time.monotonic() < self.neutral_until:
                commands = Commands()
            now = time.monotonic()
            # Preserve every transition (especially NONE releases); throttle only
            # unchanged states. Waiting for an ACK here used to stall inference.
            if (commands == self.last_commands and self.last_sent_revision == self.revision
                    and now - self.last_send < .1):
                return
            payload = {"protocolVersion": 1, "sessionId": c["sessionId"],
                       "matchId": c.get("matchId"), "generation": c["generation"],
                       "seq": self.seq, "commands": commands.payload()}
            self.seq += 1
            sent_revision = self.revision

            def acknowledged(result):
                with self.lock:
                    if sent_revision != self.revision:
                        return
                    if self.last_result and payload["seq"] < self.last_result["seq"]:
                        return
                    self.last_result = {**result, "seq": payload["seq"]}
                    if not result.get("ok") and result.get("errorCode") not in (
                            "STALE_GENERATION", "STALE_MATCH", "INVALID_SESSION"):
                        self.send_error = result.get("errorCode", "INPUT_REJECTED")

            self.sio.emit("input:state", payload, namespace=NAMESPACE, callback=acknowledged)
            self.last_commands = commands
            self.last_send = now
            self.last_sent_revision = self.revision
            if time.monotonic() - self.last_heartbeat >= .5:
                self.sio.emit("client:heartbeat", {}, namespace=NAMESPACE)
                self.last_heartbeat = time.monotonic()

    def send_preview(self, frame, sample, revision=None):
        """Send at most one current preview per 150ms, without an ACK wait."""
        from network.preview import arena_preview

        with self.lock:
            now = time.monotonic()
            c = self.context
            if (not self.sio.connected or not c.get("sessionId") or
                    "generation" not in c or
                    (revision is not None and revision != self.revision) or
                    now - self.last_preview < .15):
                return
            self.last_preview = now
            payload = {"protocolVersion": 1, "sessionId": c["sessionId"],
                       "generation": c["generation"], **arena_preview(frame, sample)}
            try:
                self.sio.emit("camera:frame", payload, namespace=NAMESPACE)
            except socketio.exceptions.BadNamespaceError:
                # Disconnect may race the preview; gameplay handles connection loss.
                return

    def close(self):
        if self.sio.connected:
            try:
                self.send(Commands())
            except Exception:
                pass
            self.sio.disconnect()
