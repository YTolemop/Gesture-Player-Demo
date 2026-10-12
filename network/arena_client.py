import threading
import time
import uuid
import socketio
from network.preview import encode_preview

class ArenaClient:
    def __init__(self, router, on_status=lambda _: None, reset=lambda: None):
        self.router, self.on_status, self.reset = router, on_status, reset
        self.sio=socketio.Client(reconnection=True, logger=False, engineio_logger=False)
        self.pending_preview=None; self.last_preview=0.0
        self.context={}; self.token=None; self.room=None; self.seq=0; self.lock=threading.RLock(); self.stop=threading.Event()
        self.sio.on('controller:context',self._context,namespace='/controller')
        self.sio.on('connect',self._connected,namespace='/controller')
        self.sio.on('disconnect',self._disconnected,namespace='/controller')
        self.thread=threading.Thread(target=self._send,daemon=True);self.thread.start()

    @property
    def paired(self):
        with self.lock:
            return bool(self.token and not self.stop.is_set() and self.sio.connected
                        and '/controller' in self.sio.namespaces and self.context.get('sessionId'))

    def call(self,event,payload):
        result=self.sio.call(event,{**payload,'requestId':str(uuid.uuid4())},namespace='/controller',timeout=5)
        if not result.get('ok'):
            raise RuntimeError(result.get('errorCode','ACK_ERROR'))
        return result

    def connect(self,server,room,pair_code,source='recognizer',algorithms=None):
        self.room=room;self.sio.connect(server,namespaces=['/controller'],socketio_path='socket.io')
        result=self.call('controller:pair',{'protocolVersion':1,'roomCode':room,'pairCode':pair_code,'source':source,'algorithms':algorithms or {}})
        with self.lock:
            if self.stop.is_set():
                raise RuntimeError('配對已解除，請重新產生配對碼')
            self.room=result['roomCode'];self.token=result['controllerToken']
            self.context={**self.context,**result};self.seq=0;self.pending_preview=None;self.last_preview=0.0
        self.router.clear();self.on_status(f"已配對 {result['seat']}，等待新 NONE 解除")
        return result

    def _connected(self):
        with self.lock:
            token,room=self.token,self.room
        if token and not self.stop.is_set():
            def resume():
                try:
                    result=self.call('controller:resume',{'roomCode':room,'controllerToken':token})
                    with self.lock:
                        if self.stop.is_set() or self.token!=token:return
                        self._context(result);self.on_status('已重新配對，請在網站確認準備並恢復')
                except Exception as exc:
                    self.close()
                    self.on_status(f'恢復配對失敗：{exc}；請重新產生配對碼')
            threading.Thread(target=resume,daemon=True).start()

    def _disconnected(self, reason=None):
        terminal=reason in (socketio.Client.reason.SERVER_DISCONNECT,socketio.Client.reason.CLIENT_DISCONNECT) or self.stop.is_set()
        with self.lock:
            self.context={};self.pending_preview=None;self.last_preview=0.0
            if terminal:
                self.token=None;self.room=None;self.stop.set()
        self.router.clear();self.reset()
        self.on_status('已解除配對，請重新產生配對碼' if terminal else '連線中斷，已停止輸出；等待重新連線')

    def _context(self,v):
        with self.lock:
            if self.stop.is_set():return
            old=self.context
            changed=any(old.get(k)!=v.get(k) for k in ('matchId','sessionId','generation','phase'))
            self.context={**old,**v}
            if old.get('sessionId')!=v.get('sessionId'):
                self.seq=0
            if changed:
                self.pending_preview=None;self.last_preview=0.0
                self.router.clear();self.reset()

    def preview(self, frame, sample):
        # Encoding is throttled; network sends run on the existing sender thread.
        if self.stop.is_set() or not self.sio.connected or '/controller' not in self.sio.namespaces:
            return
        now=time.monotonic()
        with self.lock:
            c=dict(self.context)
            if not c.get('sessionId') or c.get('source')!='recognizer' or now-self.last_preview<.15:
                return
            self.last_preview=now
        try:
            payload=encode_preview(frame, sample)
        except Exception:
            return  # Preview failures must never interrupt gesture recognition.
        if payload is None:
            return
        payload.update(protocolVersion=1, sessionId=c['sessionId'], generation=c.get('generation',1))
        with self.lock:
            if self.stop.is_set() or any(self.context.get(k)!=c.get(k) for k in ('sessionId','generation','phase','source')):
                return
            self.pending_preview=payload

    def _send(self):
        last_heartbeat=0
        while not self.stop.wait(.1):
            if not self.sio.connected or '/controller' not in self.sio.namespaces:
                continue
            with self.lock:
                c=dict(self.context)
                if not c.get('sessionId'):
                    continue
                v={'protocolVersion':1,'sessionId':c['sessionId'],'matchId':c.get('matchId'),'generation':c.get('generation',1),'seq':self.seq,'gesture':self.router.read()};self.seq+=1
            try:
                self.sio.emit('input:state',v,namespace='/controller')
                with self.lock:
                    preview=self.pending_preview;self.pending_preview=None
                    if preview and (preview['sessionId']!=self.context.get('sessionId') or preview['generation']!=self.context.get('generation') or self.context.get('source')!='recognizer'):
                        preview=None
                if preview:
                    self.sio.emit('camera:frame',preview,namespace='/controller')
                if time.monotonic()-last_heartbeat>=.5:
                    self.sio.emit('client:heartbeat',{},namespace='/controller');last_heartbeat=time.monotonic()
            except Exception as exc:
                self.on_status(str(exc))

    def availability(self,**flags):
        if self.paired:
            return self.call('controller:availability',flags)

    def close(self):
        self.router.clear()
        self.stop.set()
        with self.lock:
            c=dict(self.context)
            self.token=None;self.room=None;self.context={};self.pending_preview=None;self.last_preview=0.0
        self.thread.join(timeout=2)
        if self.sio.connected:
            if c.get('sessionId'):
                try:
                    self.sio.emit('input:state',{'protocolVersion':1,'sessionId':c['sessionId'],'matchId':c.get('matchId'),'generation':c.get('generation',1),'seq':self.seq,'gesture':'NONE'},namespace='/controller')
                except Exception:
                    pass
            self.sio.disconnect()
