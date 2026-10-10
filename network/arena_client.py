import threading
import time
import uuid
import socketio

class ArenaClient:
    def __init__(self, router, on_status=lambda _: None, reset=lambda: None):
        self.router, self.on_status, self.reset = router, on_status, reset
        self.sio=socketio.Client(reconnection=True, logger=False, engineio_logger=False)
        self.context={}; self.token=None; self.room=None; self.seq=0; self.lock=threading.RLock(); self.stop=threading.Event()
        self.sio.on('controller:context',self._context,namespace='/controller')
        self.sio.on('connect',self._connected,namespace='/controller')
        self.sio.on('disconnect',lambda *args:self._disconnected(),namespace='/controller')
        self.thread=threading.Thread(target=self._send,daemon=True);self.thread.start()

    def call(self,event,payload):
        result=self.sio.call(event,{**payload,'requestId':str(uuid.uuid4())},namespace='/controller',timeout=5)
        if not result.get('ok'):
            raise RuntimeError(result.get('errorCode','ACK_ERROR'))
        return result

    def connect(self,server,room,pair_code,source='recognizer',algorithms=None):
        self.room=room;self.sio.connect(server,namespaces=['/controller'],socketio_path='socket.io')
        result=self.call('controller:pair',{'protocolVersion':1,'roomCode':room,'pairCode':pair_code,'source':source,'algorithms':algorithms or {}})
        self.room=result['roomCode']
        self.token=result['controllerToken']
        with self.lock:
            self.context={**self.context,**result};self.seq=0
        self.router.clear();self.on_status(f"已配對 {result['seat']}，等待新 NONE 解除")
        return result

    def _connected(self):
        if self.token:
            def resume():
                try:
                    result=self.call('controller:resume',{'roomCode':self.room,'controllerToken':self.token})
                    self._context(result);self.on_status('已重新配對，請在網站確認準備並恢復')
                except Exception as exc:
                    self.on_status(str(exc))
            threading.Thread(target=resume,daemon=True).start()

    def _disconnected(self):
        with self.lock:
            self.context={}
        self.router.clear();self.reset();self.on_status('已斷線，安全解除；等待重新連線')

    def _context(self,v):
        with self.lock:
            old=self.context
            changed=any(old.get(k)!=v.get(k) for k in ('matchId','sessionId','generation','phase'))
            self.context={**old,**v}
            if old.get('sessionId')!=v.get('sessionId'):
                self.seq=0
            if changed:
                self.router.clear();self.reset()

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
                if time.monotonic()-last_heartbeat>=.5:
                    self.sio.emit('client:heartbeat',{},namespace='/controller');last_heartbeat=time.monotonic()
            except Exception as exc:
                self.on_status(str(exc))

    def availability(self,**flags):
        if self.sio.connected:
            return self.call('controller:availability',flags)

    def close(self):
        self.router.clear()
        self.stop.set();self.thread.join(timeout=2)
        if self.sio.connected:
            with self.lock:
                c=dict(self.context)
            if c.get('sessionId'):
                try:
                    self.sio.emit('input:state',{'protocolVersion':1,'sessionId':c['sessionId'],'matchId':c.get('matchId'),'generation':c.get('generation',1),'seq':self.seq,'gesture':'NONE'},namespace='/controller')
                except Exception:
                    pass
            self.sio.disconnect()
