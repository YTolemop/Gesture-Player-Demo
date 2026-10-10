import cv2
import threading
import time
import sys

def camera_backend(platform=None):
    platform = sys.platform if platform is None else platform
    if platform == 'win32':
        return cv2.CAP_DSHOW
    if platform == 'darwin':
        return cv2.CAP_AVFOUNDATION
    return cv2.CAP_ANY

class Camera:
    """Separate capture thread, single latest frame, no backlog. Preview is mirrored
    together with inference so handedness/coordinates share one contract."""
    def __init__(self, index=0):
        backend = camera_backend()
        self.cap = cv2.VideoCapture(index, backend)
        if not self.cap.isOpened() and backend != cv2.CAP_ANY:
            self.cap.release()
            self.cap = cv2.VideoCapture(index, cv2.CAP_ANY)
        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError('攝影機無法開啟；檢查索引、系統攝影機權限或其他程式占用')
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.actual = {'index': index, 'width': self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), 'height': self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT), 'fps': self.cap.get(cv2.CAP_PROP_FPS)}
        self.lock = threading.Lock()
        self.frame, self.timestamp, self.error = None, 0.0, ''
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._capture, daemon=True)
        self.thread.start()

    def _capture(self):
        while not self.stop.is_set():
            ok, frame = self.cap.read()
            with self.lock:
                self.frame = cv2.flip(frame, 1) if ok else None
                self.timestamp = time.monotonic()
                self.error = '' if ok else '攝影機讀取失敗'
            if not ok:
                self.stop.wait(.1)

    def latest(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None, self.timestamp

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)
        self.cap.release()
