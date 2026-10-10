import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'assets/models/hand_landmarker.task'
URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
SHA256 = 'fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1'

def verify_model():
    if not MODEL.is_file() or hashlib.sha256(MODEL.read_bytes()).hexdigest() != SHA256:
        raise RuntimeError('Hand Landmarker 模型缺失或雜湊不符，請執行 prepare')
    return {'path':str(MODEL),'sha256':SHA256}

def prepare():
    MODEL.parent.mkdir(parents=True,exist_ok=True)
    if not MODEL.is_file():
        temporary=MODEL.with_suffix('.download')
        urllib.request.urlretrieve(URL,temporary)
        if hashlib.sha256(temporary.read_bytes()).hexdigest()!=SHA256:
            temporary.unlink();raise RuntimeError('下載模型雜湊不符')
        temporary.replace(MODEL)
    return verify_model()
