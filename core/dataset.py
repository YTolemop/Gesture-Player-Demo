import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
from core.types import RawHandSample, CAPTURE_VERSION

def save_batch(path, samples, labels, purpose, config):
    path = Path(path)
    if purpose not in ('train', 'test') or len(samples) != len(labels) or not samples or any(not isinstance(v,(int,np.integer)) or not 0<=v<=(10 if purpose=='test' else 9) for v in labels):
        raise ValueError('資料用途／樣本數無效')
    if path.exists():
        raise ValueError('批次已存在，請使用新批次名或先明確刪除')
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = np.zeros((len(samples), 2, 21, 3), dtype=np.float32)
    present = np.zeros((len(samples), 2), dtype=np.bool_)
    meta = []
    for i, s in enumerate(samples):
        for j, hand in enumerate((s.left, s.right)):
            if hand is not None:
                raw[i, j], present[i, j] = hand, True
        m = asdict(s)
        m.pop('left'); m.pop('right'); meta.append(m)
    np.savez_compressed(path, raw=raw, present=present, labels=np.asarray(labels, dtype=np.int64))
    path.with_suffix('.json').write_text(json.dumps({'purpose': purpose, 'capture_version': CAPTURE_VERSION, 'config': config, 'samples': meta}, ensure_ascii=False, indent=2), encoding='utf-8')

def read_batch(path):
    path = Path(path)
    meta = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    with np.load(path, allow_pickle=False) as data:
        raw, present, labels = data['raw'], data['present'], data['labels']
    if meta['capture_version'] != CAPTURE_VERSION or meta['purpose'] not in ('train','test') or raw.shape != (len(labels),2,21,3) or present.shape != (len(labels),2) or len(meta['samples']) != len(labels) or not np.isfinite(raw).all() or not np.issubdtype(labels.dtype,np.integer) or ((labels<0)|(labels>(10 if meta['purpose']=='test' else 9))).any():
        raise ValueError('資料格式／擷取版本不相容')
    return [RawHandSample(raw[i,0] if present[i,0] else None, raw[i,1] if present[i,1] else None, **m) for i,m in enumerate(meta['samples'])], labels, meta
