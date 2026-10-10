import json
from pathlib import Path
LABELS = json.loads((Path(__file__).resolve().parents[1] / 'protocol/gesture_labels.json').read_text(encoding='utf-8'))['labels']
