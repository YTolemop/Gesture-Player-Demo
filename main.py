import argparse
import json
import platform
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def config():
    path=ROOT/'config.json'
    settings=json.loads((path if path.exists() else ROOT/'config.example.json').read_text(encoding='utf-8'))
    settings.update(hand_mode='single',single_hand='right',mirror_left=False,feature_version='right-hand-normalization-v2')
    return settings

def main():
    parser=argparse.ArgumentParser(description='手勢工作坊玩家工具（學員算法保留TODO）')
    parser.add_argument('command',choices=['gui','record','train','test','prepare','check'])
    args=parser.parse_args();settings=config()
    if args.command=='prepare':
        from core.resources import prepare
        print(json.dumps(prepare()));return
    if args.command=='check':
        import cv2,numpy,mediapipe,socketio,tkinter
        from core.landmarks import Landmarks
        detector=Landmarks();detector.close()
        temporary=ROOT/'data/.write-check';temporary.parent.mkdir(exist_ok=True);temporary.write_text('ok');temporary.unlink()
        print(json.dumps({'python':sys.version,'architecture':platform.architecture()[0],'platform':platform.platform(),'cv2':cv2.__version__,'numpy':numpy.__version__,'mediapipe':mediapipe.__version__,'model_load':'PASS','camera':'NOT_RUN','student_algorithms':'TODO'},ensure_ascii=False));return
    if args.command in ('gui','record'):
        from ui.app import App
        app=App(settings)
        if args.command=='record':app.tabs.select(1)
        app.run();return
    if args.command in ('train','test'):
        from core.model_io import train,evaluate
        result=(train if args.command=='train' else evaluate)(list((ROOT/'data').glob('*.npz')),ROOT/'models/knn.npz',settings);print(json.dumps(result,ensure_ascii=False,indent=2));return
if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:pass
    except Exception as exc:print(f'工具提示：{exc}',file=sys.stderr);sys.exit(1)
