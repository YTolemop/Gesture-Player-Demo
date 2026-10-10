from pathlib import Path
import json
import numpy as np
from core.dataset import read_batch
from core.features import feature, validate_prediction
from core.labels import LABELS
from student.knn import StudentKNN

def validate_training(x, y, config):
    if x.ndim!=2 or not len(x) or x.shape[1]!=63 or x.dtype!=np.float32 or not np.isfinite(x).all():
        raise ValueError('訓練特徵必須為非空有限 float32 (N,63)')
    if y.shape!=(len(x),) or not np.issubdtype(y.dtype,np.integer) or ((y<0)|(y>9)).any():
        raise ValueError('訓練標籤必須為整數 0..9')
    k=config.get('k',5)
    if isinstance(k,bool) or not isinstance(k,(int,np.integer)) or not 1<=k<=len(x):
        raise ValueError('k 必須介於 1 與樣本數之間')
    distance=config.get('distance_threshold',1.0);vote=config.get('vote_threshold',.6)
    if not np.isfinite(distance) or distance<0 or not np.isfinite(vote) or not 0<=vote<=1:
        raise ValueError('距離或投票門檻無效')

def features_from(paths, config, purpose):
    x, y, rejected = [], [], 0
    for path in paths:
        samples, labels, meta = read_batch(path)
        if meta['purpose'] != purpose:
            continue
        for sample, label in zip(samples, labels):
            result = feature(sample, config)
            if result.valid:
                x.append(result.features); y.append(label)
            else:
                rejected += 1
    if not x:
        raise ValueError(f'沒有可用的 {purpose} 特徵；檢查資料用途及正規化 TODO')
    return np.asarray(x,dtype=np.float32), np.asarray(y,dtype=np.int64), rejected

def save_model(path, x, y, config):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, features=x, labels=y)
    path.with_suffix('.json').write_text(json.dumps({'format':1,'labels':LABELS,'config':{k:v for k,v in config.items() if k!='filter_enabled'}},ensure_ascii=False,indent=2),encoding='utf-8')

def load_model(path, config):
    path=Path(path); meta=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    if meta != {'format':1,'labels':LABELS,'config':{k:v for k,v in config.items() if k!='filter_enabled'}}:
        raise ValueError('模型與特徵／手模式／標籤／設定版本不相容，請重算並訓練')
    with np.load(path, allow_pickle=False) as data:
        x, y = data['features'], data['labels']
    d=63
    if x.ndim!=2 or x.shape[1]!=d or y.shape!=(len(x),) or x.dtype!=np.float32 or not np.isfinite(x).all() or not np.issubdtype(y.dtype,np.integer) or ((y<0)|(y>9)).any():
        raise ValueError('模型資料格式無效')
    validate_training(x,y,config)
    model=StudentKNN(); model.fit(x,y,config); return model

def train(paths, path, config):
    x,y,rejected=features_from(paths,config,'train'); validate_training(x,y,config); model=StudentKNN(); model.fit(x,y,config); save_model(path,x,y,config)
    return {'training_samples':len(y),'normalization_rejected':rejected,'independent_test':False}

def evaluate(paths,path,config):
    model=load_model(path,config);x,y,rejected=features_from(paths,config,'test');matrix=np.zeros((11,11),dtype=int)
    for row,label in zip(x,y):
        pred=model.predict(row); value=validate_prediction(pred); matrix[label,LABELS.index(value) if value!='NONE' else 10]+=1
    recalls=[float(matrix[i,i]/matrix[i].sum()) if matrix[i].sum() else None for i in range(11)]
    return {'independent_test':True,'samples':len(y),'normalization_rejected':rejected,'labels':LABELS,'confusion_matrix':matrix.tolist(),'recall':recalls,'unknown_ratio':float(matrix[:,10].sum()/len(y))}
