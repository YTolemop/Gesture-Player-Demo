import json
import time
import numpy as np
import pytest
from core.types import RawHandSample, FeatureResult, Prediction
from core.features import feature, raw_valid, OutputRouter, validate_prediction
from core.dataset import save_batch, read_batch
from core.resources import verify_model

CONFIG={'hand_mode':'single','single_hand':'right','feature_version':'student-normalization-v1'}
def sample():
    hand=np.zeros((21,3),dtype=np.float32);hand[9]=[.1,.1,0]
    return RawHandSample(None,hand,640,480,time.monotonic())

def test_unfilled_normalization_is_explicit():
    class Unfilled:
        def transform(self,*args):raise NotImplementedError('StudentNormalizer.transform')
    with pytest.raises(NotImplementedError,match='StudentNormalizer.transform'):feature(sample(),CONFIG,Unfilled())

def test_no_hands_safely_bypass_student():
    s=sample();s.right=None
    result=feature(s,CONFIG);assert not result.valid and result.features is None

def test_raw_safety_and_scale():
    s=sample();assert raw_valid(s,CONFIG)[0]
    s.right[9]=0;assert not raw_valid(s,CONFIG)[0]
    s.right[9]=np.nan;assert not raw_valid(s,CONFIG)[0]

def test_feature_guard_invalid_shape_and_nan():
    class Bad:
        def transform(self,*_):return FeatureResult(True,np.zeros(128,dtype=np.float32))
    with pytest.raises(ValueError):feature(sample(),CONFIG,Bad())

def test_output_no_hand_and_unfinished_filter():
    class Unfilled:
        def reset(self):raise NotImplementedError('StudentFilter.reset')
        def update(self,*args):raise NotImplementedError('StudentFilter.update')
    r=OutputRouter();r.update(Prediction('BLOCK',True,0,1),True);assert r.read()=='BLOCK'
    r.filter=Unfilled()
    r.update(Prediction('ATTACK_SKILL',True,0,1),False,True);assert r.read()=='NONE'
    r.update(Prediction('BLOCK',True,0,1),True,True);assert r.read()=='NONE' and 'StudentFilter.update' in r.error
    r.generated-=1;assert r.read()=='NONE'

def test_prediction_guard():
    with pytest.raises(ValueError):validate_prediction(Prediction('BAD',True,0,1))
    with pytest.raises(ValueError):validate_prediction(Prediction('NONE',True,float('nan'),1))

def test_dataset_roundtrip_no_pickle_and_purpose(tmp_path):
    p=tmp_path/'資料 批次.npz';s=sample();save_batch(p,[s],[0],'test',CONFIG)
    samples,labels,meta=read_batch(p);assert labels.tolist()==[0] and meta['purpose']=='test'
    assert np.array_equal(samples[0].right,s.right)
    assert samples[0].left is None
    with pytest.raises(ValueError):save_batch(p,[s],[0],'train',CONFIG)

def test_model_hash():assert len(verify_model()['sha256'])==64

def test_protocol_all_eleven():
    from core.labels import LABELS
    assert len(LABELS)==11 and LABELS[-1]=='NONE'
    fixtures=json.loads(open('protocol/fixtures/inputs.json',encoding='utf-8').read());assert [v['gesture'] for v in fixtures['valid']]==LABELS
    import hashlib
    from pathlib import Path
    manifest=json.loads(Path('protocol/manifest.json').read_text(encoding='utf-8'))
    for path,digest in manifest['fileHashes'].items():assert hashlib.sha256((Path('protocol')/path).read_bytes()).hexdigest()==digest

def test_optional_unknown_is_test_only(tmp_path):
    p=tmp_path/'none.npz'
    with pytest.raises(ValueError):save_batch(p,[sample()],[10],'train',CONFIG)
    save_batch(p,[sample()],[10],'test',CONFIG)
    assert read_batch(p)[1].tolist()==[10]

def test_model_black_image_is_safe_no_hands():
    from core.landmarks import Landmarks
    tracker=Landmarks()
    try:
        s=tracker.detect(np.zeros((480,640,3),dtype=np.uint8),time.monotonic(),{})
        assert s.left is None and s.right is None
        assert not feature(s,CONFIG).valid
    finally:tracker.close()
