"""Student task acceptance: skips are TODO, not proof of completed algorithms."""
import numpy as np
import pytest
from student.normalization import StudentNormalizer
from student.knn import StudentKNN
from student.filtering import StudentFilter
from core.types import RawHandSample, Prediction

def test_normalization_contract():
    import json
    from pathlib import Path
    fixtures=json.loads((Path(__file__).parent/'fixtures/normalization.json').read_text())
    dimensions=set()
    for f in fixtures:
        s=RawHandSample(np.array(f['left'],dtype=np.float32) if f['left'] else None,np.array(f['right'],dtype=np.float32) if f['right'] else None,f['width'],f['height'],0)
        from core.features import feature
        try:r=feature(s,f['config'])
        except NotImplementedError:pytest.skip('正規化學員TODO未完成，非PASS')
        # Students choose D and which poses to support; only raw-invalid inputs
        # must be rejected. Recipe-specific numeric checks live in test_mixed_hands.
        if not f['valid']:
            assert not r.valid and r.features is None
        elif r.valid:
            assert r.features.ndim == 1 and r.features.size > 0
            assert r.features.dtype == np.float32 and np.isfinite(r.features).all()
            dimensions.add(r.features.size)
    assert len(dimensions) == 1, '支援的樣本須至少有一筆，且全部使用一致維度'

def test_all_hand_combinations():
    from core.features import raw_valid, feature
    hand=np.zeros((21,3),dtype=np.float32);hand[9]=[.1,.1,0]
    config={'feature_version':'hands-normalization-v3'}
    for left,right,valid in [(None,hand,True),(hand,None,True),(hand,hand,True),(None,None,False)]:
        sample=RawHandSample(left,right,640,480,0)
        assert raw_valid(sample,config)[0]==valid
        if not valid:assert not feature(sample,config).valid

def test_knn_contract():
    knn=StudentKNN();x=np.zeros((10,63),dtype=np.float32);y=np.arange(10);x[:,0]=y
    try:knn.fit(x,y,{'k':1})
    except NotImplementedError:pytest.skip('KNN學員TODO未完成，非PASS')
    from core.features import validate_prediction
    for i in range(10):assert validate_prediction(knn.predict(x[i]))!='NONE'

def test_optional_filter_contract():
    f=StudentFilter()
    try:f.reset()
    except NotImplementedError:pytest.skip('濾波選做TODO，預設未啟用')
    from core.labels import LABELS
    assert f.update(Prediction('NONE',False,None,0),0) in LABELS
