"""Exercise the published recipes and tool boundaries without filling student TODOs."""
import json
import re
from pathlib import Path

import numpy as np
import pytest

from core.dataset import save_batch, read_batch
from core.features import feature, raw_valid, OutputRouter, validate_prediction
from core.model_io import save_model, load_model, features_from, train, evaluate
from core.types import RawHandSample, FeatureResult, Prediction

CONFIG = {'feature_version': 'mixed-hands-131-v1', 'k': 1,
          'distance_threshold': 1.0, 'vote_threshold': 0.6}


def recipes():
    blocks = re.findall(r'```python\n(.*?)```',
                        (Path(__file__).parents[1] / 'README.md').read_text(encoding='utf-8'), re.S)
    found = []
    for block in blocks:
        if 'class StudentNormalizer:' in block or 'class StudentKNN:' in block:
            namespace = {}
            exec(block, namespace)
            found.append(namespace)
    basic = next(n['StudentNormalizer']() for n in found
                 if 'StudentNormalizer' in n and 131 not in n['StudentNormalizer'].transform.__code__.co_consts)
    mixed = next(n['StudentNormalizer']() for n in found
                 if 'StudentNormalizer' in n and 131 in n['StudentNormalizer'].transform.__code__.co_consts)
    knn = next(n['StudentKNN'] for n in found if 'StudentKNN' in n)
    return basic, mixed, knn


def hand(x=0.2):
    points = np.zeros((21, 3), dtype=np.float32)
    points[:] = [x, 0.3, 0]
    points[9] = [x + 0.1, 0.4, 0]
    return points


def samples():
    left, right = hand(), hand(0.6)
    return [RawHandSample(left, None, 640, 480, 0),
            RawHandSample(None, right, 640, 480, 0),
            RawHandSample(left, right, 640, 480, 0)]


def test_readme_basic_stays_right_only():
    basic, _, _ = recipes()
    for s, valid in zip(samples(), [False, True, False]):
        result = feature(s, CONFIG, basic)
        assert result.valid == valid
        if valid:
            points = s.right.astype(np.float64) * [640, 480, 640]
            centered = points - points[0]
            np.testing.assert_allclose(result.features,
                                       (centered / np.linalg.norm(centered[9])).reshape(63))


def test_readme_mixed_fixed_slots_relative_position_and_no_mutation():
    _, mixed, _ = recipes()
    ss = samples()
    before = ss[2].left.copy()
    results = [feature(s, CONFIG, mixed).features for s in ss]
    for row, presence in zip(results, [[2, 0], [0, 2], [2, 2]]):
        assert row.shape == (131,) and row.dtype == np.float32
        np.testing.assert_array_equal(row[126:128], presence)
    np.testing.assert_array_equal(results[0][63:126], 0)
    np.testing.assert_array_equal(results[1][:63], 0)
    np.testing.assert_array_equal(results[2][:63], results[0][:63])
    np.testing.assert_array_equal(results[2][63:126], results[1][63:126])
    np.testing.assert_allclose(results[2][128:131], [3.2, 0, 0], atol=1e-6)
    np.testing.assert_array_equal(ss[2].left, before)
    shifted = RawHandSample(ss[2].left + [0.05, 0.05, 0],
                            ss[2].right + [0.05, 0.05, 0], 640, 480, 0)
    np.testing.assert_allclose(feature(shifted, CONFIG, mixed).features, results[2], atol=1e-6)


def test_same_model_mixes_single_dual_end_to_end(tmp_path, monkeypatch):
    _, mixed, knn = recipes()
    monkeypatch.setattr('core.features.StudentNormalizer', lambda: mixed)
    monkeypatch.setattr('core.model_io.StudentKNN', knn)
    paths = []
    for purpose in ['train', 'test']:
        path = tmp_path / f'{purpose}.npz'
        save_batch(path, samples(), [0, 1, 4], purpose, CONFIG)
        paths.append(path)
        loaded, labels, _ = read_batch(path)
        assert labels.tolist() == [0, 1, 4]
        assert loaded[0].right is None and loaded[1].left is None
        assert loaded[2].left is not None and loaded[2].right is not None
    x, y, rejected = features_from(paths, CONFIG, 'train')
    assert x.shape == (3, 131) and rejected == 0
    path = tmp_path / 'knn.npz'
    assert train(paths, path, CONFIG)['training_samples'] == 3
    model = load_model(path, CONFIG)
    router = OutputRouter()
    for s, label in zip(samples(), ['MOVE_LEFT', 'MOVE_RIGHT', 'BLOCK']):
        pred = model.predict(feature(s, CONFIG, mixed).features)
        assert router.update(pred, tracked=True) == label
    assert router.update(tracked=False) == 'NONE'
    report = evaluate(paths, path, CONFIG)
    assert report['samples'] == 3 and report['unknown_ratio'] == 0
    assert report['confusion_matrix'][4][4] == 1
    with pytest.raises(ValueError, match='維度'):
        model.predict(np.zeros(63, dtype=np.float32))
    with pytest.raises(ValueError, match='不相容'):
        load_model(path, {**CONFIG, 'feature_version': 'changed'})
    metadata = json.loads(path.with_suffix('.json').read_text())
    metadata['feature_dimension'] = 63
    path.with_suffix('.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='格式'):
        load_model(path, CONFIG)


def test_missing_hand_does_not_match_dual_recipe():
    _, mixed, knn = recipes()
    ss = samples()
    model = knn()
    model.fit(feature(ss[2], CONFIG, mixed).features[None, :], np.array([4]), CONFIG)
    for s in ss[:2]:
        assert validate_prediction(model.predict(feature(s, CONFIG, mixed).features)) == 'NONE'


def test_published_recipes_pass_student_contracts(monkeypatch):
    from tests import test_student_contracts as contracts
    basic, mixed, knn = recipes()
    monkeypatch.setattr(contracts, 'StudentKNN', knn)
    contracts.test_knn_contract()
    for normalizer in [basic, mixed]:
        monkeypatch.setattr('core.features.StudentNormalizer', lambda: normalizer)
        contracts.test_normalization_contract()


def test_old_model_requires_retraining(tmp_path):
    path = tmp_path / 'old.npz'
    path.with_suffix('.json').write_text(json.dumps({'format': 1}))
    with pytest.raises(ValueError, match='重算並訓練'):
        load_model(path, CONFIG)


@pytest.mark.parametrize('dimension', [1, 63, 128, 131, 200])
def test_custom_feature_dimensions_roundtrip(dimension, tmp_path, monkeypatch):
    _, _, knn = recipes()
    monkeypatch.setattr('core.model_io.StudentKNN', knn)
    class Custom:
        def transform(self, sample, config):
            return FeatureResult(True, np.zeros(dimension, dtype=np.float32),
                                 featureVersion=config['feature_version'])
    vector = feature(samples()[0], CONFIG, Custom()).features
    path = tmp_path / 'model.npz'
    save_model(path, vector[None, :], np.array([0]), CONFIG)
    assert validate_prediction(load_model(path, CONFIG).predict(vector)) == 'MOVE_LEFT'


def test_varying_dimensions_rejected_before_training(tmp_path, monkeypatch):
    class Varying:
        def transform(self, sample, config):
            d = 63 if sample.left is None else 131
            return FeatureResult(True, np.zeros(d, dtype=np.float32),
                                 featureVersion=config['feature_version'])
    monkeypatch.setattr('core.features.StudentNormalizer', Varying)
    path = tmp_path / 'train.npz'
    save_batch(path, samples(), [0, 1, 4], 'train', CONFIG)
    with pytest.raises(ValueError, match='維度必須一致'):
        features_from([path], CONFIG, 'train')


@pytest.mark.parametrize('bad', [np.empty(0, dtype=np.float32),
                               np.zeros((2, 63), dtype=np.float32),
                               np.zeros(63, dtype=np.float64),
                               np.full(63, np.nan, dtype=np.float32)])
def test_feature_contract_rejects_invalid_vectors(bad):
    class Bad:
        def transform(self, sample, config):
            return FeatureResult(True, bad, featureVersion=config['feature_version'])
    with pytest.raises(ValueError):
        feature(samples()[0], CONFIG, Bad())


def test_bad_second_hand_rejects_raw_and_bypasses_student():
    s = samples()[2]
    s.left[9] = s.left[0]
    assert not raw_valid(s, CONFIG)[0]
    assert not feature(s, CONFIG).valid
    s.left = np.full((21, 3), np.nan, dtype=np.float32)
    assert not raw_valid(s, CONFIG)[0]


def test_one_action_only():
    with pytest.raises(ValueError):
        validate_prediction(Prediction(['MOVE_LEFT', 'BLOCK'], True, 0, 1))
    router = OutputRouter()
    assert router.update(Prediction(['MOVE_LEFT', 'BLOCK'], True, 0, 1), True) == 'NONE'


def test_config_preserves_feature_version_and_drops_obsolete_hand_switches(tmp_path, monkeypatch):
    import main
    monkeypatch.setattr(main, 'ROOT', tmp_path)
    (tmp_path / 'config.example.json').write_text(json.dumps(CONFIG))
    (tmp_path / 'config.json').write_text(json.dumps({**CONFIG, 'hand_mode': 'single',
                                                    'single_hand': 'right'}))
    assert main.config() == CONFIG
