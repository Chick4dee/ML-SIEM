"""Тесты stacking-меты: сборка признаков и обучаемая обёртка."""

import numpy as np

from mlsiem.detection.stacking import StackingMeta, build_meta_features


def test_meta_features_shape_and_columns():
    # 3 окна, 4 класса; проверяем, что на выходе ровно n_classes + 4 признака
    flow_probs = np.array([
        [0.7, 0.1, 0.1, 0.1],
        [0.25, 0.25, 0.25, 0.25],   # «растерянный» эксперт
        [0.0, 0.0, 0.0, 1.0],
    ])
    ae_scores = np.array([0.5, 2.0, 0.1])
    feats = build_meta_features(flow_probs, ae_scores, ae_threshold=1.0)

    assert feats.shape == (3, 4 + 4)
    # ae_ratio = score / threshold — для второго окна 2.0
    assert feats[1, 4] == 2.0


def test_entropy_reflects_confidence():
    # уверенный эксперт (one-hot) должен дать меньшую энтропию, чем «размазанный»
    confident = build_meta_features(np.array([[1.0, 0.0]]), np.array([0.0]), 1.0)
    unsure = build_meta_features(np.array([[0.5, 0.5]]), np.array([0.0]), 1.0)
    entropy_col = 2 + 2  # n_classes(2) + ae_ratio + flow_max → индекс энтропии
    assert confident[0, entropy_col] < unsure[0, entropy_col]


def test_ae_flag_is_binary():
    feats = build_meta_features(np.array([[0.5, 0.5], [0.5, 0.5]]),
                                np.array([0.2, 5.0]), ae_threshold=1.0)
    # последний столбец — флаг «AE сработал»
    assert feats[0, -1] == 0.0
    assert feats[1, -1] == 1.0


def test_fit_predict_save_load(tmp_path):
    rng = np.random.default_rng(0)
    # синтетика: класс зависит от того, «голосует» ли Flow за класс 1
    flow = rng.random((200, 2))
    flow = flow / flow.sum(axis=1, keepdims=True)
    ae = rng.random(200)
    labels = (flow[:, 1] > 0.5).astype(int)
    feats = build_meta_features(flow, ae, 0.5)

    meta = StackingMeta.fit(feats, labels, classes=["benign", "attack"], ae_threshold=0.5)
    proba = meta.predict_proba(feats)
    assert proba.shape == (200, 2)
    # на обучающих данных простую зависимость стэкинг обязан выучить почти идеально
    assert (proba.argmax(axis=1) == labels).mean() > 0.9

    loaded = StackingMeta.load(meta.save(tmp_path / "stack"))
    assert loaded.classes == meta.classes
    np.testing.assert_allclose(loaded.predict_proba(feats), proba)
