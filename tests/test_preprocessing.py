"""Тесты препроцессинга: отбор фич, fitted-артефакты, OOV, временной сплит."""

import numpy as np
import polars as pl
import pytest

from mlsiem.data.preprocessing import (
    CATEGORICAL,
    EXCLUDED,
    FeaturePreprocessor,
    select_feature_columns,
    time_split,
)


def _df(n: int = 100) -> pl.DataFrame:
    return pl.DataFrame({
        "bidirectional_first_seen_ms": list(range(n)),
        "src_ip": ["10.0.0.1"] * n,            # идентификатор — не фича
        "dst_port": [80] * n,                  # идентификатор — не фича
        "bidirectional_bytes": [float(i) for i in range(n)],  # числовая фича
        "bidirectional_packets": [float(i % 7) for i in range(n)],
        "application_name": ["HTTP" if i % 2 else "DNS" for i in range(n)],  # категория
        "application_category_name": ["Web"] * n,
        "label_class": ["benign"] * n,         # таргет — не фича
    })


def test_select_excludes_identifiers_time_targets_and_categoricals():
    cols = select_feature_columns(_df())
    assert "bidirectional_bytes" in cols
    assert "bidirectional_packets" in cols
    for forbidden in ("src_ip", "dst_port", "bidirectional_first_seen_ms", "label_class"):
        assert forbidden in EXCLUDED
        assert forbidden not in cols
    for cat in CATEGORICAL:
        assert cat not in cols


def test_transform_standardizes_numeric_to_zero_mean_unit_std():
    df = _df()
    pre = FeaturePreprocessor.fit(df)
    out = pre.transform(df)

    # на train-данных нормализованные числовые ~ нулевое среднее, единичный std
    # (std выборочный, ddof=1 — как в polars при фите)
    col = out["bidirectional_bytes"].to_numpy()
    assert abs(col.mean()) < 1e-6
    assert abs(col.std(ddof=1) - 1.0) < 1e-3


def test_categorical_encoded_with_oov_as_zero():
    train = _df()
    pre = FeaturePreprocessor.fit(train)

    serve = train.head(2).with_columns(
        pl.Series("application_name", ["HTTP", "QUIC_NEVER_SEEN"])
    )
    out = pre.transform(serve)
    codes = out["application_name"].to_list()
    assert codes[0] >= 1            # известная категория
    assert codes[1] == 0            # OOV → 0


def test_save_load_roundtrip_identical_transform(tmp_path):
    df = _df()
    pre = FeaturePreprocessor.fit(df)
    path = pre.save(tmp_path / "pre.json")
    loaded = FeaturePreprocessor.load(path)

    a = pre.transform(df).to_numpy()
    b = loaded.transform(df).to_numpy()
    np.testing.assert_array_equal(a, b)
    assert loaded.feature_names == pre.feature_names


def test_fit_on_train_only_then_apply_to_serve_uses_train_stats():
    train = _df(50)
    pre = FeaturePreprocessor.fit(train)
    means_before = pre.means.copy()
    # применение к другим данным не меняет артефакты
    serve = _df(10)
    pre.transform(serve)
    np.testing.assert_array_equal(pre.means, means_before)


def test_time_split_is_chronological_no_overlap():
    df = _df(100)
    train, val = time_split(df, val_frac=0.2)
    assert len(train) == 80
    assert len(val) == 20
    assert train["bidirectional_first_seen_ms"].max() < val["bidirectional_first_seen_ms"].min()


def test_time_split_validates_frac():
    with pytest.raises(ValueError):
        time_split(_df(), val_frac=0.0)
    with pytest.raises(ValueError):
        time_split(_df(), val_frac=1.0)
