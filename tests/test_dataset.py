"""Тесты torch-датасета: паритет окон с build_windows, метки, маски, энкодер."""

import numpy as np
import polars as pl
import pytest

from mlsiem.features.windowing import build_windows
from mlsiem.models.dataset import FlowWindowDataset, LabelEncoder


def _df(rows: list[tuple]) -> pl.DataFrame:
    cols = ["src_ip", "bidirectional_first_seen_ms", "f1", "f2", "canonical"]
    return pl.DataFrame([dict(zip(cols, r, strict=True)) for r in rows])


def test_label_encoder_deterministic_and_roundtrip(tmp_path):
    enc = LabelEncoder.fit(["benign", "dos", "benign", "worms"])
    assert enc.classes == ["benign", "dos", "worms"]  # отсортировано
    assert enc.encode("dos") == 1
    assert len(enc) == 3
    loaded = LabelEncoder.load(enc.save(tmp_path / "enc.json"))
    assert loaded.classes == enc.classes


def test_windows_parity_with_build_windows():
    # датасет должен давать те же окна/маски, что эталонный build_windows
    rows = [
        ("10.0.0.1", 100, 1.0, 0.1, "benign"),
        ("10.0.0.1", 200, 2.0, 0.2, "dos"),
        ("10.0.0.1", 300, 3.0, 0.3, "benign"),
        ("10.0.0.2", 50, 9.0, 0.9, "worms"),
    ]
    df = _df(rows)
    feature_cols = ["f1", "f2"]
    enc = LabelEncoder.fit(df["canonical"])

    windows, masks, last_rows = build_windows(df, feature_cols, window=4)
    ds = FlowWindowDataset(df, feature_cols, enc, window=4)

    assert len(ds) == len(windows)
    # сверяем поэлементно (порядок групп может отличаться → матчим по содержимому)
    got = {tuple(np.round(ds[i][0].numpy().ravel(), 5)): ds[i] for i in range(len(ds))}
    for w, m, lr in zip(windows, masks, last_rows, strict=True):
        key = tuple(np.round(w.ravel(), 5))
        assert key in got
        gw, gm, gl = got[key]
        np.testing.assert_array_equal(gw.numpy(), w)
        np.testing.assert_array_equal(gm.numpy(), m)
        # метка окна = метка последнего потока (по строке last_rows в df)
        expected_label = enc.encode(df["canonical"][int(lr)])
        assert int(gl) == expected_label


def test_getitem_shapes_and_mask_dtype():
    df = _df([("10.0.0.1", t, float(t), float(t) / 10, "benign") for t in range(100, 600, 100)])
    enc = LabelEncoder.fit(df["canonical"])
    ds = FlowWindowDataset(df, ["f1", "f2"], enc, window=8)

    chunk, mask, label = ds[0]
    assert chunk.shape == (8, 2)
    assert mask.shape == (8,)
    assert mask.dtype.is_floating_point is False
    assert label.dtype.__str__() == "torch.int64"


def test_min_context_validation():
    df = _df([("10.0.0.1", 100, 1.0, 0.1, "benign")])
    enc = LabelEncoder.fit(df["canonical"])
    with pytest.raises(ValueError):
        FlowWindowDataset(df, ["f1", "f2"], enc, window=4, min_context=5)
