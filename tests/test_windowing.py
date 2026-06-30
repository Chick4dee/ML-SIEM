"""Юнит-тесты оконного модуля: маски, прогрев, порядок по времени, stride."""

import polars as pl
import pytest

from mlsiem.data.windowing import build_windows


def _df(rows: list[tuple[str, int, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "src_ip": [r[0] for r in rows],
            "bidirectional_first_seen_ms": [r[1] for r in rows],
            "f1": [r[2] for r in rows],
        }
    )


def test_warmup_windows_are_left_padded_and_masked():
    df = _df([("10.0.0.1", 100, 1.0), ("10.0.0.1", 200, 2.0), ("10.0.0.1", 300, 3.0)])
    windows, masks, last_rows = build_windows(df, ["f1"], window=4)

    assert windows.shape == (3, 4, 1)
    # первое окно: один реальный поток, три позиции паддинга слева
    assert masks[0].tolist() == [False, False, False, True]
    assert windows[0, 3, 0] == 1.0
    assert windows[0, :3, 0].tolist() == [0.0, 0.0, 0.0]
    # третье окно: три реальных потока в хронологическом порядке
    assert masks[2].tolist() == [False, True, True, True]
    assert windows[2, 1:, 0].tolist() == [1.0, 2.0, 3.0]


def test_groups_do_not_mix_and_label_index_points_to_last_flow():
    df = _df([("10.0.0.1", 100, 1.0), ("10.0.0.2", 50, 9.0), ("10.0.0.1", 200, 2.0)])
    windows, masks, last_rows = build_windows(df, ["f1"], window=2)

    assert windows.shape[0] == 3
    # окно, оканчивающееся потоком строки 2 (f1=2.0), содержит только потоки 10.0.0.1
    i = list(last_rows).index(2)
    assert windows[i, :, 0].tolist() == [1.0, 2.0]
    # одиночный поток 10.0.0.2 не смешан с чужими
    j = list(last_rows).index(1)
    assert masks[j].tolist() == [False, True]
    assert windows[j, 1, 0] == 9.0


def test_sorted_by_event_time_not_input_order():
    df = _df([("10.0.0.1", 300, 3.0), ("10.0.0.1", 100, 1.0), ("10.0.0.1", 200, 2.0)])
    windows, _, last_rows = build_windows(df, ["f1"], window=3)

    full = windows[-1, :, 0].tolist()
    assert full == [1.0, 2.0, 3.0]
    # метка последнего окна — у самого позднего потока (строка 0, время 300)
    assert last_rows[-1] == 0


def test_stride_and_min_context():
    df = _df([("10.0.0.1", t, float(t)) for t in range(100, 700, 100)])  # 6 потоков
    windows, masks, _ = build_windows(df, ["f1"], window=4, stride=2, min_context=2)

    # концы окон: индексы 1, 3, 5 → три окна
    assert windows.shape[0] == 3
    assert masks[0].sum() == 2  # прогрев: 2 реальных потока


def test_empty_input():
    df = _df([])
    windows, masks, last_rows = build_windows(df, ["f1"], window=4)
    assert windows.shape == (0, 4, 1)
    assert masks.shape == (0, 4)
    assert last_rows.shape == (0,)


def test_invalid_args():
    df = _df([("10.0.0.1", 100, 1.0)])
    with pytest.raises(ValueError):
        build_windows(df, [], window=4)
    with pytest.raises(ValueError):
        build_windows(df, ["f1"], window=4, min_context=5)
