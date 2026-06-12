"""Нарезка потоков в окна-последовательности для CNN+LSTM.

Группировка по источнику (src_ip), сортировка по event-time самого потока —
не по времени прихода (PLAN.md, раздел 4.7). Неполные окна прогрева нового
источника дополняются нулями слева и помечаются маской: модель обязана
игнорировать паддинг, а не трактовать его как трафик (раздел 10.5, #7).
Одна и та же нарезка используется в batch-обучении и live-стриме.
"""

import numpy as np
import polars as pl

DEFAULT_WINDOW = 16
DEFAULT_STRIDE = 1

_ROW_COL = "_row_idx"


def build_windows(
    df: pl.DataFrame,
    feature_cols: list[str],
    *,
    window: int = DEFAULT_WINDOW,
    stride: int = DEFAULT_STRIDE,
    group_col: str = "src_ip",
    time_col: str = "bidirectional_first_seen_ms",
    min_context: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Режет потоки в маскированные окна.

    Каждый поток группы (начиная с min_context-го, с шагом stride) становится
    концом окна из window последних потоков того же источника; нехватка
    заполняется нулями слева.

    Возвращает (windows, masks, last_rows):
    - windows: float32 [n, window, n_features]
    - masks:   bool    [n, window], True = реальный поток, False = паддинг
    - last_rows: int64 [n] — индекс строки df последнего потока окна;
      метка окна = метка этого потока (PLAN.md, раздел 4)
    """
    if not feature_cols:
        raise ValueError("feature_cols не может быть пустым")
    if min_context < 1 or min_context > window:
        raise ValueError("min_context должен быть в пределах [1, window]")

    n_features = len(feature_cols)
    windows: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    last_rows: list[int] = []

    indexed = df.with_row_index(_ROW_COL)
    for _, group in indexed.group_by(group_col):
        g = group.sort(time_col)
        feats = g.select(feature_cols).to_numpy().astype(np.float32)
        rows = g[_ROW_COL].to_numpy()
        for end in range(min_context - 1, len(g), stride):
            start = max(0, end - window + 1)
            chunk = feats[start : end + 1]
            pad = window - len(chunk)
            if pad:
                chunk = np.vstack([np.zeros((pad, n_features), np.float32), chunk])
            mask = np.arange(window) >= pad
            windows.append(chunk)
            masks.append(mask)
            last_rows.append(int(rows[end]))

    if not windows:
        return (
            np.empty((0, window, n_features), np.float32),
            np.empty((0, window), bool),
            np.empty(0, np.int64),
        )
    return np.stack(windows), np.stack(masks), np.asarray(last_rows, np.int64)
