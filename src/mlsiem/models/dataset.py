"""Torch-датасет: размеченные потоки → маскированные окна-последовательности.

Окна нарезаются ЛЕНИВО: при stride=1 на 2M потоков плотный массив всех окон
(2M × N × F) не влезает в память, поэтому храним потоки один раз (сгруппированы
по src_ip, отсортированы по event-time) + индекс окон, а само окно собираем в
__getitem__. Семантика идентична features.windowing.build_windows (паритет
закреплён тестом): левый паддинг нулями, маска True=реальный поток, метка окна =
метка последнего потока (PLAN.md, раздел 4).

Сплит train/val делается ДО построения датасета (по времени, без ликеджа):
окно живёт внутри своей части и не пересекает границу.
"""

import json
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from mlsiem.features.windowing import DEFAULT_STRIDE, DEFAULT_WINDOW


class LabelEncoder:
    """Канонический класс (строка) ↔ целочисленный id. Порядок детерминирован."""

    def __init__(self, classes: list[str]):
        self.classes = list(classes)
        self._to_id = {c: i for i, c in enumerate(self.classes)}

    @classmethod
    def fit(cls, labels) -> "LabelEncoder":
        uniq = sorted(set(labels))
        return cls(uniq)

    def encode(self, label: str) -> int:
        return self._to_id[label]

    def __len__(self) -> int:
        return len(self.classes)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"classes": self.classes}, ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "LabelEncoder":
        return cls(json.loads(Path(path).read_text(encoding="utf-8"))["classes"])


class FlowWindowDataset(Dataset):
    """Ленивая нарезка окон поверх уже препроцессированных фич.

    df: фичи (числовые + категориальные индексы) + колонки group_col, time_col,
    label_col. feature_cols — порядок фич в тензоре окна.
    """

    def __init__(
        self,
        df: pl.DataFrame,
        feature_cols: list[str],
        encoder: LabelEncoder,
        *,
        label_col: str = "canonical",
        group_col: str = "src_ip",
        time_col: str = "bidirectional_first_seen_ms",
        window: int = DEFAULT_WINDOW,
        stride: int = DEFAULT_STRIDE,
        min_context: int = 1,
    ):
        if not 1 <= min_context <= window:
            raise ValueError("min_context должен быть в [1, window]")
        self.window = window
        self.n_features = len(feature_cols)
        # маски на каждое число левого паддинга (0..window-1) — считаем один раз
        self._masks = [np.arange(window) >= p for p in range(window)]

        # Потоки храним по группам (src_ip), отсортированными по event-time.
        self._group_feats: list[np.ndarray] = []
        self._group_labels: list[np.ndarray] = []
        self._index: list[tuple[int, int]] = []  # (group_idx, end_pos)
        win_labels: list[int] = []                # метка каждого окна (= метка end)

        for _, g in df.group_by(group_col):
            g = g.sort(time_col)
            feats = g.select(feature_cols).to_numpy().astype(np.float32)
            labels = np.array([encoder.encode(x) for x in g[label_col]], dtype=np.int64)
            gi = len(self._group_feats)
            self._group_feats.append(feats)
            self._group_labels.append(labels)
            for end in range(min_context - 1, len(g), stride):
                self._index.append((gi, end))
                win_labels.append(int(labels[end]))
        self.window_labels = np.asarray(win_labels, dtype=np.int64)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int):
        gi, end = self._index[i]
        feats = self._group_feats[gi]
        start = max(0, end - self.window + 1)
        real = feats[start : end + 1]
        pad = self.window - len(real)
        if pad:
            chunk = np.zeros((self.window, self.n_features), np.float32)
            chunk[pad:] = real
        else:
            chunk = real  # полное окно — без копирования (частый случай)
        mask = self._masks[pad]
        label = int(self._group_labels[gi][end])
        return (
            torch.from_numpy(np.ascontiguousarray(chunk)),
            torch.from_numpy(mask),
            torch.tensor(label, dtype=torch.long),
        )
